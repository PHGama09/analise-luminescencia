"""Programa 2 - Detecção da amostra (ROI).

Acha, em cada vídeo, o círculo da amostra. Tudo automático: nada precisa ser clicado.

Como a ROI é achada
  Depois que a luz de excitação apaga (a queda medida fica entre 0,20 e 0,40 s), o único brilho do quadro é o da
  própria amostra. A média dos quadros NOVOS (repetidos não contam) entre 0,6 e 1,5 s é suavizada, binarizada
  (Otsu) e o maior objeto vira o círculo. A ROI usada é 88% do raio detectado, para fugir da borda borrada.

  O vídeo de controle (sem brilho) herda a ROI mediana dos outros vídeos.

Quando um vídeo é "suspeito" (regras objetivas, cada uma com o motivo escrito)
  forma pouco circular, contraste baixo, objeto encostado na borda, área fora do esperado, raio diferente da
  mediana do lote, ROI herdada porque a detecção falhou, ou controle com brilho.
  O CENTRO não é critério: o recipiente muda de posição entre vídeos (medido: até ~0,3 raio).

Quais vídeos entram: os do curadoria.csv com incluir = sim ou pendente (nao é pulado). Sem curadoria.csv, todos
os da pasta (menos as pastas ignoradas). Nada é alterado nos vídeos nem no curadoria.csv.

Saídas (output/roi/)
  roi.json            uma entrada por vídeo (a etapa 3 lê este arquivo)
  roi.csv             o mesmo, em tabela, para olhar no Excel
  mosaico_*.png       folhas de conferência (as primeiras reúnem os suspeitos); círculo verde = usado,
                      amarelo = detectado; o controle aparece SEM esticar o contraste
  miniaturas/*.png    imagem média de cada vídeo (permite refazer o mosaico sem ler os vídeos)

Reexecutar só lê de novo os vídeos novos ou alterados. Para corrigir um vídeo que a detecção errou (exceção),
edite a entrada em roi.json (campo "roi": cx, cy, r) e ponha "manual": true: ela passa a ser preservada.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .auditoria import EXTENSOES_PADRAO, listar_videos
from .nomes import ler_nome

VERSAO_ROI = "1.0"

PADRAO = {
    "janela_s": [0.6, 1.5],          # quadros usados para achar a ROI (a luz já apagou: a queda máxima medida é 0,40 s)
    "margem_interna": 0.12,          # a ROI usa 88% do raio detectado
    "circularidade_minima": 0.85,
    "contraste_minimo": 3.0,         # brilho dentro / fora do círculo
    "desvio_raio_lote": 0.25,        # raio diferente da mediana do lote por mais que isso = suspeito
    "pico_minimo": 4.0,              # abaixo disso a janela não tem brilho de amostra (controle)
    "workers": 6,
    "mosaico": {"colunas": 4, "linhas": 6, "largura_tile": 480},
}


def parametros(cfg: dict) -> dict:
    p = {**PADRAO, **(cfg.get("roi") or {})}
    p["mosaico"] = {**PADRAO["mosaico"], **((cfg.get("roi") or {}).get("mosaico") or {})}
    return p


# ------------------------------------------------------------------ escolha dos vídeos
def _item(caminho: str, pasta: Path, nomes: dict, incluir: str, status: str, chave: str) -> dict:
    campos, _ = ler_nome(Path(caminho).name, nomes)
    return {"caminho": caminho, "arquivo": pasta / caminho, "controle": bool(campos and campos.get("tipo") == "controle"),
            "incluir": incluir, "status_auditoria": status, "chave": chave}


def selecionar_videos(cfg: dict, pasta: Path) -> tuple[list[dict], list[str], int]:
    """Vídeos a processar, na ordem do caminho. Retorna (itens, avisos, quantos foram pulados por incluir = nao).

    Lê o curadoria.csv: pula incluir = nao e vídeos ausentes; sim e pendente entram. Se um conteúdo existe em
    dois caminhos (cópia idêntica), usa o de nome sem espaço."""
    nomes = cfg.get("nomes") or {}
    avisos: list[str] = []
    arquivo = Path((cfg.get("curadoria") or {}).get("arquivo", "curadoria.csv"))
    itens, pulados = [], 0
    if not arquivo.exists():
        avisos.append(f"{arquivo} não encontrado: usando todos os vídeos da pasta")
        ignorar = (cfg.get("auditoria") or {}).get("pastas_ignoradas") or []
        for v in listar_videos(pasta, nomes.get("extensoes", EXTENSOES_PADRAO), ignorar):
            itens.append(_item(v.relative_to(pasta).as_posix(), pasta, nomes, "pendente", "", ""))
    else:
        with arquivo.open(encoding="utf-8-sig", newline="") as f:
            linhas = list(csv.DictReader(f))
        for l in linhas:
            incluir = (l.get("incluir") or "pendente").strip().lower().replace("não", "nao")
            if incluir == "nao":
                pulados += 1
                continue
            if l.get("status_auditoria") == "ausente":
                continue
            caminhos = [c.strip() for c in (l.get("caminho") or "").split(" | ") if c.strip()]
            existentes = [c for c in caminhos if (pasta / c).exists()]
            if not existentes:
                avisos.append(f"vídeo não encontrado em {pasta}: {' | '.join(caminhos)}")
                continue
            escolhido = sorted(existentes, key=lambda c: (len(re.findall(r"\s", Path(c).name)), c))[0]
            itens.append(_item(escolhido, pasta, nomes, incluir, l.get("status_auditoria", ""), l.get("chave", "")))
    itens.sort(key=lambda i: i["caminho"])
    return itens, avisos, pulados


# ------------------------------------------------------------------ detecção
def _kernel(n):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))


def media_da_janela(caminho: Path, fps: float, janela_s) -> tuple[np.ndarray | None, int]:
    """Média (BGR, float32) dos quadros NOVOS na janela de tempo. Retorna (média, quantos quadros)."""
    i0, i1 = int(round(janela_s[0] * fps)), int(round(janela_s[1] * fps))
    cap = cv2.VideoCapture(str(caminho))
    soma, n, anterior = None, 0, None
    for i in range(i1 + 1):
        ok, q = cap.read()
        if not ok:
            break
        novo = anterior is None or not np.array_equal(q, anterior)
        anterior = q
        if i >= i0 and novo:
            soma = q.astype(np.float32) if soma is None else soma + q
            n += 1
    cap.release()
    if soma is None or n == 0:
        return None, 0
    return soma / n, n


def detectar_roi(media: np.ndarray, p: dict) -> dict:
    """Acha o círculo da amostra numa imagem média. Retorna cx, cy, r (já com margem) e diagnóstico."""
    h, w = media.shape[:2]
    cinza = cv2.GaussianBlur(media.mean(axis=2), (0, 0), 5)
    pico, fundo = float(np.percentile(cinza, 99.5)), float(np.median(cinza))
    res = {"achou": False, "motivos": [], "pico": round(pico, 1), "fundo": round(fundo, 2)}
    if pico < p["pico_minimo"]:
        res["motivos"].append("sem brilho na janela (controle ou vídeo sem amostra luminescente)")
        return res
    u8 = np.clip(cinza / pico * 255, 0, 255).astype(np.uint8)
    limiar, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mascara = (u8 > limiar).astype(np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, _kernel(15))
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        res["motivos"].append("nenhum objeto encontrado")
        return res
    maior = max(contornos, key=cv2.contourArea)
    area, perim = float(cv2.contourArea(maior)), float(cv2.arcLength(maior, True))
    if area < 50 or perim <= 0:
        res["motivos"].append("objeto pequeno demais")
        return res
    m = cv2.moments(maior)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    r_eq = float(np.sqrt(area / np.pi))
    x, y, bw, bh = cv2.boundingRect(maior)
    dentro = np.zeros((h, w), np.uint8)
    cv2.circle(dentro, (int(round(cx)), int(round(cy))), int(round(r_eq)), 1, -1)
    media_dentro = float(cinza[dentro > 0].mean())
    media_fora = float(cinza[dentro == 0].mean())
    circ = 4 * np.pi * area / perim ** 2
    contraste = media_dentro / max(media_fora, 0.5)
    res.update(achou=True, cx=float(cx), cy=float(cy), r_detectado=r_eq, r=r_eq * (1 - p["margem_interna"]),
               circularidade=round(float(circ), 3), contraste=round(float(contraste), 1),
               fracao_area=round(area / (h * w), 4), limiar=round(float(limiar), 1))
    if circ < p["circularidade_minima"]:
        res["motivos"].append(f"forma pouco circular ({circ:.2f})")
    if contraste < p["contraste_minimo"]:
        res["motivos"].append(f"contraste baixo ({contraste:.1f})")
    if x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2:
        res["motivos"].append("objeto encosta na borda do quadro")
    if not 0.003 <= area / (h * w) <= 0.5:
        res["motivos"].append(f"área fora do esperado ({area / (h * w):.1%} do quadro)")
    return res


def comparar_com_lote(rois: dict, p: dict) -> dict | None:
    """Marca raios que fogem da mediana do lote e devolve a ROI mediana (para o controle e para as falhas).

    Só o raio indica detecção errada: o centro varia entre vídeos porque o recipiente muda de posição.
    O desvio do centro fica registrado, sem virar suspeita."""
    achadas = {k: v for k, v in rois.items() if v.get("achou")}
    if not achadas:
        return None
    med = {c: float(np.median([v[c] for v in achadas.values()])) for c in ("cx", "cy", "r_detectado")}
    for v in achadas.values():
        dr = abs(v["r_detectado"] - med["r_detectado"]) / med["r_detectado"]
        dc = float(np.hypot(v["cx"] - med["cx"], v["cy"] - med["cy"])) / med["r_detectado"]
        v["desvio_raio"], v["desvio_centro_raios"] = round(dr, 3), round(dc, 3)
        if dr > p["desvio_raio_lote"]:
            v["motivos"].append(f"raio {dr:.0%} diferente da mediana do lote")
    return {"cx": med["cx"], "cy": med["cy"], "r": med["r_detectado"] * (1 - p["margem_interna"]),
            "r_detectado": med["r_detectado"]}


def _assinatura(p: dict) -> str:
    base = {k: p[k] for k in ("janela_s", "margem_interna", "circularidade_minima", "contraste_minimo", "pico_minimo")}
    base["versao"] = VERSAO_ROI
    return hashlib.sha1(json.dumps(base, sort_keys=True).encode()).hexdigest()[:12]


def _nome_miniatura(caminho: str) -> str:
    return re.sub(r"[\\/:]", "__", Path(caminho).with_suffix("").as_posix()) + ".png"


def _analisar_video(item: dict, fps_cfg: float, p: dict, pasta_min: Path) -> dict:
    """Lê o início do vídeo, detecta a ROI e grava a miniatura. Retorna a entrada parcial (sem o passo do lote)."""
    arq = item["arquivo"]
    st = arq.stat()
    cap = cv2.VideoCapture(str(arq))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or fps_cfg
    cap.release()
    media, n = media_da_janela(arq, fps, p["janela_s"])
    ent = {"tamanho": st.st_size, "mtime": st.st_mtime, "fps": round(fps, 3), "miniatura": _nome_miniatura(item["caminho"])}
    if media is None:
        ent.update(det={"achou": False, "motivos": ["não foi possível ler quadros na janela da ROI"]}, largura=0, altura=0)
        return ent
    det = detectar_roi(media, p)
    det["quadros_novos_na_janela"] = n
    det["pico_imagem"] = float(np.percentile(media, 99.5))
    h, w = media.shape[:2]
    ent.update(det=det, largura=w, altura=h)
    mini = cv2.resize(np.clip(media, 0, 255).round().astype(np.uint8), (640, int(round(640 * h / w))), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(pasta_min / ent["miniatura"]), mini)
    return ent


# ------------------------------------------------------------------ mosaico
def _ascii(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")


def _tile(e: dict, pasta_min: Path, largura: int, pico_ref: float) -> np.ndarray:
    im = cv2.imread(str(pasta_min / e["miniatura"]))
    w, h = max(e.get("largura") or 1280, 1), max(e.get("altura") or 720, 1)
    th = int(round(largura * h / w))
    if im is None:
        im = np.zeros((int(round(640 * h / w)), 640, 3), np.uint8)
    esc = im.shape[1] / w
    det = e["det"]
    sem_esticar = e.get("controle") and not det.get("achou")
    ref = pico_ref if sem_esticar else max(float(det.get("pico_imagem", 1.0)), 1.0)
    im = np.clip(im.astype(np.float32) / ref * 255, 0, 255).astype(np.uint8)
    if det.get("achou"):
        cv2.circle(im, (int(det["cx"] * esc), int(det["cy"] * esc)), int(det["r_detectado"] * esc), (0, 255, 255), 1)
    if e.get("roi"):
        r = e["roi"]
        cv2.circle(im, (int(r["cx"] * esc), int(r["cy"] * esc)), int(r["r"] * esc), (0, 255, 0), 2)
    im = cv2.resize(im, (largura, th), interpolation=cv2.INTER_AREA)
    fonte = largura / 800
    cv2.putText(im, _ascii(Path(e["caminho"]).name)[:44], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * fonte + 0.15, (255, 255, 255), 1, cv2.LINE_AA)
    notas = []
    if sem_esticar:
        notas.append((f"SEM esticar contraste (pico real {det.get('pico', 0):.1f} de 255)", (0, 200, 255)))
    if e.get("origem", "").startswith("herdada") or e.get("origem") == "manual":
        notas.append((_ascii(f"ROI {e['origem'].upper()}"), (0, 200, 255)))
    if e.get("suspeito"):
        notas.append((_ascii("SUSPEITO: " + "; ".join(e["motivos"]))[:70], (60, 60, 255)))
    for k, (txt, cor) in enumerate(reversed(notas)):
        cv2.putText(im, txt, (6, th - 8 - 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * fonte + 0.1, cor, 1, cv2.LINE_AA)
    return im


def gerar_mosaicos(entradas: list[dict], pasta_roi: Path, p: dict, so_suspeitos: bool = False) -> list[Path]:
    """Folhas de conferência. As primeiras reúnem os suspeitos; as demais seguem a ordem do caminho."""
    m = p["mosaico"]
    cols, linhas, larg = int(m["colunas"]), int(m["linhas"]), int(m["largura_tile"])
    pasta_min = pasta_roi / "miniaturas"
    picos = [e["det"]["pico_imagem"] for e in entradas if e["det"].get("achou") and "pico_imagem" in e["det"]]
    pico_ref = float(np.median(picos)) if picos else 100.0
    for velho in pasta_roi.glob("mosaico_*.png"):
        velho.unlink()
    grupos = [("mosaico_suspeitos", [e for e in entradas if e.get("suspeito")])]
    if not so_suspeitos:
        grupos.append(("mosaico", [e for e in entradas if not e.get("suspeito")]))
    gravados = []
    for prefixo, lista in grupos:
        por_folha = cols * linhas
        for k in range(0, len(lista), por_folha):
            tiles = [_tile(e, pasta_min, larg, pico_ref) for e in lista[k:k + por_folha]]
            th = max(t.shape[0] for t in tiles)
            tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
            while len(tiles) % cols:
                tiles.append(np.zeros((th, larg, 3), np.uint8))
            folha = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
            caminho = pasta_roi / f"{prefixo}_{k // por_folha + 1:02d}.png"
            cv2.imwrite(str(caminho), folha)
            gravados.append(caminho)
    return gravados


# ------------------------------------------------------------------ saída
def _gravar_json(caminho: Path, dados: dict):
    tmp = caminho.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dados, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    tmp.replace(caminho)


def _gravar_csv(caminho: Path, videos: dict):
    campos = ["caminho", "controle", "incluir", "status_auditoria", "origem", "suspeito", "motivos", "cx", "cy", "r",
              "r_detectado", "circularidade", "contraste", "pico", "fundo", "desvio_raio", "desvio_centro_raios", "quadros_novos_na_janela"]
    with caminho.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        for c, e in videos.items():
            det = e.get("det", {})
            linha = {"caminho": c, **{k: e.get(k, "") for k in ("controle", "incluir", "status_auditoria", "origem", "suspeito")},
                     "motivos": " | ".join(e.get("motivos", []))}
            linha.update({k: round(det[k], 2) if isinstance(det.get(k), float) else det.get(k, "")
                          for k in ("r_detectado", "circularidade", "contraste", "pico", "fundo", "desvio_raio", "desvio_centro_raios", "quadros_novos_na_janela")})
            roi = e.get("roi") or {}
            linha.update({k: round(roi[k], 2) if k in roi else "" for k in ("cx", "cy", "r")})
            w.writerow(linha)


def rodar(cfg: dict, pasta=None, saida=None, refazer: bool = False, workers: int | None = None,
          so_mosaico: bool = False, so_suspeitos: bool = False) -> int:
    inicio = time.perf_counter()
    p = parametros(cfg)
    prot = cfg.get("protocolo") or {}
    pasta = Path(pasta or (cfg.get("pastas") or {}).get("videos", "data"))
    if not pasta.is_dir():
        raise NotADirectoryError(f"Pasta de vídeos não encontrada: {pasta.resolve()}")
    pasta_roi = Path(saida or Path((cfg.get("pastas") or {}).get("saida", "output")) / "roi")
    (pasta_roi / "miniaturas").mkdir(parents=True, exist_ok=True)
    json_path = pasta_roi / "roi.json"
    try:
        antigo = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}
    except (OSError, ValueError):
        antigo = {}
    velhos = antigo.get("videos", {})

    if so_mosaico:
        if not velhos:
            raise FileNotFoundError(f"{json_path} não existe: rode 'luminescencia roi' antes")
        entradas = [{"caminho": c, **e} for c, e in velhos.items()]
        folhas = gerar_mosaicos(entradas, pasta_roi, p, so_suspeitos)
        print(f"Mosaico refeito a partir das miniaturas: {len(folhas)} folha(s) em {pasta_roi}")
        return 0

    itens, avisos, pulados = selecionar_videos(cfg, pasta)
    if not itens:
        print(f"Nenhum vídeo para processar em {pasta.resolve()}")
        return 1
    assinatura = _assinatura(p)
    workers = max(1, int(workers or p["workers"]))
    fps_cfg = float(prot.get("fps_esperado", 15))

    # ---- 1) detecção (só o que é novo ou mudou)
    ents, pendentes, do_cache = {}, [], 0
    for it in itens:
        c, antigo_e = it["caminho"], velhos.get(it["caminho"])
        st = it["arquivo"].stat()
        if antigo_e and antigo_e.get("manual"):
            ents[c] = deepcopy(antigo_e)
            do_cache += 1
        elif (antigo_e and not refazer and antigo_e.get("assinatura") == assinatura and antigo_e.get("tamanho") == st.st_size
              and abs(antigo_e.get("mtime", 0) - st.st_mtime) < 1 and (pasta_roi / "miniaturas" / antigo_e.get("miniatura", "?")).exists()):
            ents[c] = {k: antigo_e[k] for k in ("tamanho", "mtime", "fps", "miniatura", "det", "largura", "altura")}
            do_cache += 1
        else:
            pendentes.append(it)
    erros = []
    if pendentes:
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            futuros = {pool.submit(_analisar_video, it, fps_cfg, p, pasta_roi / "miniaturas"): it for it in pendentes}
            for f in tqdm(as_completed(futuros), total=len(futuros), desc="ROI", unit="vídeo"):
                it = futuros[f]
                try:
                    ents[it["caminho"]] = f.result()
                except Exception as erro:                     # um vídeo ruim não derruba o lote
                    erros.append(f"{it['caminho']}: {type(erro).__name__}: {erro}")
                    ents[it["caminho"]] = {"tamanho": 0, "mtime": 0, "fps": fps_cfg, "miniatura": "", "largura": 0, "altura": 0,
                                           "det": {"achou": False, "motivos": [f"erro ao ler o vídeo: {erro}"]}}
            pool.shutdown()
        except KeyboardInterrupt:
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    # ---- 2) comparação com o lote e ROI final
    por_item = {it["caminho"]: it for it in itens}
    dets = {c: deepcopy(e["det"]) for c, e in ents.items() if not e.get("manual") and not por_item[c]["controle"]}
    mediana = comparar_com_lote(dets, p)
    finais: dict[str, dict] = {}
    for c, it in por_item.items():
        e = ents[c]
        if e.get("manual"):
            e.update(controle=it["controle"], incluir=it["incluir"], status_auditoria=it["status_auditoria"], chave=it["chave"])
            finais[c] = e
            continue
        det = dets.get(c) or deepcopy(e["det"])
        motivos = list(det["motivos"])
        roi, origem = None, "sem ROI"
        if it["controle"]:
            if mediana:
                roi, origem = {k: mediana[k] for k in ("cx", "cy", "r")}, "herdada (mediana do lote)"
            if det.get("achou"):
                motivos.append(f"controle com brilho (pico {det['pico']:.1f}): conferir se está mesmo no escuro")
        elif det.get("achou"):
            roi, origem = {k: det[k] for k in ("cx", "cy", "r")}, "detectada"
        elif mediana:
            roi, origem = {k: mediana[k] for k in ("cx", "cy", "r")}, "herdada (mediana do lote)"
            motivos.append("ROI herdada da mediana porque a detecção falhou")
        controle_no_escuro = it["controle"] and not det.get("achou")
        finais[c] = {**{k: e[k] for k in ("tamanho", "mtime", "fps", "miniatura", "largura", "altura")}, "assinatura": assinatura,
                     "manual": False, "controle": it["controle"], "incluir": it["incluir"], "status_auditoria": it["status_auditoria"],
                     "chave": it["chave"], "origem": origem, "roi": roi, "det": det,
                     "motivos": [] if controle_no_escuro else motivos,
                     "suspeito": bool(motivos) and not controle_no_escuro}
        if controle_no_escuro:
            finais[c]["nota"] = "controle sem brilho: ROI herdada (esperado)"

    dados = {"versao": VERSAO_ROI, "gerado_em": time.strftime("%Y-%m-%d %H:%M:%S"), "parametros": {k: p[k] for k in p if k != "workers"},
             "mediana_do_lote": mediana, "videos": finais}
    _gravar_json(json_path, dados)
    _gravar_csv(pasta_roi / "roi.csv", finais)
    entradas = [{"caminho": c, **e} for c, e in finais.items()]
    folhas = gerar_mosaicos(entradas, pasta_roi, p, so_suspeitos)

    # ---- 3) resumo
    suspeitos = [e for e in entradas if e["suspeito"]]
    amostras = [e for e in entradas if not e["controle"]]
    detectadas = [e["det"] for e in amostras if e["det"].get("achou")]
    print(f"\nROI  |  {len(itens)} vídeo(s) ({len(amostras)} amostras, {len(itens) - len(amostras)} controle)  |  do cache: {do_cache}  |  "
          f"lidos agora: {len(pendentes)}  |  pulados (incluir = nao): {pulados}  |  tempo: {time.perf_counter() - inicio:.0f}s")
    for a in avisos:
        print(f"  aviso: {a}")
    for e in erros:
        print(f"  ERRO: {e}")
    if detectadas:
        print(f"Amostra achada em {len(detectadas)} de {len(amostras)} vídeos. Raio {min(d['r_detectado'] for d in detectadas):.0f}-"
              f"{max(d['r_detectado'] for d in detectadas):.0f} px | centro x {min(d['cx'] for d in detectadas):.0f}-{max(d['cx'] for d in detectadas):.0f}, "
              f"y {min(d['cy'] for d in detectadas):.0f}-{max(d['cy'] for d in detectadas):.0f} (o centro varia: a ROI é por vídeo)")
    if suspeitos:
        print(f"\nSUSPEITOS ({len(suspeitos)}): veja as primeiras folhas do mosaico")
        for e in suspeitos:
            print(f"   - {e['caminho']}: {'; '.join(e['motivos'])}")
    else:
        print("Nenhum vídeo suspeito.")
    print(f"\nArquivos em {pasta_roi}: roi.json, roi.csv, {len(folhas)} folha(s) de mosaico, miniaturas/")
    return 0