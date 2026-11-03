"""Programa 3 - Extração dos dados.

Entrada: os vídeos, o roi.json (programa 2) e o curadoria.csv. Saída (output/extracao/):
  <nome do vídeo>.csv     uma linha por quadro NOVO (quadro idêntico ao anterior é descartado)
  extracao_resumo.csv     uma linha por vídeo: ROI usada, quadros, saturação, níveis inicial/final/piso por canal
  cache_extracao.json     o que já foi extraído (reexecutar só refaz o que mudou)

Colunas do CSV de cada vídeo
  frame, tempo_s    índice do quadro na PRIMEIRA ocorrência e índice / fps do contêiner (timestamps fixos, jitter 0)
  fase              excitacao | decaimento, pelo MESMO corte para todos os vídeos (protocolo.instante_queda_s)
  media_R/G/B       média dos pixels da ROI;  mediana_R/G/B: mediana dos pixels da ROI
  desvio_cinza      desvio padrão do cinza dentro da ROI (irregularidade espacial)
  saturados_pct     % dos pixels da ROI com algum canal >= saturacao_limiar
  fundo_R/G/B       média de um anel logo fora da ROI (pode conter halo da própria amostra)
  canto_R/G/B       média de 4 cantos do quadro, longe da amostra (nível de preto da câmera)
  repeticoes        quantas vezes este quadro se repetiu logo depois

Nada aqui subtrai fundo, piso ou controle: o CSV guarda só o que depende dos pixels.

Piso de ruído e aviso "perto do ruído" (extracao_resumo.csv)
  O piso de cada vídeo é o nível de preto do vídeo de CONTROLE (Amostra_referencia, sem amostra luminescente) medido
  DENTRO da ROI daquele vídeo: o preto da câmera não é uniforme no quadro (as bordas são mais claras que o centro) e o
  recipiente muda de posição, então o piso tem que ser lido onde a amostra está. Ele é a média pixel a pixel dos
  quadros novos do controle depois do corte (o mesmo corte das amostras), tirada dentro do círculo da ROI.
  piso_ref_R/G/B           esse piso, por canal
  perto_do_ruido_R/G/B     (nível inicial - piso) < fator_ruido * piso
  piso_canto_R/G/B         piso pelos 4 cantos do PRÓPRIO vídeo (critério antigo)
  perto_do_ruido_canto_*   o aviso pelo critério antigo, só para comparar
  deriva_preto_R/G/B       preto dos cantos do vídeo menos o dos cantos do controle: se o preto mudou entre as gravações
                           (a referência vale para um só dia); preto_difere_da_ref marca valores acima de deriva_preto_max
  piso_fonte               referencia | cantos (sem controle utilizável, ou extracao.piso_fonte: cantos)
  Isto é refeito a cada execução a partir do resumo (não relê os vídeos de amostra), então não invalida o cache.

Entram os vídeos com incluir = sim ou pendente (nao é pulado). O resumo traz status_auditoria e incluir de cada
vídeo, para filtrar depois (por exemplo, vídeos com gravação travada) sem apagar nada.
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .roi import selecionar_videos

VERSAO_EXTRACAO = "1.0"

PADRAO = {
    "saturacao_limiar": 250,     # pixel com algum canal >= isto conta como saturado
    "cantos_px": 80,             # lado dos 4 quadrados de canto usados para o nível de preto
    "piso_minimo": 1.0,          # piso de ruído mínimo
    "fator_ruido": 5.0,          # canal cuja (amplitude inicial - piso) < fator * piso é marcado como perto do ruído
    "janela_fim_s": 10.0,        # o "nível final" é a mediana dos últimos tantos segundos do vídeo
    "piso_fonte": "referencia",  # referencia = preto do vídeo de controle dentro de cada ROI | cantos = cantos de cada vídeo
    "deriva_preto_max": 1.0,     # níveis de cinza: preto dos cantos do vídeo menos o do controle acima disso = aviso (provisório)
    "workers": 6,
}
CANAIS = ("R", "G", "B")


def parametros(cfg: dict) -> dict:
    return {**PADRAO, **(cfg.get("extracao") or {})}


def _mediana_u8(px: np.ndarray) -> np.ndarray:
    """Mediana por canal de um array (n, 3) uint8, por histograma (muito mais rápida que np.median)."""
    n = px.shape[0]
    lo, hi = (n - 1) // 2, n // 2
    saida = np.empty(3)
    for c in range(3):
        acum = np.cumsum(np.bincount(px[:, c], minlength=256))
        saida[c] = (np.searchsorted(acum, lo + 1) + np.searchsorted(acum, hi + 1)) / 2
    return saida


def extrair_video(caminho: Path, roi: dict, fps_cfg: float, instante_queda_s: float, p: dict) -> tuple[pd.DataFrame, float]:
    """Uma linha por quadro NOVO. O tempo é o da primeira ocorrência (índice / fps do contêiner)."""
    cap = cv2.VideoCapture(str(caminho))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or fps_cfg
    ok, q = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("não foi possível ler o vídeo")
    h, w = q.shape[:2]
    cx, cy, r = roi["cx"], roi["cy"], roi["r"]
    ext = 1.5 * r + 2
    x0, x1 = max(0, int(cx - ext)), min(w, int(cx + ext) + 1)
    y0, y1 = max(0, int(cy - ext)), min(h, int(cy + ext) + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    m_roi = d2 <= r ** 2
    m_fundo = (d2 >= (1.25 * r) ** 2) & (d2 <= (1.5 * r) ** 2)
    mask_roi, mask_fundo = m_roi.astype(np.uint8), m_fundo.astype(np.uint8)       # para cv2.mean
    k = int(p["cantos_px"])
    cantos = [(slice(0, k), slice(0, k)), (slice(0, k), slice(w - k, w)), (slice(h - k, h), slice(0, k)), (slice(h - k, h), slice(w - k, w))]
    limiar = int(p["saturacao_limiar"])

    linhas, anterior, idx = [], None, 0
    while ok:
        if anterior is not None and np.array_equal(q, anterior):
            linhas[-1]["repeticoes"] += 1
        else:
            corte = q[y0:y1, x0:x1]
            px = corte[m_roi]                                            # (n, 3) uint8, BGR
            media = np.array(cv2.mean(corte, mask=mask_roi)[:3])
            mediana = _mediana_u8(px)
            fundo = np.array(cv2.mean(corte, mask=mask_fundo)[:3]) if m_fundo.any() else np.full(3, np.nan)
            canto = np.mean([cv2.mean(np.ascontiguousarray(q[s]))[:3] for s in cantos], axis=0)
            t = idx / fps
            linhas.append({
                "frame": idx, "tempo_s": round(t, 4),
                "fase": "excitacao" if t < instante_queda_s else "decaimento",
                "media_R": media[2], "media_G": media[1], "media_B": media[0],
                "mediana_R": mediana[2], "mediana_G": mediana[1], "mediana_B": mediana[0],
                "desvio_cinza": float(px.mean(axis=1).std()),
                "saturados_pct": float((px.max(axis=1) >= limiar).mean() * 100),
                "fundo_R": fundo[2], "fundo_G": fundo[1], "fundo_B": fundo[0],
                "canto_R": canto[2], "canto_G": canto[1], "canto_B": canto[0],
                "repeticoes": 0,
            })
        anterior = q
        idx += 1
        ok, q = cap.read()
    cap.release()
    return pd.DataFrame(linhas), fps


def resumir(df: pd.DataFrame, fps: float, instante_queda_s: float, p: dict) -> dict:
    """Números por vídeo: quadros, saturação e, por canal, nível inicial, final, piso de ruído e aviso de ruído."""
    dec = df[df["tempo_s"] >= instante_queda_s]
    repetidos = int(df["repeticoes"].sum())
    total = len(df) + repetidos
    res = {
        "fps": round(fps, 3), "n_quadros_novos": len(df), "n_quadros_total": total,
        "repetidos_pct": round(100 * repetidos / max(total, 1), 1),
        "duracao_s": round(float(df["tempo_s"].iloc[-1]) + 1 / fps, 3) if len(df) else 0.0,
        "primeiro_decaimento_s": float(dec["tempo_s"].iloc[0]) if len(dec) else float("nan"),
        "quadros_decaimento": len(dec),
    }
    sat = dec[dec["saturados_pct"] > 0]
    res["saturacao_ate_s"] = float(sat["tempo_s"].max()) if len(sat) else 0.0
    res["saturados_pct_inicio"] = round(float(dec["saturados_pct"].iloc[0]), 2) if len(dec) else float("nan")
    fim = df[df["tempo_s"] >= (float(df["tempo_s"].iloc[-1]) - p["janela_fim_s"])] if len(df) else df
    for c in CANAIS:
        ini = float(dec[f"media_{c}"].iloc[0]) if len(dec) else float("nan")
        piso = max(float(dec[f"canto_{c}"].median()), p["piso_minimo"]) if len(dec) else p["piso_minimo"]
        res[f"inicio_{c}"] = round(ini, 2)
        res[f"final_{c}"] = round(float(fim[f"media_{c}"].median()), 2) if len(fim) else float("nan")
        res[f"piso_canto_{c}"] = round(piso, 2)
        res[f"fundo_anel_{c}"] = round(float(dec[f"fundo_{c}"].median()), 2) if len(dec) else float("nan")
        res[f"perto_do_ruido_{c}"] = bool((ini - piso) < p["fator_ruido"] * piso)
    return res


def mapa_de_preto(caminho: Path, instante_queda_s: float, fps_cfg: float, p: dict) -> dict:
    """Nível de preto do vídeo de controle, pixel a pixel.

    Média dos quadros NOVOS depois do corte (a luz de excitação já apagou e o controle não brilha). Retorna o mapa
    (h, w, 3, BGR, float32), quantos quadros entraram e a média dos 4 cantos (R, G, B)."""
    cap = cv2.VideoCapture(str(caminho))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or fps_cfg
    soma, n, anterior, idx = None, 0, None, 0
    ok, q = cap.read()
    while ok:
        if (anterior is None or not np.array_equal(q, anterior)) and round(idx / fps, 4) >= instante_queda_s:
            soma = q.astype(np.float32) if soma is None else soma + q
            n += 1
        anterior = q
        idx += 1
        ok, q = cap.read()
    cap.release()
    if soma is None:
        raise RuntimeError("o vídeo de controle não tem quadros depois do corte")
    mapa = soma / n
    h, w = mapa.shape[:2]
    k = int(p["cantos_px"])
    cantos = [mapa[:k, :k], mapa[:k, w - k:], mapa[h - k:, :k], mapa[h - k:, w - k:]]
    canto_bgr = np.mean([c.reshape(-1, 3).mean(axis=0) for c in cantos], axis=0)
    return {"mapa": mapa, "n": n, "canto_RGB": canto_bgr[::-1]}


def piso_na_roi(mapa: np.ndarray, cx: float, cy: float, r: float) -> np.ndarray:
    """Média do mapa de preto dentro do círculo da ROI, em R, G, B. NaN se a ROI cai fora do quadro."""
    h, w = mapa.shape[:2]
    x0, x1, y0, y1 = max(0, int(cx - r)), min(w, int(cx + r) + 1), max(0, int(cy - r)), min(h, int(cy + r) + 1)
    if x1 <= x0 or y1 <= y0:
        return np.full(3, np.nan)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    if not m.any():
        return np.full(3, np.nan)
    return mapa[y0:y1, x0:x1][m].mean(axis=0)[::-1]                  # BGR -> RGB


def aplicar_piso(linhas: list[dict], ref: dict | None, p: dict) -> None:
    """Define o piso de ruído e o aviso perto_do_ruido de cada vídeo (altera as linhas do resumo).

    Com controle: piso = preto do controle dentro da ROI do vídeo. Sem controle (ou ROI fora do quadro): cantos do vídeo.
    Só usa números que já estão no resumo (inicio_*, piso_canto_*, roi_*), por isso não precisa reler os vídeos."""
    fator, minimo = p["fator_ruido"], p["piso_minimo"]
    for l in linhas:
        piso_roi = piso_na_roi(ref["mapa"], l["roi_cx"], l["roi_cy"], l["roi_r"]) if ref is not None else np.full(3, np.nan)
        usa_ref = bool(np.isfinite(piso_roi).all())
        l["piso_fonte"] = "referencia" if usa_ref else "cantos"
        desvios = []
        for i, c in enumerate(CANAIS):
            ini, canto = l[f"inicio_{c}"], l[f"piso_canto_{c}"]
            piso = max(float(piso_roi[i]), minimo) if usa_ref else canto
            l[f"piso_ref_{c}"] = round(float(piso_roi[i]), 2) if usa_ref else float("nan")
            l[f"perto_do_ruido_{c}"] = bool((ini - piso) < fator * piso)
            l[f"perto_do_ruido_canto_{c}"] = bool((ini - canto) < fator * canto)
            if ref is not None:
                l[f"deriva_preto_{c}"] = round(canto - max(float(ref["canto_RGB"][i]), minimo), 2)
                desvios.append(abs(l[f"deriva_preto_{c}"]))
        l["preto_difere_da_ref"] = bool(desvios) and max(desvios) > p["deriva_preto_max"]


def _assinatura(roi: dict, base: float, p: dict, st) -> str:
    dados = {"roi": [round(roi["cx"], 2), round(roi["cy"], 2), round(roi["r"], 2)], "base": base, "versao": VERSAO_EXTRACAO,
             "p": {k: p[k] for k in ("saturacao_limiar", "cantos_px", "piso_minimo", "fator_ruido", "janela_fim_s")},
             "tamanho": st.st_size, "mtime": int(st.st_mtime)}
    return hashlib.sha1(json.dumps(dados, sort_keys=True).encode()).hexdigest()[:16]


def _nomes_csv(itens: list[dict]) -> dict[str, str]:
    """Nome do CSV de cada vídeo: o nome do arquivo; se dois vídeos têm o mesmo nome, o caminho inteiro."""
    contagem: dict[str, int] = {}
    for it in itens:
        contagem[Path(it["caminho"]).stem] = contagem.get(Path(it["caminho"]).stem, 0) + 1
    saida = {}
    for it in itens:
        stem = Path(it["caminho"]).stem
        saida[it["caminho"]] = (stem if contagem[stem] == 1 else Path(it["caminho"]).with_suffix("").as_posix().replace("/", "__")) + ".csv"
    return saida


def _meta(item: dict, entrada: dict, arquivo_csv: str) -> dict:
    roi = entrada["roi"]
    return {"caminho": item["caminho"], "arquivo_csv": arquivo_csv, "chave": item["chave"], "controle": item["controle"],
            "incluir": item["incluir"], "status_auditoria": item["status_auditoria"],
            "roi_cx": round(roi["cx"], 2), "roi_cy": round(roi["cy"], 2), "roi_r": round(roi["r"], 2),
            "roi_origem": entrada.get("origem", ""), "roi_suspeito": bool(entrada.get("suspeito")),
            "roi_motivos": " | ".join(entrada.get("motivos", []))}


def _processar(item: dict, entrada: dict, base: float, fps_cfg: float, p: dict, pasta_csv: Path, nome_csv: str) -> dict:
    t0 = time.perf_counter()
    df, fps = extrair_video(item["arquivo"], entrada["roi"], fps_cfg, base, p)
    tmp = pasta_csv / (nome_csv + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(pasta_csv / nome_csv)
    res = {**_meta(item, entrada, nome_csv), **resumir(df, fps, base, p)}
    res.update(versao_extracao=VERSAO_EXTRACAO, extraido_em=time.strftime("%Y-%m-%d %H:%M:%S"), tempo_s_processamento=round(time.perf_counter() - t0, 1))
    return res


def rodar(cfg: dict, pasta=None, saida=None, roi_json=None, refazer: bool = False, workers: int | None = None, filtro: list[str] | None = None) -> int:
    inicio = time.perf_counter()
    p = parametros(cfg)
    prot = cfg.get("protocolo") or {}
    base = float(prot.get("instante_queda_s", 0.5))
    fps_cfg = float(prot.get("fps_esperado", 15))
    pasta = Path(pasta or (cfg.get("pastas") or {}).get("videos", "data"))
    if not pasta.is_dir():
        raise NotADirectoryError(f"Pasta de vídeos não encontrada: {pasta.resolve()}")
    raiz_saida = Path((cfg.get("pastas") or {}).get("saida", "output"))
    roi_path = Path(roi_json or raiz_saida / "roi" / "roi.json")
    if not roi_path.exists():
        raise FileNotFoundError(f"{roi_path} não existe: rode 'luminescencia roi' antes da extração")
    rois = json.loads(roi_path.read_text(encoding="utf-8")).get("videos", {})
    pasta_csv = Path(saida or raiz_saida / "extracao")
    pasta_csv.mkdir(parents=True, exist_ok=True)
    cache_path = pasta_csv / "cache_extracao.json"
    try:
        cache = {} if refazer else json.loads(cache_path.read_text(encoding="utf-8")).get("videos", {})
    except (OSError, ValueError):
        cache = {}

    itens, avisos, pulados = selecionar_videos(cfg, pasta)
    todos = list(itens)                                  # o controle é procurado aqui, mesmo que --videos o filtre
    if filtro:
        itens = [i for i in itens if any(f.lower() in i["caminho"].lower() for f in filtro)]
    sem_roi = [i["caminho"] for i in itens if not (rois.get(i["caminho"]) or {}).get("roi")]
    for c in sem_roi:
        avisos.append(f"sem ROI em roi.json (rode 'luminescencia roi'): {c}")
    itens = [i for i in itens if i["caminho"] not in sem_roi]
    if not itens:
        print("Nenhum vídeo para extrair.")
        for a in avisos:
            print(f"  aviso: {a}")
        return 1
    nomes_csv = _nomes_csv(itens)
    workers = max(1, int(workers or p["workers"]))

    resumos, pendentes, do_cache = {}, [], 0
    for it in itens:
        c, ent = it["caminho"], rois[it["caminho"]]
        assin = _assinatura(ent["roi"], base, p, it["arquivo"].stat())
        antigo = cache.get(c)
        if antigo and antigo.get("assinatura") == assin and (pasta_csv / nomes_csv[c]).exists():
            resumos[c] = {**antigo["resumo"], **_meta(it, ent, nomes_csv[c])}     # incluir/status/ROI podem ter mudado sem reler
            do_cache += 1
        else:
            pendentes.append((it, assin))

    erros = []
    novo_cache = {c: {"assinatura": cache[c]["assinatura"], "resumo": r} for c, r in resumos.items() if c in cache}
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futuros = {pool.submit(_processar, it, rois[it["caminho"]], base, fps_cfg, p, pasta_csv, nomes_csv[it["caminho"]]): (it, assin)
                   for it, assin in pendentes}
        for f in tqdm(as_completed(futuros), total=len(futuros), desc="Extraindo", unit="vídeo", disable=not pendentes):
            it, assin = futuros[f]
            try:
                resumos[it["caminho"]] = f.result()
                novo_cache[it["caminho"]] = {"assinatura": assin, "resumo": resumos[it["caminho"]]}
            except Exception as erro:                                  # um vídeo com problema não derruba o lote
                erros.append(f"{it['caminho']}: {type(erro).__name__}: {erro}")
        pool.shutdown()
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:                                                           # guarda o que já foi feito, mesmo com Ctrl+C
        try:
            cache_path.write_text(json.dumps({"versao": VERSAO_EXTRACAO, "videos": novo_cache}, default=float), encoding="utf-8")
        except OSError:
            pass

    linhas = [resumos[c] for c in sorted(resumos)]

    # ---- piso de ruído: preto do vídeo de controle dentro de cada ROI (ou cantos, se não houver controle)
    ref, ref_caminho = None, ""
    if p["piso_fonte"] == "referencia":
        controles = [i for i in todos if i["controle"]]
        if not controles:
            avisos.append("sem vídeo de controle (ou ele está com incluir = nao): piso de ruído pelos cantos de cada vídeo")
        else:
            if len(controles) > 1:
                avisos.append(f"{len(controles)} vídeos de controle; usando {controles[0]['caminho']}")
            try:
                print(f"Medindo o preto do vídeo de controle ({controles[0]['caminho']})...")
                ref, ref_caminho = mapa_de_preto(controles[0]["arquivo"], base, fps_cfg, p), controles[0]["caminho"]
            except Exception as erro:
                avisos.append(f"não foi possível medir o preto do controle ({type(erro).__name__}: {erro}): piso pelos cantos")
    aplicar_piso(linhas, ref, p)
    campos = list(dict.fromkeys(k for l in linhas for k in l))
    resumo_csv = pasta_csv / "extracao_resumo.csv"
    with resumo_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)

    # ---- resumo no terminal
    amostras = [l for l in linhas if not l["controle"]]
    print(f"\nEXTRAÇÃO  |  {len(linhas)} vídeo(s)  |  do cache: {do_cache}  |  extraídos agora: {len(pendentes) - len(erros)}  |  "
          f"pulados (incluir = nao): {pulados}  |  tempo: {time.perf_counter() - inicio:.0f}s")
    for a in avisos:
        print(f"  aviso: {a}")
    for e in erros:
        print(f"  ERRO: {e}")
    if amostras:
        med_rep = float(np.median([l["repetidos_pct"] for l in amostras]))
        fora = [l for l in amostras if abs(l["repetidos_pct"] - med_rep) > 3]
        print(f"Quadros repetidos: mediana {med_rep:.0f}% | quadros novos por vídeo: {min(l['n_quadros_novos'] for l in amostras)}-{max(l['n_quadros_novos'] for l in amostras)}")
        for l in fora:
            print(f"  atenção: {l['caminho']} tem {l['repetidos_pct']:.0f}% de quadros repetidos (status_auditoria: {l['status_auditoria']})")
        if ref is not None:
            med = {c: float(np.nanmedian([l[f"piso_ref_{c}"] for l in amostras])) for c in CANAIS}
            cref = dict(zip(CANAIS, ref["canto_RGB"]))
            print(f"  piso de ruído: controle {ref_caminho} ({ref['n']} quadros novos após o corte)")
            print("    preto dentro das ROIs (mediana dos vídeos): " + " | ".join(f"{c} {med[c]:.2f}" for c in CANAIS)
                  + "   cantos do controle: " + " | ".join(f"{c} {cref[c]:.2f}" for c in CANAIS))
            difere = [l for l in amostras if l["preto_difere_da_ref"]]
            dmax = max(abs(l[f"deriva_preto_{c}"]) for l in amostras for c in CANAIS)
            print(f"    preto dos cantos de cada vídeo contra o do controle: maior diferença {dmax:.2f} nível(is); "
                  f"{len(difere)} vídeo(s) acima de {p['deriva_preto_max']:g}")
            for l in difere[:3]:
                print(f"      atenção: {l['caminho']} (preto diferente do controle: confira se o dia de gravação é o mesmo)")
        else:
            motivo = "extracao.piso_fonte: cantos" if p["piso_fonte"] == "cantos" else "sem vídeo de controle utilizável"
            print(f"  piso de ruído: pelos cantos de cada vídeo ({motivo})")
        for c in CANAIS:
            n = sum(l[f"perto_do_ruido_{c}"] for l in amostras)
            extra = f" (pelos cantos seriam {sum(l[f'perto_do_ruido_canto_{c}'] for l in amostras)})" if ref is not None else ""
            print(f"  canal {c}: perto do ruído em {n} de {len(amostras)} vídeos{extra}")
        sat = [l for l in amostras if l["saturacao_ate_s"] > 0]
        if sat:
            print(f"  saturação depois do corte: {len(sat)} vídeo(s), até {max(l['saturacao_ate_s'] for l in sat):.1f} s (veja saturados_pct nos CSVs)")
        susp = [l for l in linhas if l["roi_suspeito"]]
        if susp:
            print(f"  {len(susp)} vídeo(s) com ROI suspeita (confira o mosaico): {', '.join(l['caminho'] for l in susp[:5])}{' ...' if len(susp) > 5 else ''}")
    print(f"\nArquivos em {pasta_csv}: um CSV por vídeo, extracao_resumo.csv")
    return 1 if erros else 0