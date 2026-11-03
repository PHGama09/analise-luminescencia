"""Protótipo da extração: ROI automática + quadros repetidos descartados + ajuste do decaimento.

Serve para responder, com poucos vídeos, antes de construir os programas 2 e 3:
  1. as curvas ajustam com ~8 quadros novos por segundo e com canais perto do piso de ruído?
  2. quanto o erro de ±0,125 s no corte do decaimento muda a amplitude e o tau?
  3. a ROI automática funciona (e o círculo muda de vídeo para vídeo)?

Nada aqui é manual e nada altera vídeos, curadoria.csv ou config.yaml. Resultados em output/prototipo/.

Uso (na pasta do projeto, com o venv ativo):
    python -m luminescencia.prototipo
    python -m luminescencia.prototipo --roi-fixa          # NÃO recomendado: o recipiente muda de posição entre vídeos
    python -m luminescencia.prototipo --videos "Amostras A/1/B_15_v1a.avi" Amostra_referencia.avi

Como a ROI é achada: depois que a luz de excitação apaga, o único brilho do quadro é o da própria amostra.
A média dos quadros novos entre 0,6 e 1,5 s é binarizada (Otsu) e o maior objeto vira o círculo da amostra.
O vídeo de controle não brilha: ele herda a ROI mediana dos outros vídeos.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

if __package__ in (None, ""):                       # permite rodar como arquivo: python src/luminescencia/prototipo.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from luminescencia.config import carregar_config
from luminescencia.modelos import MODELOS, ajustar_canais

VIDEOS_PADRAO = [
    "Amostra_referencia.avi",
    "Amostras A/1/B_15_v1a.avi",
    "Amostras A/1/R_15_v1a.avi",
    "Amostras A/1/G_15_v1a.avi",
    "Amostras A/1/RGB_05_05_05_v1a.avi",
    "Amostras A/1/RB_075_075_v1a.avi",
    "Amostras C/1/RB_075_075_v1c.avi",
]

PADRAO = {
    "videos": VIDEOS_PADRAO,
    "janela_roi_s": [0.6, 1.5],      # quadros usados para achar a ROI (a luz já apagou: a queda máxima medida é 0,40 s)
    "margem_interna": 0.12,          # a ROI usa 88% do raio detectado, para fugir da borda borrada
    "circularidade_minima": 0.85,
    "contraste_minimo": 3.0,         # brilho dentro / fora do círculo
    "desvio_raio_lote": 0.25,        # raio diferente da mediana do lote por mais que isso = suspeito
    "piso_minimo": 1.0,              # piso de ruído usado se não houver controle
    "fator_ruido": 5.0,              # canal cuja amplitude inicial < fator * piso é marcado como perto do ruído
    "deslocamentos_corte_s": [-0.125, 0.0, 0.125],   # sensibilidade do corte (0,125 s = passo real da câmera)
}


# ------------------------------------------------------------------ ROI
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
    """Acha o círculo da amostra numa imagem média. Retorna dict com cx, cy, r (já com margem) e diagnóstico."""
    h, w = media.shape[:2]
    cinza = cv2.GaussianBlur(media.mean(axis=2), (0, 0), 5)
    pico, fundo = float(np.percentile(cinza, 99.5)), float(np.median(cinza))
    res = {"achou": False, "motivos": [], "pico": round(pico, 1), "fundo": round(fundo, 2)}
    if pico < 4:
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
    res.update(achou=True, cx=cx, cy=cy, r_detectado=r_eq, r=r_eq * (1 - p["margem_interna"]),
               circularidade=round(float(circ), 3), contraste=round(contraste, 1),
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


def comparar_com_lote(rois: dict, p: dict):
    """Marca raios que fogem da mediana do lote e fornece a ROI mediana (para controle e falhas).

    O centro NÃO é critério de suspeita: nos vídeos reais o recipiente muda de posição entre vídeos (até ~0,3 raio)
    com o raio quase constante, então só o desvio do raio indica detecção errada (o centro fica no relatório).
    """
    achadas = {k: v for k, v in rois.items() if v["achou"]}
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


# ------------------------------------------------------------------ extração
def extrair_video(caminho: Path, roi: dict, fps: float, instante_queda_s: float) -> pd.DataFrame:
    """Uma linha por quadro NOVO (o repetido é descartado; o tempo é o da primeira ocorrência)."""
    cx, cy, r = roi["cx"], roi["cy"], roi["r"]
    cap = cv2.VideoCapture(str(caminho))
    ok, q = cap.read()
    if not ok:
        raise RuntimeError("não foi possível ler o vídeo")
    h, w = q.shape[:2]
    ext = 1.5 * r + 2
    x0, x1 = max(0, int(cx - ext)), min(w, int(cx + ext) + 1)
    y0, y1 = max(0, int(cy - ext)), min(h, int(cy + ext) + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    m_roi = d2 <= r ** 2
    m_fundo = (d2 >= (1.25 * r) ** 2) & (d2 <= (1.5 * r) ** 2)

    linhas, anterior, idx = [], None, 0
    while ok:
        if anterior is not None and np.array_equal(q, anterior):
            linhas[-1]["repeticoes"] += 1
        else:
            corte = q[y0:y1, x0:x1]
            px = corte[m_roi].astype(np.float32)                 # BGR
            fd = corte[m_fundo].astype(np.float32).mean(axis=0) if m_fundo.any() else np.full(3, np.nan)
            m = px.mean(axis=0)
            t = idx / fps
            linhas.append({
                "frame": idx, "tempo_s": round(t, 4),
                "fase": "excitacao" if t < instante_queda_s else "decaimento",
                "media_R": m[2], "media_G": m[1], "media_B": m[0],
                "desvio_cinza": float(px.mean(axis=1).std()),
                "saturados_pct": float((px.max(axis=1) >= 250).mean() * 100),
                "fundo_R": fd[2], "fundo_G": fd[1], "fundo_B": fd[0],
                "repeticoes": 0,
            })
        anterior = q
        idx += 1
        ok, q = cap.read()
    cap.release()
    return pd.DataFrame(linhas)


# ------------------------------------------------------------------ ajuste
def _valor_no_tempo(linha: dict, t_rel: float) -> float:
    mod = MODELOS.get(linha.get("modelo", ""))
    if mod is None or any(pd.isna(linha.get(k)) for k in mod.params):
        return float("nan")
    return float(mod.func(t_rel, *[linha[k] for k in mod.params]))


def ajustar_com_cortes(df: pd.DataFrame, nome: str, cortes, base: float, canais: dict, aj: dict, piso: dict, p: dict) -> list[dict]:
    saida = []
    sem_fase = df.drop(columns=["fase"])                    # força o corte por tempo
    for corte in cortes:
        tab = ajustar_canais(
            sem_fase, corte, canais, modelos=aj.get("modelos", ["exp_simples", "exp_dupla"]),
            r2_minimo=aj.get("r2_minimo"), delta_bic=aj.get("delta_bic"),
            fracao_minima_componente=aj.get("fracao_minima_componente"),
            separacao_minima_tau=aj.get("separacao_minima_tau"), minimo_pontos=aj.get("minimo_pontos"),
        )
        decai = sem_fase[sem_fase["tempo_s"] >= corte]
        t0 = float(decai["tempo_s"].iloc[0]) if len(decai) else float("nan")
        for linha in tab.to_dict("records"):
            col = canais[linha["canal"]]
            inicial = float(decai[col].iloc[0]) if len(decai) else float("nan")
            limite = max(p["fator_ruido"] * piso.get(linha["canal"], p["piso_minimo"]), 1.0)
            linha.update(
                video=nome, corte_s=round(corte, 4), t0_s=t0, valor_inicial=inicial,
                valor_no_corte_base=_valor_no_tempo(linha, base - t0),   # modelo avaliado no MESMO instante absoluto
                perto_do_ruido=bool(inicial < limite), limite_ruido=round(limite, 2),
                quadros_novos_por_s=round(len(decai) / max(decai["tempo_s"].iloc[-1] - t0, 1e-9), 2) if len(decai) > 1 else float("nan"),
            )
            saida.append(linha)
    return saida


def resumo_sensibilidade(aj: pd.DataFrame, base: float) -> pd.DataFrame:
    linhas = []
    for (v, c), g in aj.groupby(["video", "canal"]):
        vi, tau = g["valor_no_corte_base"].to_numpy(float), g["tau_medio"].to_numpy(float)
        ref = g.iloc[int((g["corte_s"] - base).abs().to_numpy().argmin())]
        def amp(x):
            x = x[np.isfinite(x)]
            return float((x.max() - x.min()) / abs(x.mean())) if len(x) > 1 and x.mean() != 0 else float("nan")
        linhas.append({
            "video": v, "canal": c, "perto_do_ruido": bool(ref["perto_do_ruido"]),
            "modelo_no_corte_base": ref["modelo"], "r2_no_corte_base": ref.get("r2"),
            "variacao_valor_no_corte": amp(vi), "variacao_tau": amp(tau),
            "modelo_muda_com_o_corte": g["modelo"].nunique() > 1,
        })
    return pd.DataFrame(linhas)


# ------------------------------------------------------------------ verificação visual
def gravar_mosaico(imagens: list[tuple[str, np.ndarray, dict, dict | None]], caminho: Path):
    """Uma folha com a imagem média de cada vídeo e o círculo usado (verde) / detectado (amarelo)."""
    tw, th, cols = 640, 360, 3
    blocos = []
    for nome, media, roi, det in imagens:
        pico = max(float(np.percentile(media, 99.5)), 1.0)
        im = np.clip(media / pico * 255, 0, 255).astype(np.uint8)
        if det and det.get("achou"):
            cv2.circle(im, (int(det["cx"]), int(det["cy"])), int(det["r_detectado"]), (0, 255, 255), 2)
        cv2.circle(im, (int(roi["cx"]), int(roi["cy"])), int(roi["r"]), (0, 255, 0), 3)
        im = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.putText(im, nome[:48], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        origem = (det or {}).get("origem", "")
        if origem:
            cv2.putText(im, origem, (8, th - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2, cv2.LINE_AA)
        blocos.append(im)
    while len(blocos) % cols:
        blocos.append(np.zeros((th, tw, 3), np.uint8))
    linhas = [np.hstack(blocos[i:i + cols]) for i in range(0, len(blocos), cols)]
    cv2.imwrite(str(caminho), np.vstack(linhas))


# ------------------------------------------------------------------ principal
def rodar(cfg: dict, videos: list[str], saida: Path, roi_fixa: bool = False) -> int:
    p = {**PADRAO, **cfg.get("prototipo", {})}
    prot = cfg.get("protocolo", {})
    base = float(prot.get("instante_queda_s", 0.5))
    fps_cfg = float(prot.get("fps_esperado", 15))
    pasta = Path(cfg.get("pastas", {}).get("videos", "data"))
    palavra_controle = str(cfg.get("nomes", {}).get("controle", "referencia")).lower()
    aj = cfg.get("ajuste", {})
    canais = aj.get("canais", {"R": "media_R", "G": "media_G", "B": "media_B"})
    cortes = [base + d for d in p["deslocamentos_corte_s"]]

    caminhos = {}
    for v in videos:
        c = Path(v) if Path(v).is_absolute() else pasta / v
        if not c.exists():
            print(f"  ! não encontrado: {c}")
            continue
        caminhos[v] = c
    if not caminhos:
        print("Nenhum vídeo encontrado. Confira pastas.videos no config.yaml.")
        return 1
    (saida / "extracao").mkdir(parents=True, exist_ok=True)

    # ---- 1) ROI de cada vídeo
    print("1/3  Detectando a amostra (ROI)...")
    medias, rois, fps_de = {}, {}, {}
    for v, c in caminhos.items():
        cap = cv2.VideoCapture(str(c))
        fps_de[v] = cap.get(cv2.CAP_PROP_FPS) or fps_cfg
        cap.release()
        medias[v], n = media_da_janela(c, fps_de[v], p["janela_roi_s"])
        if medias[v] is None:
            rois[v] = {"achou": False, "motivos": ["não foi possível ler quadros na janela da ROI"]}
        else:
            rois[v] = detectar_roi(medias[v], p)
            rois[v]["quadros_novos_na_janela"] = n
    mediana = comparar_com_lote(rois, p)
    if mediana is None:
        print("Nenhuma ROI foi detectada em nenhum vídeo. Veja o mosaico e os motivos abaixo.")
    finais = {}
    for v, d in rois.items():
        controle = palavra_controle in Path(v).stem.lower()
        if roi_fixa and mediana:
            finais[v] = dict(mediana)
            d["origem"] = "ROI FIXA (mediana do lote)"
        elif d["achou"]:
            finais[v] = {k: d[k] for k in ("cx", "cy", "r")}
            d["origem"] = ""
        elif mediana:
            finais[v] = dict(mediana)
            d["origem"] = "ROI HERDADA (mediana do lote)"
            if not controle:
                d["motivos"].append("ROI herdada da mediana porque a detecção falhou")
        d["suspeito"] = bool(d["motivos"]) and not (controle and not d["achou"])
        d["controle"] = controle
        print(f"   {'SUSPEITO' if d['suspeito'] else 'ok      '}  {v}"
              + (f"  [{'; '.join(d['motivos'])}]" if d["motivos"] else "")
              + (f"  ({d['origem']})" if d.get("origem") else ""))
    if not finais:
        return 1
    mos = [(v, medias[v], finais[v], rois[v]) for v in caminhos if medias.get(v) is not None and v in finais]
    gravar_mosaico(mos, saida / "verificacao_roi.png")
    j = {v: {**{k: (round(x, 2) if isinstance(x, float) else x) for k, x in rois[v].items()}, "roi_usada": {k: round(x, 2) for k, x in finais[v].items()}} for v in finais}
    (saida / "roi_prototipo.json").write_text(json.dumps(j, ensure_ascii=False, indent=1), encoding="utf-8")
    if mediana:
        achadas = [d for d in rois.values() if d["achou"]]
        print(f"   spread da ROI no lote: raio {min(d['r_detectado'] for d in achadas):.0f}-{max(d['r_detectado'] for d in achadas):.0f} px | "
              f"centro x {min(d['cx'] for d in achadas):.0f}-{max(d['cx'] for d in achadas):.0f}, "
              f"y {min(d['cy'] for d in achadas):.0f}-{max(d['cy'] for d in achadas):.0f}")

    # ---- 2) extração
    print("2/3  Extraindo (quadros repetidos descartados)...")
    dfs = {}
    for v, c in caminhos.items():
        if v not in finais:
            continue
        try:
            df = extrair_video(c, finais[v], fps_de[v], base)
        except Exception as erro:                       # um vídeo com problema não derruba o lote
            print(f"   ! {v}: {erro}")
            continue
        df.to_csv(saida / "extracao" / (Path(v).stem + ".csv"), index=False, encoding="utf-8-sig")
        dfs[v] = df
        total = len(df) + int(df["repeticoes"].sum())
        print(f"   {v}: {len(df)} quadros novos de {total} ({df['repeticoes'].sum() / total:.0%} repetidos)")

    # ---- piso de ruído = o que o controle mede depois da queda
    piso = {}
    for v, df in dfs.items():
        if rois[v]["controle"]:
            fim = df[df["tempo_s"] >= base]
            piso = {k: max(float(fim[col].median()), p["piso_minimo"]) for k, col in canais.items()}
            print(f"   piso de ruído (controle, mediana na ROI): " + ", ".join(f"{k}={x:.2f}" for k, x in piso.items()))
    if not piso:
        print(f"   sem vídeo de controle: piso de ruído assumido = {p['piso_minimo']}")

    # ---- 3) ajuste
    print("3/3  Ajustando o decaimento (3 cortes, para medir a sensibilidade)...")
    todas = []
    for v, df in dfs.items():
        if rois[v]["controle"]:
            continue
        todas += ajustar_com_cortes(df, v, cortes, base, canais, aj, piso, p)
    if not todas:
        print("Nada para ajustar.")
        return 1
    tab = pd.DataFrame(todas)
    ordem = ["video", "canal", "corte_s", "modelo", "r2", "tau_medio", "valor_inicial", "valor_no_corte_base", "perto_do_ruido",
             "n_pontos", "quadros_novos_por_s", "rmse", "fracao_rapida", "r2_exp_simples", "r2_exp_dupla", "bic_exp_simples", "bic_exp_dupla", "observacao"]
    tab = tab[[c for c in ordem if c in tab.columns] + [c for c in tab.columns if c not in ordem]]
    tab.to_csv(saida / "ajuste_prototipo.csv", index=False, encoding="utf-8-sig")
    sens = resumo_sensibilidade(tab, base)
    sens.to_csv(saida / "sensibilidade_corte.csv", index=False, encoding="utf-8-sig")

    # ---- resumo no terminal
    print(f"\nAJUSTE NO CORTE BASE ({base:g} s)")
    visto = tab[(tab["corte_s"] - base).abs() < 1e-6]
    cols = [c for c in ["video", "canal", "modelo", "r2", "tau_medio", "valor_inicial", "perto_do_ruido", "n_pontos"] if c in visto.columns]
    print(visto[cols].round(4).to_string(index=False))
    ok = sens[~sens["perto_do_ruido"]]
    print(f"\nCanais fora do piso de ruído: {len(ok)} de {len(sens)}")
    for nome, col in (("valor do modelo em t=corte base", "variacao_valor_no_corte"), ("tau médio", "variacao_tau")):
        x = ok[col].dropna()
        if len(x):
            print(f"  variação de {nome} ao mover o corte ±{max(abs(d) for d in p['deslocamentos_corte_s']):g} s: "
                  f"mediana {x.median():.1%} | máxima {x.max():.1%}")
    print(f"  o modelo escolhido muda com o corte em {int(ok['modelo_muda_com_o_corte'].sum())} de {len(ok)} canais")
    sim = visto[~visto["perto_do_ruido"]]
    if len(sim):
        print("  modelos escolhidos (canais fora do ruído): " + ", ".join(f"{k}={n}" for k, n in sim["modelo"].replace("", "nenhum").value_counts().items()))
        print(f"  R² ≥ {aj.get('r2_minimo', 0.99)} em {int((sim['r2'] >= aj.get('r2_minimo', 0.99)).sum())} de {len(sim)}")
    print(f"\nArquivos em {saida}: verificacao_roi.png, roi_prototipo.json, ajuste_prototipo.csv, sensibilidade_corte.csv, extracao/*.csv")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="prototipo", description="Protótipo: ROI automática + extração + ajuste")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--videos", nargs="+", help="caminhos relativos a pastas.videos (padrão: 7 vídeos escolhidos)")
    ap.add_argument("--saida", default=None, help="pasta de saída (padrão: <pastas.saida>/prototipo)")
    ap.add_argument("--roi-fixa", action="store_true", help="usa a ROI mediana do lote em todos os vídeos")
    a = ap.parse_args(argv)
    cfg = carregar_config(a.config)
    saida = Path(a.saida) if a.saida else Path(cfg.get("pastas", {}).get("saida", "output")) / "prototipo"
    videos = a.videos or cfg.get("prototipo", {}).get("videos") or VIDEOS_PADRAO
    return rodar(cfg, videos, saida, a.roi_fixa)


if __name__ == "__main__":
    raise SystemExit(main())