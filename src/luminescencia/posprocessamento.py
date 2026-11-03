"""Programa 5 - Pós-processamento.

Entrada: output/consolidacao/ (series.csv e videos.csv).
Saída (output/posprocessamento/):
  series_pos.csv                  só o decaimento, a partir do começo da menor janela de sensibilidade (padrão 2,0 s)
  videos_pos.csv                  uma linha por vídeo: sinalizadores de saturação, canal ativo/ruidoso e se entra no ajuste
  relatorio_posprocessamento.txt  o que entrou, o que ficou de fora e por quê, e o equilíbrio entre amostras

O que esta etapa FAZ (decidido em 02/11/2026)
  - Janela do ajuste: todos os vídeos começam no MESMO instante (posprocessamento.inicio_ajuste_s, padrão 2,5 s), depois
    da saturação da grande maioria. Não há janela por vídeo, para não criar diferença entre grupos.
  - Vídeo que ainda satura depois do início da janela: marcado (satura_apos_*) e fora do ajuste principal
    (usar_no_ajuste_* = False). Os dados dele continuam em series_pos.csv para um ajuste à parte.
  - Canal perto do ruído: fora do ajuste. A decisão vem do SINAL MEDIDO (perto_do_ruido_*), nunca da proporção do nome do
    arquivo: usar prop_* para decidir quem é ajustado faria a IA ler o alvo pela ausência do ajuste (decidido em 02/11/2026).
    Canal sem o pó mas com sinal (vazamento de cor da câmera) é ajustado como qualquer outro. canal_ativo_* continua nas
    tabelas só para conferência e NÃO deve ser entrada da IA.
  - Sensibilidade: series_pos.csv guarda também os instantes das janelas de sensibilidade (2,0 / 2,5 / 3,0 s). O ajuste
    escolhe o recorte por tempo_s; esta etapa só diz, para cada início, quais vídeos ainda saturam.

O que NÃO faz: não ajusta modelos, não altera valores (corr_* chega igual ao da consolidação), não realinha o tempo,
não linearizar a gama e não calcula métricas de cor (ficam para depois, sob decisão sua).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CANAIS = ["R", "G", "B"]
FASE_DECAIMENTO = "decaimento"
IDENT = ["video", "amostra", "grupo", "num_video", "proporcao", "prop_R", "prop_G", "prop_B"]


def parametros(cfg: dict) -> dict:
    p = {"inicio_ajuste_s": 2.5, "janelas_sensibilidade": [2.0, 2.5, 3.0],
         "saturacao_pct_max": 0.0, "frac_nao_positivo_max": 0.10}
    p.update(cfg.get("posprocessamento", {}) or {})
    corte = float(cfg.get("protocolo", {}).get("instante_queda_s", 0.5))
    if p["inicio_ajuste_s"] < corte:
        raise ValueError(f"posprocessamento.inicio_ajuste_s ({p['inicio_ajuste_s']}) vem antes do corte ({corte} s)")
    p["janelas_sensibilidade"] = sorted({float(x) for x in p["janelas_sensibilidade"]} | {float(p["inicio_ajuste_s"])})
    return p


def _sufixo(x: float) -> str:
    return f"{x:g}".replace(".", "_") + "s"


def recortar(series: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Só decaimento, a partir do início da menor janela de sensibilidade. Valores não são alterados."""
    inicio = min(p["janelas_sensibilidade"])
    dec = series[series["fase"].astype(str).str.lower().str.startswith("decai")]
    return dec[dec["tempo_s"] >= inicio - 1e-9].reset_index(drop=True)


def resumir_videos(videos: pd.DataFrame, rec: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Uma linha por vídeo com saturação, qualidade por canal e a decisão usar_no_ajuste_*."""
    inicio = p["inicio_ajuste_s"]
    linhas = []
    for _, v in videos.iterrows():
        d_all = rec[rec["video"] == v["video"]]
        d = d_all[d_all["tempo_s"] >= inicio - 1e-9]
        sat = d_all[d_all["saturados_pct"] > p["saturacao_pct_max"]]
        sat_ate = float(sat["tempo_s"].max()) if len(sat) else 0.0
        L = {k: v[k] for k in IDENT}
        L.update(n_pontos=len(d), tempo_inicial_s=float(d["tempo_s"].min()) if len(d) else np.nan,
                 tempo_final_s=float(d["tempo_s"].max()) if len(d) else np.nan, sat_ate_s=round(sat_ate, 3))
        for x in p["janelas_sensibilidade"]:
            L[f"satura_apos_{_sufixo(x)}"] = bool(sat_ate >= x)
        satura = sat_ate >= inicio
        for c in CANAIS:
            ativo = bool(v[f"canal_ativo_{c}"])
            ruido = bool(v.get(f"perto_do_ruido_{c}", False))
            serie = d[f"corr_{c}"] if len(d) else pd.Series(dtype=float)
            frac = float((serie <= 0).mean()) if len(serie) else np.nan
            motivo = []
            if ruido:
                motivo.append("perto_do_ruido")
            if satura:
                motivo.append(f"saturacao_ate_{sat_ate:g}s")
            L[f"canal_ativo_{c}"] = ativo                       # só conferência: vem do nome do arquivo (alvo)
            L[f"sinal_presente_{c}"] = not ruido                # medido no vídeo
            L[f"frac_nao_positivo_{c}"] = round(frac, 4) if frac == frac else frac
            L[f"aviso_cauda_no_ruido_{c}"] = bool((not ruido) and frac == frac and frac > p["frac_nao_positivo_max"])
            L[f"usar_no_ajuste_{c}"] = not motivo
            L[f"motivo_fora_{c}"] = ";".join(motivo)
        for k in ("piso_video_R", "piso_video_G", "piso_video_B", "final_corr_R", "final_corr_G", "final_corr_B"):
            if k in v:
                L[k] = v[k]
        linhas.append(L)
    return pd.DataFrame(linhas)


def _passos(rec: pd.DataFrame, inicio: float) -> dict:
    d = rec[rec["tempo_s"] >= inicio - 1e-9]
    dt = d.groupby("video")["tempo_s"].diff().dropna().round(3)
    return (dt.value_counts(normalize=True).head(5) * 100).round(1).to_dict() if len(dt) else {}


def _relatorio(vp: pd.DataFrame, rec: pd.DataFrame, p: dict) -> str:
    L = ["PÓS-PROCESSAMENTO", "=" * 60,
         f"Vídeos: {len(vp)}   início do ajuste: {p['inicio_ajuste_s']:g} s (igual para todos)   "
         f"janelas de sensibilidade: {', '.join(f'{x:g}' for x in p['janelas_sensibilidade'])} s",
         f"Pontos por vídeo na janela principal: {int(vp['n_pontos'].min())}-{int(vp['n_pontos'].max())}", "",
         "Vídeos que ainda saturam depois de cada instante:"]
    for x in p["janelas_sensibilidade"]:
        L.append(f"  depois de {x:g} s: {int(vp[f'satura_apos_{_sufixo(x)}'].sum())}")
    sat = vp[vp["sat_ate_s"] >= p["inicio_ajuste_s"]].sort_values("sat_ate_s")
    if len(sat):
        L += ["", f"Saturam depois de {p['inicio_ajuste_s']:g} s (fora do ajuste principal; dados em series_pos.csv):"]
        L += [f"  {r.video} ({r.amostra}): até {r.sat_ate_s:g} s" for r in sat.itertuples()]
    L += ["", "Canais que entram no ajuste principal, por amostra (usados/total de vídeos; decidido pelo sinal medido, não pela proporção):"]
    for c in CANAIS:
        usa = vp[vp[f"usar_no_ajuste_{c}"]]
        por = {a: f"{int((usa['amostra'] == a).sum())}/{int((vp['amostra'] == a).sum())}" for a in sorted(vp["amostra"].unique())}
        com, sem = vp[vp[f"canal_ativo_{c}"]], vp[~vp[f"canal_ativo_{c}"]]
        L.append(f"  {c}: {len(usa)} de {len(vp)}   (por amostra: {por}; com o pó: {int(com[f'usar_no_ajuste_{c}'].sum())}/{len(com)}, "
                 f"sem o pó, só vazamento de cor: {int(sem[f'usar_no_ajuste_{c}'].sum())}/{len(sem)})")
    L += ["", "Canais com a cauda no ruído (muitos pontos com sinal corrigido <= 0; não saem do ajuste, só aviso; qualquer canal com sinal):"]
    algum = False
    for c in CANAIS:
        n = int(vp[f"aviso_cauda_no_ruido_{c}"].sum())
        if n:
            algum = True
            L.append(f"  {c}: {n} vídeos com mais de {p['frac_nao_positivo_max']:.0%} dos pontos <= 0 "
                     f"(mediana da fração: {vp.loc[vp[f'aviso_cauda_no_ruido_{c}'], f'frac_nao_positivo_{c}'].median():.2f})")
    if not algum:
        L.append("  nenhum")
    L += ["", "Espaçamento entre quadros novos na janela principal (% dos intervalos, s): " + str(_passos(rec, p["inicio_ajuste_s"]))]
    L.append("  O tempo é irregular (0,067 / 0,133 / 0,2 s). Nada foi reamostrado; o ajuste usa o tempo_s como está.")
    return "\n".join(L) + "\n"


def rodar(cfg: dict, pasta_consolidacao=None, saida=None) -> int:
    p = parametros(cfg)
    raiz = Path(cfg.get("pastas", {}).get("saida", "output"))
    pasta_in = Path(pasta_consolidacao or raiz / "consolidacao")
    for nome in ("series.csv", "videos.csv"):
        if not (pasta_in / nome).exists():
            raise FileNotFoundError(f"{pasta_in / nome} não existe (rode 'luminescencia consolidar')")
    pasta_out = Path(saida or raiz / "posprocessamento")
    pasta_out.mkdir(parents=True, exist_ok=True)

    series = pd.read_csv(pasta_in / "series.csv", encoding="utf-8-sig")
    videos = pd.read_csv(pasta_in / "videos.csv", encoding="utf-8-sig")
    sem_serie = sorted(set(videos["video"]) - set(series["video"]))
    videos = videos[videos["video"].isin(series["video"])]
    rec = recortar(series, p)
    vp = resumir_videos(videos, rec, p)

    cols = IDENT + ["frame", "tempo_s", "media_R", "media_G", "media_B", "corr_R", "corr_G", "corr_B", "saturados_pct"]
    rec[[c for c in cols if c in rec.columns]].to_csv(pasta_out / "series_pos.csv", index=False, encoding="utf-8-sig")
    vp.to_csv(pasta_out / "videos_pos.csv", index=False, encoding="utf-8-sig")
    texto = _relatorio(vp, rec, p)
    if sem_serie:
        texto += f"\nAVISO: {len(sem_serie)} vídeo(s) de videos.csv sem série em series.csv: {', '.join(sem_serie[:5])}\n"
    (pasta_out / "relatorio_posprocessamento.txt").write_text(texto, encoding="utf-8")
    print(texto)
    print(f"Arquivos em {pasta_out}: series_pos.csv, videos_pos.csv, relatorio_posprocessamento.txt")
    return 0
