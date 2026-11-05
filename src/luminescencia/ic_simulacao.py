"""Simulação do intervalo de confiança do erro em C (D6, só análise).

Pergunta: com as 16 composições em 1 preparação de validação, qual a largura do
IC 95% do MAE e quanto ela encolhe com 2 ou 3 preparações? E qual a chance de
passar o critério D3 (<= 70% / <= 50% do MAE da média) por sorte?

Só entram componentes MEDIDOS em output/deriva (deriva_variancia.csv e
deriva_por_grandeza.csv) e a grade de composições (nomes das proporções, sem
prop_*). Nenhum número de resultado está escrito aqui: tudo vem dos CSVs ou do
config. Não treina nada, não lê dataset.csv, não altera nenhum arquivo existente.

Modelo (suposições declaradas também no relatório): o erro escalar da composição
i na preparação p é
    e_ip = d_i + S_p + U_ip + R_ip / sqrt(n_reg)
  d_i  erro sistemático da composição (igual em todas as preparações)   <- frac_proporcao
  S_p  efeito da preparação compartilhado por todas as composições      <- frac_amostra * frac_offset_comum
  U_ip efeito da preparação específico da composição                    <- frac_amostra * (1 - frac_offset_comum)
  R_ip ruído entre regravações (média de n_reg por preparação)          <- frac_regravacao
Os componentes são normais, independentes, com variâncias proporcionais às
frações medidas, e a escala total é calibrada para que E|e| (1 preparação) seja
o MAE verdadeiro hipotético m da grade.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import erf

DIVISOR_PROP = 1.5  # protocolo: f = prop / 1,5 (a soma das proporções é 1,5)
_CANAIS = ("R", "G", "B")


# ---------------------------------------------------------------- composições
def parse_proporcao(nome: str) -> np.ndarray:
    """'R0.5_G1' -> fração (R, G, B) = prop/1,5; canal ausente = 0."""
    prop = dict.fromkeys(_CANAIS, 0.0)
    for parte in str(nome).split("_"):
        m = re.fullmatch(r"([RGB])(\d+(?:\.\d+)?)", parte.strip())
        if not m:
            raise ValueError(f"proporção não reconhecida: {nome!r}")
        prop[m.group(1)] += float(m.group(2))
    v = np.array([prop[c] for c in _CANAIS])
    if abs(v.sum() - DIVISOR_PROP) > 1e-6:
        raise ValueError(f"proporção {nome!r}: soma {v.sum():g} != {DIVISOR_PROP:g}")
    return v / DIVISOR_PROP


def baseline_media(nomes: list[str]) -> tuple[float, np.ndarray]:
    """MAE da previsão constante (média das frações da grade) e o erro por composição."""
    F = np.vstack([parse_proporcao(n) for n in nomes])
    erro_i = np.abs(F - F.mean(axis=0)).mean(axis=1)
    return float(erro_i.mean()), erro_i


def ler_proporcoes(pasta_deriva: Path) -> list[str]:
    """Composições = união das de deriva_celulas.csv e deriva_descartadas.csv."""
    nomes: set[str] = set()
    for arq in ("deriva_celulas.csv", "deriva_descartadas.csv"):
        p = pasta_deriva / arq
        if p.exists():
            nomes |= set(pd.read_csv(p, encoding="utf-8-sig")["proporcao"].astype(str))
    if not nomes:
        raise FileNotFoundError(f"sem deriva_celulas.csv em {pasta_deriva}; rode 'luminescencia deriva'")
    return sorted(nomes)


# ---------------------------------------------------------------- componentes
def componentes_variancia(df: pd.DataFrame, grandezas: list[str] | None, quantil: float) -> dict:
    """Frações (proporção, amostra, regravação; soma 1) e parte compartilhada, do deriva_variancia."""
    d = df if not grandezas else df[df["grandeza"].isin(grandezas)]
    d = d.dropna(subset=["frac_proporcao", "frac_amostra", "frac_regravacao", "frac_offset_comum"])
    if d.empty:
        raise ValueError("nenhuma linha de deriva_variancia.csv para as grandezas pedidas")
    q = {c: float(d[c].quantile(quantil)) for c in
         ("frac_proporcao", "frac_amostra", "frac_regravacao", "frac_offset_comum")}
    soma = q["frac_proporcao"] + q["frac_amostra"] + q["frac_regravacao"]
    return {"w_prop": q["frac_proporcao"] / soma, "w_am": q["frac_amostra"] / soma,
            "w_reg": q["frac_regravacao"] / soma, "offset": min(max(q["frac_offset_comum"], 0.0), 1.0),
            "n_linhas": int(len(d))}


def coeficientes(w: dict, n_reg: int) -> dict:
    """Desvios-padrão dos 4 componentes POR UNIDADE de MAE verdadeiro (E|e| de 1 preparação = m)."""
    pesos = {"d": w["w_prop"], "S": w["w_am"] * w["offset"],
             "U": w["w_am"] * (1 - w["offset"]), "R": w["w_reg"] / n_reg}
    k = sum(pesos.values())
    if k <= 0:
        raise ValueError("variância total nula")
    tau = np.sqrt(np.pi / 2)  # E|N(0, tau^2 m^2)| = m
    return {c: float(tau * np.sqrt(p / k)) for c, p in pesos.items()}


def _folded(mu: np.ndarray, sigma: float) -> np.ndarray:
    """E|N(mu, sigma^2)| exato."""
    if sigma <= 0:
        return np.abs(mu)
    return sigma * np.sqrt(2 / np.pi) * np.exp(-mu**2 / (2 * sigma**2)) + mu * erf(mu / (sigma * np.sqrt(2)))


# ---------------------------------------------------------------- simulação
def simular(coef: dict, grade_m: list[float], n_prep: int, n_comp: int, replicas: int,
            semente: int, nivel_ic: float, limites: dict, mae_media: float,
            erro_media_i: np.ndarray) -> list[dict]:
    """Uma linha por MAE verdadeiro m. Números aleatórios comuns entre os m (todos os
    componentes escalam linearmente com m). limites = {'minimo': razao, 'forte': razao}."""
    rng = np.random.default_rng([semente, n_prep])
    Zd = rng.standard_normal((replicas, 1, n_comp))
    Zs = rng.standard_normal((replicas, n_prep, 1))
    Zu = rng.standard_normal((replicas, n_prep, n_comp))
    Zr = rng.standard_normal((replicas, n_prep, n_comp))
    sig_prep = float(np.sqrt(coef["S"] ** 2 + coef["U"] ** 2 + coef["R"] ** 2))
    a = (1 - nivel_ic) / 2
    linhas = []
    for m in grade_m:
        e = m * (coef["d"] * Zd + coef["S"] * Zs + coef["U"] * Zu + coef["R"] * Zr)
        ae = np.abs(e)
        obs = ae.mean(axis=(1, 2))
        mu = m * coef["d"] * Zd[:, 0, :]
        verd = _folded(mu, m * sig_prep).mean(axis=1)  # MAE verdadeiro da grade nesta réplica
        dev = obs - verd
        q_lo, q_hi = np.quantile(dev, [a, 1 - a])
        pior_que_media = (ae.mean(axis=1) > erro_media_i[None, :]).any(axis=1)  # condição (b)
        linha = {"mae_verdadeiro": m, "n_preparacoes": n_prep, "ic_largura": float(q_hi - q_lo),
                 "ic_largura_rel": float((q_hi - q_lo) / m), "dev_dp": float(dev.std(ddof=1)),
                 "mae_obs_medio": float(obs.mean())}
        for nivel, razao in limites.items():
            lim = razao * mae_media
            linha[f"limite_{nivel}"] = lim
            linha[f"cumpre_{nivel}"] = bool(m <= lim)
            linha[f"p_{nivel}"] = float((obs <= lim).mean())
            linha[f"p_{nivel}_e_b"] = float(((obs <= lim) & ~pior_que_media).mean())
        linhas.append(linha)
    return linhas


# ---------------------------------------------------------------- relatório
def _tabela(df: pd.DataFrame, col: str, fmt: str) -> str:
    t = df.pivot(index="mae_verdadeiro", columns="n_preparacoes", values=col)
    cab = "MAE verd.  " + "  ".join(f"{p:>8d} prep" for p in t.columns)
    corpo = [f"{m:8.3f}  " + "  ".join(f"{v:>12{fmt}}" for v in t.loc[m]) for m in t.index]
    return "\n".join([cab] + corpo)


def rodar(cfg: dict, pasta_deriva: str | Path | None = None, saida: str | Path | None = None):
    c = cfg["ic_simulacao"]
    base = Path(cfg.get("pastas", {}).get("saida", "output"))
    pasta_deriva = Path(pasta_deriva) if pasta_deriva else base / c.get("pasta_deriva", "deriva")
    saida = Path(saida) if saida else base / "ic_simulacao"
    saida.mkdir(parents=True, exist_ok=True)

    nomes = c.get("proporcoes") or ler_proporcoes(pasta_deriva)
    mae_media, erro_i = baseline_media(list(nomes))
    var = pd.read_csv(pasta_deriva / "deriva_variancia.csv", encoding="utf-8-sig")
    grandezas = c.get("grandezas")
    limites = {"minimo": float(c["nivel_minimo_razao_max"]), "forte": float(c["nivel_forte_razao_max"])}
    n_reg = int(c["n_regravacoes"])

    cv = None
    pg = pasta_deriva / "deriva_por_grandeza.csv"
    if pg.exists():
        g = pd.read_csv(pg, encoding="utf-8-sig")
        g = g[g["grandeza"].isin(grandezas)] if grandezas else g
        if len(g):
            cv = (float(g["cv_reg_mediano"].median()), float(g["diferenca_rel_mediana"].median()))

    comp_linhas, todas = [], []
    for cen, q in c["cenarios_quantil"].items():
        w = componentes_variancia(var, grandezas, float(q))
        coef = coeficientes(w, n_reg)
        comp_linhas.append({"cenario": cen, "quantil": q, **w, **{f"dp_{k}_por_m": v for k, v in coef.items()}})
        for P in c["n_preparacoes"]:
            for lin in simular(coef, list(c["mae_verdadeiro"]), int(P), len(nomes), int(c["replicas"]),
                               int(c["semente"]), float(c["nivel_ic"]), limites, mae_media, erro_i):
                todas.append({"cenario": cen, **lin})
    comp = pd.DataFrame(comp_linhas)
    res = pd.DataFrame(todas)
    comp.to_csv(saida / "ic_componentes.csv", index=False, encoding="utf-8-sig")
    res.to_csv(saida / "ic_simulacao.csv", index=False, encoding="utf-8-sig")

    L = ["IC DO ERRO EM C POR SIMULAÇÃO (D6; só análise; nada treinado; D6 provisória)",
         "=" * 78,
         f"Composições: {len(nomes)} | réplicas: {c['replicas']} | semente: {c['semente']} | IC {c['nivel_ic']:.0%} | "
         f"regravações por preparação: {n_reg}",
         f"Grandezas usadas nas frações de variância: {grandezas or 'todas'}",
         f"MAE da média (calculado da grade, f = prop/1,5): {mae_media:.4f} | "
         f"limite mínimo {limites['minimo']:.0%} = {limites['minimo'] * mae_media:.4f} | "
         f"forte {limites['forte']:.0%} = {limites['forte'] * mae_media:.4f}",
         "Erro da média por composição (|f - média|, média dos 3 canais; é o teto da condição b): "
         + ", ".join(f"{n} {e:.3f}" for n, e in sorted(zip(nomes, erro_i), key=lambda t: t[1])),
         ""]
    if cv:
        L.append(f"Medido em deriva_por_grandeza.csv (mediana das grandezas usadas): CV entre regravações = {cv[0]:.3f}; "
                 f"diferença relativa entre amostras = {cv[1]:.3f}.")
    L += ["", "SUPOSIÇÕES (declaradas; o resultado só vale dentro delas)",
          "-" * 78,
          "1. Erro escalar por composição e preparação: e = d_i + S_p + U_ip + R_ip/sqrt(n_reg), componentes normais e independentes.",
          "2. As variâncias dos componentes são proporcionais às frações medidas na DESCRIÇÃO (log) das grandezas: d_i = entre proporções, "
          "S_p = entre amostras x deslocamento comum, U_ip = entre amostras x (1 - deslocamento comum), R = entre regravações. "
          "NÃO foi medido que o erro de um modelo se reparte assim; é uma suposição.",
          "3. d_i (erro sistemático da composição) é o mesmo em todas as preparações: não encolhe com mais preparações. S, U e R encolhem.",
          "4. Escala: E|e| de uma preparação = MAE verdadeiro m (grade do config), esperança também sobre o sorteio de d_i; logo P(passar) mistura a incerteza dos d_i com a da amostragem, e a largura do IC (item 5) só a da amostragem, condicionada aos d_i. Com P preparações, MAE observado = média dos |e| em P x composições.",
          "5. Largura do IC = quantil superior - inferior de (MAE observado - MAE verdadeiro da grade daquela réplica; folded normal exata).",
          "6. Passa 'a' se MAE observado <= razão x MAE da média; 'a e b' se, além disso, nenhuma composição (média das P preparações) fica pior que a da média. "
          "A condição (c) (linha de base física, pareada) NÃO é simulada.",
          "7. O erro escalar substitui o erro por fração (R, G, B); o MAE da média usa a mesma grade (média das frações das composições).",
          ""]
    L += ["COMPONENTES MEDIDOS (deriva_variancia.csv)", "-" * 78, comp.round(4).to_string(index=False), ""]
    for cen in comp["cenario"]:
        r = res[res["cenario"] == cen]
        L += [f"=== CENÁRIO {cen} ===", "",
              "Largura do IC 95% do MAE (absoluta) por nº de preparações:", _tabela(r, "ic_largura", ".4f"), "",
              "Largura relativa ao MAE verdadeiro:", _tabela(r, "ic_largura_rel", ".3f"), ""]
        for nivel in limites:
            L += [f"P(MAE observado <= {limites[nivel]:.0%} da média) [cumpre de verdade: m <= {limites[nivel] * mae_media:.4f}]:",
                  _tabela(r, f"p_{nivel}", ".3f"), "",
                  f"P(passa 'a' {nivel} e 'b'):", _tabela(r, f"p_{nivel}_e_b", ".3f"), ""]
        p1, pmax = int(r["n_preparacoes"].min()), int(r["n_preparacoes"].max())
        w1 = r[r["n_preparacoes"] == p1].set_index("mae_verdadeiro")["ic_largura"]
        wm = r[r["n_preparacoes"] == pmax].set_index("mae_verdadeiro")["ic_largura"]
        L.append(f"Razão largura({pmax} prep)/largura({p1} prep), por MAE verdadeiro: "
                 + ", ".join(f"{m:.3f}: {wm[m] / w1[m]:.2f}" for m in w1.index))
        for nivel in limites:
            falso = r[(~r[f"cumpre_{nivel}"]) & (r["n_preparacoes"] == p1)]
            if len(falso):
                L.append(f"Passagem {nivel} por sorte com {p1} prep (m acima do limite): máximo {falso[f'p_{nivel}'].max():.3f} "
                         f"em m = {falso.loc[falso[f'p_{nivel}'].idxmax(), 'mae_verdadeiro']:.3f}.")
        L.append("")
    (saida / "relatorio_ic_simulacao.txt").write_text("\n".join(L), encoding="utf-8", newline="\n")
    print(f"Relatório: {saida / 'relatorio_ic_simulacao.txt'}")
    return 0
