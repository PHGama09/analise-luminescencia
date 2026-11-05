"""Correção da luz de fundo (etapa `luz`, v0.18.0; decisão de 04/11/2026, opção 1). Colunas PARALELAS: nada é sobrescrito.

Problema (README, "Anomalias entre amostras"): a luz geral medida nos cantos do vídeo (piso_canto_*, igual nos três canais) varia
entre regravações e alonga o decaimento medido, mesmo depois da correção do preto. Dentro da mesma preparação, ln(grandeza) cresce
quase linearmente com o canto. O efeito aparece só onde está o pó, então nenhuma medida fora da ROI o desconta (o anel foi testado
e rejeitado).

Correção (empírica; a causa física não foi confirmada):
    grandeza_luz = grandeza * exp(-b * (canto - canto_ref))
  canto      média de piso_canto_R/G/B do vídeo (output/consolidacao/videos.csv; canto real desde a v0.17.0)
  b          inclinação de ln(grandeza) contra o canto, estimada DENTRO das preparações (proporção x amostra: tira composição e
             sessão), SÓ nas amostras de treino (divisao.treino) e sem os saturados (D5). A amostra C nunca entra na calibração.
  canto_ref  mediana do canto nos vídeos de calibração (o valor corrigido é "o que a grandeza seria com essa luz")
Uma inclinação por coluna (canal x grandeza, e por razão entre canais). Só grandezas de FORMA do decaimento (tau, t50, t10,
r_Ts, auc_Hs, razões de t50); I0 e razões de I0 não mudam com a luz e não são corrigidos.

Linhas usadas na calibração de uma coluna do canal c: sinal_presente_c verdadeiro (regra MEDIDA, nunca prop_*) e valor > 0.
Valor <= 0 ou ausente é copiado sem correção. Com menos de luz.min_videos vídeos ou luz.min_preparacoes preparações, a inclinação
não é estimada (b = 0, coluna marcada).

Verificação não circular: validação cruzada deixando uma amostra de treino de fora (A <-> B). A inclinação é estimada nas outras e
aplicada na que ficou de fora; o relatório mostra, nela, a dispersão entre regravações e a correlação canto x grandeza, sem e com a
correção. (Na própria amostra de calibração a correlação vai a zero por construção e não prova nada.)

Segundo teste, também cruzado: SEPARAÇÃO entre composições. Em cada amostra de treino, os vídeos são corrigidos com a inclinação
estimada nas OUTRAS; depois a variância de ln(grandeza) é dividida em entre composições, entre sessões (preparações da mesma
composição) e entre regravações. separação = variância entre composições / (sessão + regravação), sem e com a correção. É o que a
IA precisa: distinguir composições acima do ruído. Só amostras de treino; composições presentes em pelo menos 2 delas.

Saída:
  output/tabela/luz.csv     video, canto_luz e as colunas corrigidas COM O MESMO NOME da original (para trocar em entradas.py)
  output/luz/inclinacoes.csv, validacao_cruzada.csv, separacao.csv, relatorio_luz.txt
Uso nas entradas: entradas.corrigir_luz: true (padrão false). Nada daqui altera dataset.csv ou descritores.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CANAIS = ["R", "G", "B"]
ENC = "utf-8-sig"
PADROES = {
    "grandezas_canal": ["c0_tau_medio", "cl_tau_medio", "md_t50", "md_t10", "md_r_2s", "md_r_5s", "md_r_10s", "md_r_20s",
                        "md_r_40s", "md_auc_10s", "md_auc_30s"],
    "razoes": ["razao_t50_R_G", "razao_t50_G_B", "razao_t50_R_B"],
    "min_videos": 30,
    "min_preparacoes": 5,
}


def parametros(cfg: dict) -> dict:
    p = {k: (list(v) if isinstance(v, list) else v) for k, v in PADROES.items()}
    p.update((cfg or {}).get("luz") or {})
    return p


def _canais_da_coluna(col: str) -> list[str]:
    if col.startswith("razao_"):
        return col.split("_")[-2:]
    return [col[0]]


def montar_tabela(ds: pd.DataFrame, desc: pd.DataFrame, videos: pd.DataFrame, p: dict) -> tuple[pd.DataFrame, list[str]]:
    """Uma linha por vídeo: meta, canto, sinal_presente_* e as colunas candidatas que existirem."""
    cols = [f"{c}_{g}" for c in CANAIS for g in p["grandezas_canal"]] + list(p["razoes"])
    meta = ["video", "amostra", "proporcao", "excluido_saturacao"] + [f"sinal_presente_{c}" for c in CANAIS]
    faltam = [m for m in meta if m not in ds.columns]
    if faltam:
        raise ValueError(f"dataset.csv sem as colunas {faltam}")
    t = ds[meta + [c for c in cols if c in ds.columns]]
    t = t.merge(desc[["video"] + [c for c in cols if c in desc.columns and c not in t.columns]], on="video", how="left",
                validate="one_to_one")
    cantos = [f"piso_canto_{c}" for c in CANAIS]
    if any(c not in videos.columns for c in cantos):
        raise ValueError("consolidacao/videos.csv sem piso_canto_R/G/B")
    t = t.merge(videos[["video"] + cantos], on="video", how="left", validate="one_to_one")
    t["canto_luz"] = t[cantos].mean(axis=1)
    t["prep"] = t["proporcao"].astype(str) + "|" + t["amostra"].astype(str)
    return t, [c for c in cols if c in t.columns]


def linhas_validas(t: pd.DataFrame, col: str) -> pd.Series:
    ok = t[col].gt(0) & t["canto_luz"].notna() & ~t["excluido_saturacao"].astype(bool)
    for c in _canais_da_coluna(col):
        ok &= t[f"sinal_presente_{c}"].astype(bool)
    return ok


def estimar_inclinacao(t: pd.DataFrame, col: str, amostras: list[str], p: dict) -> dict:
    """b de ln(col) ~ canto dentro das preparações, só nas `amostras`. Devolve b, n, preparações e o status."""
    s = t[linhas_validas(t, col) & t["amostra"].isin(amostras)]
    s = s[s.groupby("prep")["video"].transform("size") >= 2]
    n, n_prep = len(s), s["prep"].nunique()
    if n < p["min_videos"] or n_prep < p["min_preparacoes"]:
        return {"coluna": col, "b": 0.0, "n_videos": n, "n_preparacoes": n_prep, "status": "não estimada (poucos dados)"}
    y = np.log(s[col].astype(float))
    x = s["canto_luz"].astype(float)
    yw = y - y.groupby(s["prep"]).transform("mean")
    xw = x - x.groupby(s["prep"]).transform("mean")
    den = float((xw ** 2).sum())
    if den <= 0:
        return {"coluna": col, "b": 0.0, "n_videos": n, "n_preparacoes": n_prep, "status": "não estimada (canto sem variação)"}
    b = float((xw * yw).sum() / den)
    resid = yw - b * xw
    ep = float(np.sqrt((resid ** 2).sum() / max(n - n_prep - 1, 1) / den))
    return {"coluna": col, "b": b, "erro_padrao_b": ep, "n_videos": n, "n_preparacoes": n_prep, "status": "estimada"}


def aplicar(t: pd.DataFrame, col: str, b: float, canto_ref: float) -> pd.Series:
    """Coluna corrigida: só valores > 0 com canto conhecido; os outros são copiados."""
    v = t[col].astype(float).copy()
    ok = v.gt(0) & t["canto_luz"].notna()
    v[ok] = v[ok] * np.exp(-b * (t.loc[ok, "canto_luz"] - canto_ref))
    return v


def _dispersao_e_rho(s: pd.DataFrame, valores: pd.Series) -> tuple[float, float]:
    y = np.log(valores)
    yw = y - y.groupby(s["prep"]).transform("mean")
    xw = s["canto_luz"] - s["canto_luz"].groupby(s["prep"]).transform("mean")
    rho = float(pd.Series(yw.values).corr(pd.Series(xw.values), method="spearman")) if len(s) > 2 else np.nan
    return float(yw.std()), rho


def validacao_cruzada(t: pd.DataFrame, cols: list[str], treino: list[str], p: dict) -> pd.DataFrame:
    linhas = []
    if len(treino) < 2:
        return pd.DataFrame(linhas)
    for col in cols:
        for fora in treino:
            resto = [a for a in treino if a != fora]
            est = estimar_inclinacao(t, col, resto, p)
            s = t[linhas_validas(t, col) & (t["amostra"] == fora)]
            s = s[s.groupby("prep")["video"].transform("size") >= 2]
            if est["status"] != "estimada" or len(s) < 3:
                continue
            ref = float(t.loc[linhas_validas(t, col) & t["amostra"].isin(resto), "canto_luz"].median())
            dp0, rho0 = _dispersao_e_rho(s, s[col].astype(float))
            dp1, rho1 = _dispersao_e_rho(s, aplicar(s, col, est["b"], ref))
            linhas.append({"coluna": col, "calibrada_em": "+".join(resto), "testada_em": fora, "b": est["b"], "n_teste": len(s),
                           "dp_sem": dp0, "dp_com": dp1, "rho_sem": rho0, "rho_com": rho1})
    return pd.DataFrame(linhas)


def _decompor(s: pd.DataFrame, y: pd.Series) -> tuple[float, float, float]:
    """Variância de y entre composições, entre sessões (dentro da composição) e entre regravações (dentro da preparação)."""
    m_comp = y.groupby(s["proporcao"]).transform("mean")
    m_prep = y.groupby(s["prep"]).transform("mean")
    return float(m_comp.var(ddof=0)), float((m_prep - m_comp).var(ddof=0)), float((y - m_prep).var(ddof=0))


def separacao_cruzada(t: pd.DataFrame, cols: list[str], treino: list[str], p: dict) -> pd.DataFrame:
    """Separação entre composições sem e com a correção; cada amostra de treino é corrigida com a inclinação das outras."""
    linhas = []
    if len(treino) < 2:
        return pd.DataFrame(linhas)
    for col in cols:
        s = t[linhas_validas(t, col) & t["amostra"].isin(treino)]
        s = s[s.groupby("prep")["video"].transform("size") >= 2]
        s = s[s.groupby("proporcao")["amostra"].transform("nunique") >= 2]
        if s["proporcao"].nunique() < 2:
            continue
        corr = pd.Series(np.nan, index=s.index)
        ok = True
        for fora in treino:
            resto = [a for a in treino if a != fora]
            est = estimar_inclinacao(t, col, resto, p)
            if est["status"] != "estimada":
                ok = False
                break
            ref = float(t.loc[linhas_validas(t, col) & t["amostra"].isin(resto), "canto_luz"].median())
            m = s["amostra"] == fora
            corr[m] = aplicar(s[m], col, est["b"], ref)
        if not ok:
            continue
        c0, se0, r0 = _decompor(s, np.log(s[col].astype(float)))
        c1, se1, r1 = _decompor(s, np.log(corr))
        linhas.append({"coluna": col, "n": len(s), "separacao_sem": c0 / (se0 + r0), "separacao_com": c1 / (se1 + r1),
                       "fracao_sessao_sem": se0 / (c0 + se0 + r0), "fracao_sessao_com": se1 / (c1 + se1 + r1),
                       "fracao_regravacao_sem": r0 / (c0 + se0 + r0), "fracao_regravacao_com": r1 / (c1 + se1 + r1)})
    sep = pd.DataFrame(linhas)
    if len(sep):
        sep["ganho_pct"] = (sep["separacao_com"] / sep["separacao_sem"] - 1) * 100
    return sep


def _relatorio(inc: pd.DataFrame, cv: pd.DataFrame, canto_ref: float, treino: list[str], t: pd.DataFrame,
               sep: pd.DataFrame | None = None) -> str:
    L = ["CORREÇÃO DA LUZ DE FUNDO (colunas paralelas; dataset.csv e descritores.csv não mudam)", "=" * 78,
         f"Calibração: amostras {', '.join(treino)} (divisao.treino), sem saturados, dentro das preparações. C não entra.",
         f"canto_ref = {canto_ref:.3f} (mediana do canto médio RGB na calibração). Correção: x * exp(-b * (canto - canto_ref)).",
         "Uso: entradas.corrigir_luz: true troca as colunas abaixo pelas de output/tabela/luz.csv (padrão: false).", "",
         "Inclinações (b em % por nível de canto; ep = erro padrão):"]
    for r in inc.itertuples():
        ep = f" ± {r.erro_padrao_b * 100:.1f}" if r.status == "estimada" else ""
        L.append(f"  {r.coluna:18s} b = {r.b * 100:+6.2f}%{ep:9s} n = {r.n_videos:>3} em {r.n_preparacoes:>2} preparações  [{r.status}]")
    L += ["", "Validação cruzada (calibra numa amostra de treino, testa na outra): na amostra de teste,",
          "  dp = dispersão de ln(grandeza) entre regravações da mesma preparação; rho = Spearman canto x grandeza dentro da preparação."]
    if len(cv):
        for r in cv.itertuples():
            L.append(f"  {r.coluna:18s} {r.calibrada_em}->{r.testada_em}: dp {r.dp_sem * 100:5.1f}% -> {r.dp_com * 100:5.1f}%   "
                     f"rho {r.rho_sem:+.2f} -> {r.rho_com:+.2f}   (n = {r.n_teste})")
        resumo = cv.assign(ganho=1 - cv["dp_com"] / cv["dp_sem"])
        L += ["", f"Resumo: a dispersão caiu em {int((resumo['ganho'] > 0).sum())} de {len(resumo)} testes; "
                  f"queda mediana {resumo['ganho'].median() * 100:.0f}%; |rho| mediano {cv['rho_sem'].abs().median():.2f} -> "
                  f"{cv['rho_com'].abs().median():.2f}."]
        piora = resumo[resumo["ganho"] < 0]
        if len(piora):
            L.append("Testes em que a correção PIOROU a dispersão: " + ", ".join(f"{r.coluna} ({r.calibrada_em}->{r.testada_em})"
                                                                              for r in piora.itertuples()))
    else:
        L.append("  (precisa de pelo menos 2 amostras de treino)")
    L += ["", "Separação entre composições (cruzada: cada amostra de treino corrigida com a inclinação das outras).",
          "  separação = variância entre composições / (variância entre sessões + entre regravações), em ln(grandeza)."]
    if sep is not None and len(sep):
        for r in sep.itertuples():
            L.append(f"  {r.coluna:18s} {r.separacao_sem:7.2f} -> {r.separacao_com:7.2f}  ({r.ganho_pct:+6.1f}%)   "
                     f"sessão {r.fracao_sessao_sem * 100:4.1f}% -> {r.fracao_sessao_com * 100:4.1f}%   "
                     f"regravação {r.fracao_regravacao_sem * 100:4.1f}% -> {r.fracao_regravacao_com * 100:4.1f}%")
        canal = sep["coluna"].map(lambda c: "razões" if c.startswith("razao_") else c[0])
        L += ["", f"Resumo: a separação melhorou em {int((sep['ganho_pct'] > 0).sum())} de {len(sep)} colunas; ganho mediano "
                  f"{sep['ganho_pct'].median():+.1f}%; por grupo: "
                  + ", ".join(f"{g} {v:+.1f}%" for g, v in sep.groupby(canal)["ganho_pct"].median().items()) + "."]
        pior = sep[sep["ganho_pct"] < 0]
        if len(pior):
            L.append("Colunas em que a separação PIOROU: " + ", ".join(f"{r.coluna} ({r.ganho_pct:+.1f}%)" for r in pior.itertuples()))
    else:
        L.append("  (precisa de pelo menos 2 amostras de treino e composições presentes em mais de uma)")
    L += ["", "Canto médio por amostra (mediana): " + str(t.groupby("amostra")["canto_luz"].median().round(2).to_dict()),
          "Limites: correção empírica, causa não confirmada (teste de luz acesa x apagada pendente); supõe efeito linear e igual",
          "entre composições; não remove toda a diferença entre sessões. Decidir com/sem ANTES do pré-registro."]
    return "\n".join(L) + "\n"


def rodar(cfg: dict, pasta_tabela=None, pasta_consolidacao=None, saida=None) -> int:
    p = parametros(cfg)
    raiz = Path((cfg.get("pastas") or {}).get("saida", "output"))
    tab = Path(pasta_tabela or raiz / "tabela")
    cons = Path(pasta_consolidacao or raiz / "consolidacao")
    out = Path(saida or raiz / "luz")
    for arq in (tab / "dataset.csv", tab / "descritores.csv", cons / "videos.csv"):
        if not arq.is_file():
            raise FileNotFoundError(f"{arq} não existe (rode 'luminescencia tudo' antes)")
    ds = pd.read_csv(tab / "dataset.csv", encoding=ENC, low_memory=False)
    desc = pd.read_csv(tab / "descritores.csv", encoding=ENC)
    videos = pd.read_csv(cons / "videos.csv", encoding=ENC, low_memory=False)
    treino = [str(a) for a in ((cfg.get("divisao") or {}).get("treino") or ["A", "B"])]

    t, cols = montar_tabela(ds, desc, videos, p)
    calib = t["amostra"].isin(treino) & ~t["excluido_saturacao"].astype(bool)
    canto_ref = float(t.loc[calib, "canto_luz"].median())
    inc = pd.DataFrame([estimar_inclinacao(t, c, treino, p) for c in cols])
    luz = t[["video", "canto_luz"]].copy()
    for r in inc.itertuples():
        luz[r.coluna] = aplicar(t, r.coluna, r.b, canto_ref).round(4)
    cv = validacao_cruzada(t, cols, treino, p)
    sep = separacao_cruzada(t, cols, treino, p)

    out.mkdir(parents=True, exist_ok=True)
    tab.mkdir(parents=True, exist_ok=True)
    luz.to_csv(tab / "luz.csv", index=False, encoding=ENC)
    inc.assign(canto_ref=canto_ref).to_csv(out / "inclinacoes.csv", index=False, encoding=ENC)
    cv.to_csv(out / "validacao_cruzada.csv", index=False, encoding=ENC)
    sep.to_csv(out / "separacao.csv", index=False, encoding=ENC)
    texto = _relatorio(inc, cv, canto_ref, treino, t, sep)
    (out / "relatorio_luz.txt").write_text(texto, encoding="utf-8")
    print(texto)
    print(f"Arquivos: {tab / 'luz.csv'}; {out}/inclinacoes.csv, validacao_cruzada.csv, separacao.csv, relatorio_luz.txt")
    return 0
