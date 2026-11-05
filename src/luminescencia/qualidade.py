"""Qualidade e viés dos diagnósticos do ajuste (BLOCO 2 parte B, comando `qualidade`). SÓ ANÁLISE: nenhum peso é aplicado.

Pergunta: os diagnósticos (<canal>_<cl|c0|md>_{ok,r2,rmse,inicio_s,fim_s,n_pontos}), bloqueados como ENTRADA pela E1, servem
como PESO de vídeo? Um peso por um diagnóstico que depende da composição (proporção) favorece composições e cria viés.

Por diagnóstico e canal o módulo mede, só em treino (divisao.treino, A+B) e sem os saturados (a amostra C nunca é aberta aqui):
  (i)   distribuição por proporção e por amostra;
  (ii)  η² da proporção e da amostra, com permutação de semente fixa. A unidade da permutação é a PREPARAÇÃO (proporção x amostra),
        nunca o vídeo: as regravações da mesma preparação ficam juntas, como manda o protocolo. O η² bruto cresce só por ter
        muitos grupos e por haver correlação dentro da preparação; por isso o veredito usa o η² AJUSTADO (observado menos a média
        do η² sob permutação) e exige p < alfa. A amostra é medida como variância ADICIONAL explicada depois da proporção
        (permutação da amostra dentro da proporção);
  (iii) vídeos candidatos a baixo peso (listados, não aplicados) e quantas preparações perderiam peso.

Veredito por diagnóstico (calculado, com os limiares do config, seção `qualidade`):
  "usável como peso"   η² ajustado da proporção < limite_eta2_prop
  "viés"               η² ajustado >= limite e p < alfa
  "inconclusivo"       η² ajustado >= limite, mas p >= alfa (poucos dados para separar do acaso)
  "sem variação"       um valor só: não serve de peso
  "dados insuficientes" poucas linhas ou proporções com valor
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .entradas import colunas_de_diagnostico

_RE_COL = re.compile(r"^([RGB])_(cl|c0|md)_(.+)$")
PADROES = {
    "amostras": None, "n_permutacoes": 2000, "semente": 20261104,
    "limite_eta2_prop": 0.10, "limite_eta2_amostra": 0.10, "alfa": 0.05,
    "min_linhas": 30, "min_proporcoes": 4,
    "quantil_baixo": 0.10, "fracao_prep_maioria": 0.5,
    "direcao": {"r2": "maior", "ok": "maior", "n_pontos": "maior", "fim_s": "maior", "rmse": "menor", "inicio_s": "menor"},
}
V_USAVEL, V_VIES, V_INCONCL = "usável como peso", "viés", "inconclusivo"
V_CONST, V_POUCO = "sem variação", "dados insuficientes"


def parametros(cfg: dict) -> dict:
    p = {k: (dict(v) if isinstance(v, dict) else v) for k, v in PADROES.items()}
    q = dict(cfg.get("qualidade", {}) or {})
    direcao = dict(p["direcao"])
    direcao.update(q.pop("direcao", None) or {})
    p.update(q)
    p["direcao"] = direcao
    if p["amostras"] is None:
        p["amostras"] = list((cfg.get("divisao", {}) or {}).get("treino", ["A", "B"]))
    if not isinstance(p["n_permutacoes"], int) or p["n_permutacoes"] < 50:
        raise ValueError("qualidade.n_permutacoes deve ser inteiro >= 50")
    for chave in ("limite_eta2_prop", "limite_eta2_amostra", "alfa", "quantil_baixo", "fracao_prep_maioria"):
        if not 0 < float(p[chave]) < 1:
            raise ValueError(f"qualidade.{chave} deve estar entre 0 e 1")
    if any(v not in ("maior", "menor") for v in direcao.values()):
        raise ValueError("qualidade.direcao: use 'maior' ou 'menor' (qual lado é o melhor)")
    return p


def _num(s: pd.Series) -> pd.Series:
    """Coluna numérica; True/False viram 1/0, ausente vira NaN."""
    try:
        return s.astype("float64")
    except (TypeError, ValueError):
        mapa = s.astype(str).str.strip().str.lower().map({"true": 1.0, "false": 0.0})
        return mapa.fillna(pd.to_numeric(s, errors="coerce")).astype("float64")


def _eta2(x: np.ndarray, codes: np.ndarray, k: int) -> float:
    ss_t = float(((x - x.mean()) ** 2).sum())
    if ss_t <= 0:
        return float("nan")
    n = np.bincount(codes, minlength=k)
    s = np.bincount(codes, weights=x, minlength=k)
    ok = n > 0
    return float((n[ok] * (s[ok] / n[ok] - x.mean()) ** 2).sum() / ss_t)


def _extra_amostra(x_res: np.ndarray, am: np.ndarray, prop: np.ndarray, kp: int, ka: int, ss_t: float) -> float:
    """Fração da variância total explicada pela amostra DEPOIS da proporção (Frisch-Waugh: tudo residualizado pela proporção)."""
    if ka < 2 or ss_t <= 0:
        return float("nan")
    n = np.bincount(prop, minlength=kp).astype(float)
    n[n == 0] = 1.0
    cols = []
    for lvl in range(ka - 1):
        d = (am == lvl).astype(float)
        cols.append(d - (np.bincount(prop, weights=d, minlength=kp) / n)[prop])
    d_res = np.column_stack(cols)
    beta = np.linalg.lstsq(d_res, x_res, rcond=None)[0]
    return float(((d_res @ beta) ** 2).sum() / ss_t)


def _preparar(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Só treino (p['amostras']) e sem saturados; devolve cópia com os códigos de proporção, amostra e preparação."""
    d = df[df["amostra"].isin(p["amostras"])].copy()
    if "excluido_saturacao" in d.columns:
        d = d[~d["excluido_saturacao"].astype(str).str.lower().isin(["true", "1", "1.0"])]
    d = d.reset_index(drop=True)
    d["proporcao"] = d["proporcao"].astype(str)
    d["amostra"] = d["amostra"].astype(str)
    d["_prep"] = d["proporcao"] + " | " + d["amostra"]
    return d


def _permutacoes(prep_prop: np.ndarray, prep_am: np.ndarray, n: int, semente: int) -> tuple[np.ndarray, np.ndarray]:
    """Permutações no nível da PREPARAÇÃO: a proporção circula entre preparações; a amostra circula DENTRO da proporção."""
    rng = np.random.default_rng(semente)
    por_prop = [np.flatnonzero(prep_prop == g) for g in np.unique(prep_prop)]
    pp = np.empty((n, len(prep_prop)), dtype=np.int64)
    pa = np.empty_like(pp)
    for b in range(n):
        pp[b] = rng.permutation(prep_prop)
        pa[b] = prep_am
        for idx in por_prop:
            if len(idx) > 1:
                pa[b, idx] = rng.permutation(prep_am[idx])
    return pp, pa


def _p_perm(obs: float, nulo: np.ndarray) -> float:
    nulo = nulo[~np.isnan(nulo)]
    return float("nan") if np.isnan(obs) or nulo.size == 0 else float((1 + (nulo >= obs).sum()) / (1 + nulo.size))


def _veredito(status: str, aj: float, pval: float, p: dict) -> str:
    if status != "ok":
        return status
    if not aj >= p["limite_eta2_prop"]:
        return V_USAVEL
    return V_VIES if pval < p["alfa"] else V_INCONCL


def _candidatos(x: pd.Series, direcao: str, p: dict) -> pd.Series:
    """Máscara dos vídeos candidatos a baixo peso: estritamente piores que o quantil_baixo da distribuição (binário: o valor pior, se minoritário)."""
    v = x.dropna()
    if v.nunique() < 2:
        return pd.Series(False, index=x.index)
    lado = v if direcao == "maior" else -v
    if set(v.unique()) <= {0.0, 1.0}:
        pior = lado == lado.min()
        m = pd.Series(False, index=x.index)
        if pior.mean() <= 0.5:
            m.loc[pior.index] = pior
        return m
    corte = lado.quantile(p["quantil_baixo"])
    m = pd.Series(False, index=x.index)
    m.loc[lado.index] = lado < corte
    return m


def analisar(df: pd.DataFrame, colunas: list[str], p: dict) -> dict:
    """Núcleo (sem arquivos). `df` = dataset (+ descritores) com amostra, proporcao, video e as colunas de diagnóstico."""
    d = _preparar(df, p)
    if d.empty:
        raise ValueError(f"nenhuma linha de treino ({p['amostras']}) sem saturação")
    props = sorted(d["proporcao"].unique())
    amostras = sorted(d["amostra"].unique())
    d["_cp"] = d["proporcao"].map({v: i for i, v in enumerate(props)})
    d["_ca"] = d["amostra"].map({v: i for i, v in enumerate(amostras)})
    d["_ip"], preps = pd.factorize(d["_prep"])
    prep = d.groupby("_ip").agg(cp=("_cp", "first"), ca=("_ca", "first"))
    perm_p, perm_a = _permutacoes(prep["cp"].to_numpy(), prep["ca"].to_numpy(), p["n_permutacoes"], p["semente"])
    kp, ka = len(props), len(amostras)

    eta_linhas, dist_p, dist_a, cand, resumo = [], [], [], [], []
    total_preps = int(len(preps))
    for col in colunas:
        m = _RE_COL.match(col)
        canal, caminho, diag = m.groups()
        x_all = _num(d[col])
        ok = x_all.notna().to_numpy()
        x = x_all.to_numpy()[ok]
        ip, cp, ca = d["_ip"].to_numpy()[ok], d["_cp"].to_numpy()[ok], d["_ca"].to_numpy()[ok]
        n = int(ok.sum())
        n_prop = int(len(np.unique(cp)))
        n_dist = int(len(np.unique(x))) if n else 0
        status = "ok"
        if n < p["min_linhas"] or n_prop < p["min_proporcoes"]:
            status = V_POUCO
        elif n_dist < 2:
            status = V_CONST
        reg = {"diagnostico": col, "canal": canal, "caminho": caminho, "diag": diag, "n": n, "n_proporcoes": n_prop,
               "n_preparacoes": int(len(np.unique(ip))), "n_distintos": n_dist}
        nan = float("nan")
        e_p = nulo_p = e_a = e_x = nulo_x = nan
        pv_p = pv_x = nan
        if status == "ok":
            e_p = _eta2(x, cp, kp)
            nulo_p = np.array([_eta2(x, perm_p[b][ip], kp) for b in range(p["n_permutacoes"])])
            pv_p = _p_perm(e_p, nulo_p)
            e_a = _eta2(x, ca, ka)
            ss_t = float(((x - x.mean()) ** 2).sum())
            n_g = np.bincount(cp, minlength=kp).astype(float)
            n_g[n_g == 0] = 1.0
            x_res = x - (np.bincount(cp, weights=x, minlength=kp) / n_g)[cp]
            e_x = _extra_amostra(x_res, ca, cp, kp, ka, ss_t)
            nulo_x = np.array([_extra_amostra(x_res, perm_a[b][ip], cp, kp, ka, ss_t) for b in range(p["n_permutacoes"])])
            pv_x = _p_perm(e_x, nulo_x)
        aj = e_p - float(np.nanmean(nulo_p)) if status == "ok" else nan
        aj_x = e_x - float(np.nanmean(nulo_x)) if status == "ok" and not np.isnan(e_x) else nan
        reg.update({"eta2_prop": e_p, "eta2_prop_nulo_medio": float(np.nanmean(nulo_p)) if status == "ok" else nan,
                    "eta2_prop_ajustado": aj, "p_prop": pv_p,
                    "eta2_amostra": e_a, "eta2_amostra_extra": e_x, "eta2_amostra_extra_ajustado": aj_x, "p_amostra_extra": pv_x})
        reg["aviso_amostra"] = ("depende da amostra (deriva de sessão)" if (not np.isnan(aj_x) and aj_x >= p["limite_eta2_amostra"] and pv_x < p["alfa"]) else "")
        # a AUSÊNCIA do diagnóstico (canal sem ajuste) depende da proporção? Aí o peso ficaria indefinido de forma não aleatória
        aus = (~ok).astype(float)
        reg["n_ausentes"] = int(aus.sum())
        e_au = pv_au = nan
        if 0 < aus.sum() < len(aus):
            e_au = _eta2(aus, d["_cp"].to_numpy(), kp)
            nulo_au = np.array([_eta2(aus, perm_p[b][d["_ip"].to_numpy()], kp) for b in range(p["n_permutacoes"])])
            pv_au = _p_perm(e_au, nulo_au)
            e_au = e_au - float(np.nanmean(nulo_au))
        reg["eta2_prop_ausencia_ajustado"], reg["p_prop_ausencia"] = e_au, pv_au
        reg["status"] = status
        reg["veredito"] = _veredito(status, aj, pv_p, p)
        eta_linhas.append(reg)

        # distribuição por proporção e por amostra
        for chave, destino in (("proporcao", dist_p), ("amostra", dist_a)):
            for g, sub in d.assign(_x=x_all).groupby(chave):
                v = sub["_x"].dropna()
                destino.append({"diagnostico": col, "canal": canal, "caminho": caminho, "diag": diag, chave: g, "n_videos": len(sub),
                                "n_validos": len(v), "n_ausentes": len(sub) - len(v),
                                "media": v.mean() if len(v) else nan, "dp": v.std(ddof=1) if len(v) > 1 else nan,
                                "mediana": v.median() if len(v) else nan, "min": v.min() if len(v) else nan, "max": v.max() if len(v) else nan})

        # candidatos a baixo peso (só listados)
        if status == "ok":
            mc = _candidatos(x_all, p["direcao"].get(diag, "maior"), p)
            sub = d.loc[mc, ["video", "proporcao", "amostra", "_prep"]].copy()
            sub.insert(0, "diagnostico", col)
            sub["valor"] = x_all[mc].to_numpy()
            sub["veredito_diagnostico"] = reg["veredito"]
            cand.append(sub)
            por_prep = d.assign(_c=mc.to_numpy()).groupby("_prep")["_c"].agg(["sum", "size"])
            frac = por_prep["sum"] / por_prep["size"]
            por_prop_c = sub.groupby("proporcao").size()
            resumo.append({"diagnostico": col, "veredito": reg["veredito"], "n_candidatos": int(mc.sum()),
                           "preparacoes_com_candidato": int((por_prep["sum"] > 0).sum()),
                           "preparacoes_maioria_candidata": int((frac >= p["fracao_prep_maioria"]).sum()),
                           "preparacoes_total": total_preps, "n_proporcoes_com_candidato": int(len(por_prop_c)),
                           "proporcao_mais_afetada": por_prop_c.idxmax() if len(por_prop_c) else "",
                           "fracao_dos_candidatos_na_mais_afetada": float(por_prop_c.max() / por_prop_c.sum()) if len(por_prop_c) else nan})
    colunas_cand = ["diagnostico", "video", "proporcao", "amostra", "_prep", "valor", "veredito_diagnostico"]
    return {"eta2": pd.DataFrame(eta_linhas), "dist_proporcao": pd.DataFrame(dist_p), "dist_amostra": pd.DataFrame(dist_a),
            "candidatos": (pd.concat(cand, ignore_index=True) if cand else pd.DataFrame(columns=colunas_cand)).rename(columns={"_prep": "preparacao"}),
            "candidatos_resumo": pd.DataFrame(resumo),
            "info": {"n_videos": int(len(d)), "n_preparacoes": total_preps, "n_proporcoes": kp, "amostras": amostras,
                     "n_diagnosticos": len(colunas)}}


def _fmt(df: pd.DataFrame, cols: list[str]) -> str:
    return df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}", na_rep="-")


def montar_relatorio(res: dict, p: dict) -> str:
    e, r, i = res["eta2"], res["candidatos_resumo"], res["info"]
    L = [f"RELATÓRIO QUALIDADE E VIÉS DOS DIAGNÓSTICOS (luminescencia {__version__})",
         "Só análise: nenhum peso foi aplicado em nenhuma tabela.", "",
         f"Base: amostras {i['amostras']} (sem os saturados; a amostra C não é aberta): {i['n_videos']} vídeos, "
         f"{i['n_preparacoes']} preparações, {i['n_proporcoes']} proporções, {i['n_diagnosticos']} diagnósticos.",
         f"Permutação no nível da preparação: {p['n_permutacoes']} sorteios, semente {p['semente']}.",
         f"Limiares de decisão (config, seção qualidade): η² ajustado da proporção < {p['limite_eta2_prop']} = usável; "
         f"p < {p['alfa']} para chamar de viés; candidato a baixo peso = pior que o quantil {p['quantil_baixo']}.", "",
         "1. VEREDITO POR DIAGNÓSTICO (η² ajustado = observado - média sob permutação)", ""]
    L.append(_fmt(e, ["diagnostico", "n", "eta2_prop", "eta2_prop_ajustado", "p_prop", "eta2_amostra_extra_ajustado", "p_amostra_extra", "veredito"]))
    L += ["", "2. CONTAGEM DOS VEREDITOS", ""]
    L.append(e["veredito"].value_counts().rename_axis("veredito").reset_index(name="n_diagnosticos").to_string(index=False))
    por_canal = e.groupby(["canal", "veredito"]).size().unstack(fill_value=0)
    L += ["", "por canal:", por_canal.to_string()]
    L += ["", "3. AVISOS", ""]
    av = e[e["aviso_amostra"] != ""]
    L.append("Diagnósticos que dependem da AMOSTRA depois da proporção (um peso por eles favoreceria uma sessão):")
    L.append(", ".join(av["diagnostico"]) if len(av) else "nenhum")
    au = e[(e["p_prop_ausencia"] < p["alfa"]) & (e["eta2_prop_ausencia_ajustado"] >= p["limite_eta2_prop"])]
    L.append("Diagnósticos cuja AUSÊNCIA (canal sem ajuste) depende da proporção (o peso ficaria indefinido de modo não aleatório):")
    L.append(", ".join(au["diagnostico"]) if len(au) else "nenhum")
    L += ["", "4. CANDIDATOS A BAIXO PESO (listados em candidatos_baixo_peso.csv; nada aplicado)", ""]
    if len(r):
        L.append(_fmt(r, ["diagnostico", "veredito", "n_candidatos", "preparacoes_com_candidato", "preparacoes_maioria_candidata",
                          "preparacoes_total", "n_proporcoes_com_candidato", "proporcao_mais_afetada", "fracao_dos_candidatos_na_mais_afetada"]))
    L += ["", "Como ler: 'preparacoes_com_candidato' = preparações com pelo menos 1 vídeo candidato; 'preparacoes_maioria_candidata' = preparações em que",
          f"a fração de vídeos candidatos é >= {p['fracao_prep_maioria']}. Se os candidatos de um diagnóstico se concentram em poucas proporções",
          "('fracao_dos_candidatos_na_mais_afetada' alta), o peso por ele reduziria o peso de composições, não de vídeos ruins ao acaso.", "",
          "5. LIMITES", "",
          "- Com 2 preparações por proporção (A e B), a permutação tem poder limitado; 'inconclusivo' significa isso, não 'sem viés'.",
          "- 'Usável' diz que o diagnóstico NÃO depende da proporção; não prova que pese o erro de predição (isso exige o ganho em validação cruzada).",
          "- Os limiares são decisões provisórias do config; o orientador deve confirmá-los antes de qualquer peso.", ""]
    return "\n".join(L)


def rodar(cfg: dict, pasta_tabela: str | None = None, saida: str | None = None) -> int:
    base = Path(cfg.get("pastas", {}).get("saida", "output"))
    tabela = Path(pasta_tabela) if pasta_tabela else base / "tabela"
    saida = Path(saida) if saida else base / "qualidade"
    p = parametros(cfg)
    ds = pd.read_csv(tabela / "dataset.csv", encoding="utf-8-sig")
    desc = pd.read_csv(tabela / "descritores.csv", encoding="utf-8-sig")
    diag_desc = [c for c in colunas_de_diagnostico(desc.columns) if c not in ds.columns]
    df = ds.merge(desc[["video", *diag_desc]], on="video", how="left") if diag_desc else ds
    colunas = [c for c in colunas_de_diagnostico(df.columns) if _RE_COL.match(c)]
    if not colunas:
        raise ValueError("nenhuma coluna de diagnóstico encontrada em dataset.csv/descritores.csv")
    res = analisar(df, colunas, p)
    saida.mkdir(parents=True, exist_ok=True)
    for nome, chave in (("diagnosticos_eta2.csv", "eta2"), ("distribuicao_por_proporcao.csv", "dist_proporcao"),
                        ("distribuicao_por_amostra.csv", "dist_amostra"), ("candidatos_baixo_peso.csv", "candidatos"),
                        ("candidatos_resumo.csv", "candidatos_resumo")):
        res[chave].round(6).to_csv(saida / nome, index=False, encoding="utf-8", lineterminator="\n")
    texto = montar_relatorio(res, p)
    (saida / "relatorio_qualidade.txt").write_text(texto, encoding="utf-8", newline="\n")
    print(texto)
    print(f"\nSaída em {saida}")
    return 0
