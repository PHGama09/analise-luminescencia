"""B5 (Q4, Fase 1): deriva de sessão entre as amostras A, B e C, por proporção.

Pergunta: A, B e C (preparações diferentes, gravadas em sessões diferentes) diferem de forma sistemática, a ponto de a IA poder
aprender a sessão e não a composição? Só análise: lê dataset.csv e descritores.csv, não altera nenhum número deles.

Para cada proporção, cada canal COM SINAL e cada grandeza (I0, t50, r_5s, auc_10s, tau médio cl e c0, mais as razões entre canais):
média por amostra (até 5 regravações), razão B/A e C/A, CV entre regravações, tamanho de efeito (eta² e d, em log) e dois testes de
n pequeno (permutação sobre eta², semente fixa, e Kruskal-Wallis). Depois: ordem A>B>C se repete além do acaso (binomial), decomposição
da variância (entre proporções / entre amostras / entre regravações) e um mapa de calor de ln(razão).

"Canal com sinal" é regra MEDIDA, nunca prop_*: o canal precisa ter a grandeza medida (ajuste/descritor existe) em pelo menos
`min_n_por_amostra` regravações de cada amostra, e a mediana de inicio_corr_<canal> da proporção precisa passar de
`limiar_inicio_corr` (níveis). Os níveis do canal sem o pó se sobrepõem aos do canal com o pó (ver entradas.py); esta regra não
separa "com o pó" de "sem o pó", só descarta o que está no ruído. Nenhum número ou conclusão está escrito aqui: tudo sai dos CSVs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CANAIS = ["R", "G", "B"]
META_DESC = ["amostra", "grupo", "num_video", "proporcao"]


def parametros(cfg: dict) -> dict:
    p = {
        "grandezas": {"I0": "{c}_md_I0", "t50": "{c}_md_t50", "r_5s": "{c}_md_r_5s", "auc_10s": "{c}_md_auc_10s",
                      "tau_cl": "{c}_cl_tau_medio", "tau_c0": "{c}_c0_tau_medio"},
        "razoes": {"razao_I0": "razao_I0_{par}", "razao_t50": "razao_t50_{par}"},
        "pares": ["R_G", "G_B", "R_B"],
        "limiar_inicio_corr": {"R": 20.0, "G": 20.0, "B": 20.0},
        "min_n_por_amostra": 3, "alfa": 0.05, "eta2_minimo": 0.30, "alfa_global": 0.01,
        "n_permutacoes": 5000, "semente": 12345,
        "fator_acima_do_acaso": 1.5, "fracao_offset_comum_minima": 0.5, "dpi": 120,
    }
    p.update(cfg.get("deriva", {}) or {})
    if p["min_n_por_amostra"] < 2:
        raise ValueError("deriva.min_n_por_amostra deve ser >= 2")
    if not 0 < p["alfa"] < 1 or not 0 < p["alfa_global"] < 1:
        raise ValueError("deriva.alfa e deriva.alfa_global devem estar entre 0 e 1")
    if p["n_permutacoes"] < 100:
        raise ValueError("deriva.n_permutacoes deve ser >= 100")
    return p


# ----------------------------------------------------------------------------------------------------------------- estatística
def _eta2_d(y: np.ndarray, g: np.ndarray, k: int) -> tuple[float, float]:
    n = len(y)
    medias = np.array([y[g == i].mean() for i in range(k)])
    ns = np.array([(g == i).sum() for i in range(k)])
    sst = float(((y - y.mean()) ** 2).sum())
    ssb = float((ns * (medias - y.mean()) ** 2).sum())
    ssw = sst - ssb
    eta2 = ssb / sst if sst > 0 else 0.0
    amp = float(medias.max() - medias.min())
    sdw = np.sqrt(max(ssw, 0.0) / (n - k)) if n > k else np.nan
    d = amp / sdw if sdw and sdw > 0 else (np.inf if amp > 0 else 0.0)
    return eta2, d


def estatisticas_celula(valores: dict, amostras: list[str], n_perm: int, rng: np.random.Generator) -> dict:
    """valores: {amostra: array positivo}. Devolve médias, razões, CV entre regravações, eta²/d (em log), p de permutação e de Kruskal-Wallis."""
    xs = [np.asarray(valores[s], float) for s in amostras]
    k, ns = len(xs), [len(x) for x in xs]
    medias = np.array([x.mean() for x in xs])
    cv_reg = float(np.sqrt(np.mean([(x.std(ddof=1) / x.mean()) ** 2 for x in xs])))
    y = np.log(np.concatenate(xs))
    g = np.repeat(np.arange(k), ns)
    eta2, d = _eta2_d(y, g, k)
    idx = rng.permuted(np.tile(np.arange(len(y)), (n_perm, 1)), axis=1)
    yp = y[idx]
    onehot = np.eye(k)[g]
    cont = onehot.sum(0)
    mp = (yp @ onehot) / cont
    sst = float(((y - y.mean()) ** 2).sum())
    ssb = (cont * (mp - y.mean()) ** 2).sum(1)
    eta2_p = ssb / sst if sst > 0 else np.zeros(n_perm)
    ssw_p = sst - ssb
    sdw_p = np.sqrt(np.maximum(ssw_p, 0) / (len(y) - k))
    with np.errstate(divide="ignore", invalid="ignore"):
        d_p = (mp.max(1) - mp.min(1)) / sdw_p
    d_nulo = float(np.nanmean(d_p[np.isfinite(d_p)])) if np.isfinite(d_p).any() else np.nan
    p_perm = float((1 + (eta2_p >= eta2 - 1e-12).sum()) / (n_perm + 1))
    try:
        h, p_kw = stats.kruskal(*xs)
        eps2 = float(h / (len(y) - 1))
    except ValueError:                                  # todos os valores idênticos
        p_kw, eps2 = 1.0, 0.0
    ordem = ">".join(amostras[i] for i in np.argsort(-medias, kind="stable"))
    out = {f"media_{s}": float(m) for s, m in zip(amostras, medias)}
    out.update({f"n_{s}": n for s, n in zip(amostras, ns)})
    out.update({
        "razao_BA": float(medias[1] / medias[0]), "razao_CA": float(medias[2] / medias[0]),
        "cv_reg": cv_reg, "diferenca_rel": float((medias.max() - medias.min()) / medias.mean()),
        "eta2": float(eta2), "d": float(d), "d_nulo": d_nulo, "eps2_kw": eps2, "p_perm": p_perm, "p_kw": float(p_kw), "ordem": ordem,
    })
    return out


def decomposicao(df: pd.DataFrame) -> dict:
    """df com colunas prop, amostra, y (log do valor). Soma de quadrados aninhada: proporção > amostra > regravação.

    Também separa o que é um deslocamento comum de cada amostra (o mesmo fator em todas as proporções, que normalizar por sessão
    removeria) do que é específico da proporção."""
    g = df.groupby("prop")["y"]
    mp = g.transform("mean")
    mpa = df.groupby(["prop", "amostra"])["y"].transform("mean")
    ss_prop = float(((mp - df["y"].mean()) ** 2).sum())
    ss_am = float(((mpa - mp) ** 2).sum())
    ss_reg = float(((df["y"] - mpa) ** 2).sum())
    tot = ss_prop + ss_am + ss_reg
    n_prop = df["prop"].nunique()
    n_cel = df.groupby(["prop", "amostra"]).ngroups
    n_am = df["amostra"].nunique()
    df_am, df_reg = n_cel - n_prop, len(df) - n_cel
    desl = (mpa - mp).groupby(df["amostra"]).transform("mean")        # deslocamento comum de cada amostra
    ss_am_depois = float(((mpa - mp - desl) ** 2).sum())
    df_am_depois = df_am - (n_am - 1)
    ms_reg = ss_reg / df_reg if df_reg > 0 else np.nan
    f = (ss_am / df_am) / ms_reg if df_am > 0 and ms_reg and ms_reg > 0 else np.nan
    f2 = (ss_am_depois / df_am_depois) / ms_reg if df_am_depois > 0 and ms_reg and ms_reg > 0 else np.nan
    return {
        "frac_proporcao": ss_prop / tot if tot > 0 else np.nan, "frac_amostra": ss_am / tot if tot > 0 else np.nan,
        "frac_regravacao": ss_reg / tot if tot > 0 else np.nan,
        "frac_offset_comum": 1 - ss_am_depois / ss_am if ss_am > 0 else np.nan,
        "F_amostra": f, "p_F_amostra": float(stats.f.sf(f, df_am, df_reg)) if np.isfinite(f) else np.nan,
        "F_amostra_sem_offset": f2,
        "p_F_sem_offset": float(stats.f.sf(f2, df_am_depois, df_reg)) if np.isfinite(f2) else np.nan,
        "n_proporcoes": int(n_prop), "n_obs": int(len(df)),
    }


# ----------------------------------------------------------------------------------------------------------------- montagem
def carregar(pasta_tabela: Path) -> pd.DataFrame:
    ler = lambda nome: pd.read_csv(Path(pasta_tabela) / nome, encoding="utf-8-sig")
    ds, de = ler("dataset.csv"), ler("descritores.csv")
    extra = [c for c in de.columns if c not in META_DESC and c not in ds.columns]
    return ds.merge(de[["video"] + extra], on="video", how="left", validate="one_to_one")


def _canal_com_sinal(sub: pd.DataFrame, canal: str, p: dict) -> bool:
    lim = (p["limiar_inicio_corr"] or {}).get(canal)
    if lim is None:
        return True
    col = f"inicio_corr_{canal}"
    return col in sub and bool(sub[col].median() >= float(lim))


def _valores_por_amostra(sub: pd.DataFrame, col: str, amostras: list[str]) -> dict:
    out = {}
    for s in amostras:
        v = pd.to_numeric(sub.loc[sub["amostra"] == s, col], errors="coerce").to_numpy(float)
        out[s] = v[np.isfinite(v) & (v > 0)]
    return out


def calcular_celulas(df: pd.DataFrame, p: dict, amostras: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Devolve (células, descartadas, dados) onde dados[(grupo, grandeza)] = DataFrame prop/amostra/y para a decomposição."""
    rng = np.random.default_rng(p["semente"])
    regs, desc, dados = [], [], {}
    itens = []
    for prop, sub in df.groupby("proporcao", sort=True):
        for c in CANAIS:
            if _canal_com_sinal(sub, c, p):
                itens += [(prop, sub, c, nome, tpl.format(c=c)) for nome, tpl in p["grandezas"].items()]
            else:
                desc.append({"proporcao": prop, "grupo": c, "grandeza": "(todas)", "motivo": "sinal abaixo do limiar de inicio_corr"})
        for par in p["pares"]:
            a, b = par.split("_")
            if _canal_com_sinal(sub, a, p) and _canal_com_sinal(sub, b, p):
                itens += [(prop, sub, par.replace("_", "/"), nome, tpl.format(par=par)) for nome, tpl in p["razoes"].items()]
    for prop, sub, grupo, nome, col in itens:
        if col not in sub:
            desc.append({"proporcao": prop, "grupo": grupo, "grandeza": nome, "motivo": f"coluna {col} ausente"})
            continue
        vals = _valores_por_amostra(sub, col, amostras)
        ruim = [f"{s}:n={len(v)}" for s, v in vals.items() if len(v) < p["min_n_por_amostra"]]
        if ruim:
            desc.append({"proporcao": prop, "grupo": grupo, "grandeza": nome, "motivo": "poucas regravações " + " ".join(ruim)})
            continue
        r = {"proporcao": prop, "grupo": grupo, "grandeza": nome,
             "inicio_corr_mediano": float(sub[[f"inicio_corr_{x}" for x in grupo.split("/") if f"inicio_corr_{x}" in sub]].median().min())}
        r.update(estatisticas_celula(vals, amostras, p["n_permutacoes"], rng))
        r["detectada"] = bool(r["p_perm"] < p["alfa"] and r["eta2"] >= p["eta2_minimo"])
        regs.append(r)
        d = dados.setdefault((grupo, nome), [])
        d += [(prop, s, float(np.log(x))) for s in amostras for x in vals[s]]
    cel = pd.DataFrame(regs)
    dados = {k: pd.DataFrame(v, columns=["prop", "amostra", "y"]) for k, v in dados.items()}
    return cel, pd.DataFrame(desc, columns=["proporcao", "grupo", "grandeza", "motivo"]), dados


def resumo_ordem(cel: pd.DataFrame, amostras: list[str], alfa_global: float) -> pd.DataFrame:
    """Por (grupo, grandeza): quantas proporções têm ordem monotônica (A>B>C ou inversa) contra o acaso (2 de 6 ordens)."""
    direta, inversa = ">".join(amostras), ">".join(amostras[::-1])
    linhas = []
    for (grupo, nome), s in cel.groupby(["grupo", "grandeza"], sort=False):
        n = len(s)
        nd, ni = int((s["ordem"] == direta).sum()), int((s["ordem"] == inversa).sum())
        lr_b, lr_c = np.log(s["razao_BA"]), np.log(s["razao_CA"])
        def wil(x):
            try:
                return float(stats.wilcoxon(x).pvalue) if len(x) >= 6 else np.nan
            except ValueError:
                return np.nan
        linhas.append({
            "grupo": grupo, "grandeza": nome, "n_proporcoes": n, "n_direta": nd, "n_inversa": ni,
            "esperado_monotonica": n / 3, "p_monotonica": float(stats.binomtest(nd + ni, n, 1 / 3, alternative="greater").pvalue),
            "p_direta": min(1.0, 2 * float(stats.binomtest(nd, n, 1 / 6, alternative="greater").pvalue)),
            "p_inversa": min(1.0, 2 * float(stats.binomtest(ni, n, 1 / 6, alternative="greater").pvalue)),
            "mediana_BA": float(np.exp(lr_b.median())), "mediana_CA": float(np.exp(lr_c.median())),
            "p_wilcoxon_BA": wil(lr_b), "p_wilcoxon_CA": wil(lr_c),
        })
    return pd.DataFrame(linhas)


def resumo_variancia(dados: dict) -> pd.DataFrame:
    linhas = []
    for (grupo, nome), d in dados.items():
        linhas.append({"grupo": grupo, "grandeza": nome, **decomposicao(d)})
    return pd.DataFrame(linhas)


def tabela_sentido(cel: pd.DataFrame) -> pd.DataFrame:
    """Linha = grupo|proporção, coluna = grandeza, valor = ordem das amostras (com * se a deriva foi detectada)."""
    c = cel.copy()
    c["linha"] = c["grupo"] + " | " + c["proporcao"]
    c["texto"] = c["ordem"] + np.where(c["detectada"], "*", "")
    return c.pivot(index="linha", columns="grandeza", values="texto")


def resumo_por_grandeza(cel: pd.DataFrame, var: pd.DataFrame, p: dict) -> pd.DataFrame:
    g = cel.assign(d=cel["d"].replace(np.inf, np.nan)).groupby("grandeza", sort=False).agg(
        n_celulas=("detectada", "size"), fracao_detectada=("detectada", "mean"), eta2_mediano=("eta2", "median"),
        cv_reg_mediano=("cv_reg", "median"), diferenca_rel_mediana=("diferenca_rel", "median"), d_mediano=("d", "median"), d_nulo=("d_nulo", "median"))
    g["p_binomial"] = [float(stats.binomtest(int(round(r.fracao_detectada * r.n_celulas)), int(r.n_celulas), p["alfa"], alternative="greater").pvalue)
                       for r in g.itertuples()]
    v = var.groupby("grandeza", sort=False)[["frac_proporcao", "frac_amostra", "frac_regravacao", "frac_offset_comum"]].median()
    return g.join(v).reset_index()


def _fmt(x, f="{:.2f}"):
    return "-" if x is None or not np.isfinite(x) else f.format(x)


def _pfmt(p):
    return "-" if p is None or not np.isfinite(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


# ----------------------------------------------------------------------------------------------------------------- conclusão
def conclusoes(cel: pd.DataFrame, ordem: pd.DataFrame, var: pd.DataFrame, p: dict, amostras: list[str]) -> list[str]:
    """Cinco linhas, todas montadas a partir dos números calculados (os veredictos vêm de regras do config)."""
    N, det = len(cel), int(cel["detectada"].sum())
    p_bin = float(stats.binomtest(det, N, p["alfa"], alternative="greater").pvalue)
    existe = p_bin < p["alfa_global"]
    d_med, dn_med = float(cel["d"].replace(np.inf, np.nan).median()), float(cel["d_nulo"].median())
    fator = d_med / dn_med if dn_med > 0 else np.nan
    maior = np.isfinite(fator) and fator >= p["fator_acima_do_acaso"]
    por_g = cel.groupby("grandeza").agg(n=("detectada", "size"), det=("detectada", "sum"), eta2=("eta2", "median"))
    por_g["p"] = [float(stats.binomtest(int(r.det), int(r.n), p["alfa"], alternative="greater").pvalue) for r in por_g.itertuples()]
    por_g["frac"] = por_g["det"] / por_g["n"]
    por_g = por_g.sort_values(["frac", "eta2"])
    seguras = [f"{g} ({_fmt(r.frac * 100, '{:.0f}')}%)" for g, r in por_g.iterrows() if r.p >= p["alfa_global"]]
    inseguras = [f"{g} ({_fmt(r.frac * 100, '{:.0f}')}%)" for g, r in por_g.iloc[::-1].iterrows() if r.p < p["alfa_global"]]
    mono = ordem.groupby("grandeza").agg(n=("n_proporcoes", "sum"), m=("n_direta", "sum"), i=("n_inversa", "sum"))
    mono["frac"] = (mono["m"] + mono["i"]) / mono["n"]
    ord_alem = [g for g, r in mono.iterrows() if stats.binomtest(int(r.m + r.i), int(r.n), 1 / 3, alternative="greater").pvalue < p["alfa_global"]]
    off = float(var["frac_offset_comum"].median())
    ramos = var["grandeza"].str.startswith("razao_")
    pg = resumo_por_grandeza(cel, var, p).set_index("grandeza")
    pares = [(g, g.replace("razao_", "")) for g in pg.index if g.startswith("razao_") and g.replace("razao_", "") in pg.index]
    txt_raz = "; ".join(f"{r} {_fmt(pg.at[r, 'fracao_detectada'] * 100, '{:.0f}')}% contra {b} {_fmt(pg.at[b, 'fracao_detectada'] * 100, '{:.0f}')}% das células "
                        f"(variância entre amostras {_fmt(pg.at[r, 'frac_amostra'] * 100, '{:.0f}')}% contra {_fmt(pg.at[b, 'frac_amostra'] * 100, '{:.0f}')}%)" for r, b in pares)
    am_med = float(var.loc[~ramos, "frac_amostra"].median())
    pr_med = float(var.loc[~ramos, "frac_proporcao"].median())
    dif_med, cv_med = float(cel["diferenca_rel"].median()), float(cel["cv_reg"].median())
    L = [
        f"1. {'Há' if existe else 'Não há (distinguível do acaso)'} deriva de sessão: {det} de {N} células (proporção x canal/razão x grandeza) "
        f"com p_perm < {p['alfa']} e eta² >= {p['eta2_minimo']} (acaso esperado: {_fmt(p['alfa'] * N, '{:.1f}')}; binomial p = {_pfmt(p_bin)}).",
        f"2. A diferença entre amostras {'é maior que' if maior else 'não passa muito de'} a esperada só pela variação entre regravações: d mediano = "
        f"{_fmt(d_med)} contra {_fmt(dn_med)} esperado sem deriva (fator {_fmt(fator)}; limite do config {p['fator_acima_do_acaso']}); "
        f"diferença relativa mediana entre amostras {_fmt(dif_med * 100, '{:.1f}')}% contra CV mediano entre regravações {_fmt(cv_med * 100, '{:.1f}')}%.",
        f"3. Detecção não acima do acaso em: {', '.join(seguras) if seguras else 'nenhuma grandeza'}. "
        f"Deriva acima do acaso (fração das células) em: {', '.join(inseguras) if inseguras else 'nenhuma'}.",
        f"4. Ordem monotônica (A>B>C ou inversa) acima do acaso (1/3) em: {', '.join(ord_alem) if ord_alem else 'nenhuma grandeza'}. "
        f"Variância (log, mediana das grandezas por canal): {_fmt(pr_med * 100, '{:.0f}')}% entre proporções contra {_fmt(am_med * 100, '{:.0f}')}% entre amostras.",
        f"5. Recomendação (calculada): validar A+B -> C (C é outra preparação/sessão e o CV entre regravações subestima o erro de generalização); "
        + (f"normalizar por sessão tende a ajudar (deslocamento comum = {_fmt(off * 100, '{:.0f}')}% da variância entre amostras, mediana)"
           if off >= p["fracao_offset_comum_minima"] else
           f"normalizar por sessão removeria pouco (deslocamento comum = {_fmt(off * 100, '{:.0f}')}% da variância entre amostras; o resto é específico da proporção)")
        + "; " + (f"razões entre canais: {txt_raz}" if pares else "sem razões entre canais nos dados") + ".",
    ]
    return L


# ----------------------------------------------------------------------------------------------------------------- figura e relatório
def figura(cel: pd.DataFrame, destino: Path, p: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cel = cel.copy()
    cel["linha"] = cel["grupo"] + " | " + cel["proporcao"]
    ordem_g = list(dict.fromkeys(cel["grandeza"]))
    ordem_l = list(dict.fromkeys(cel.sort_values(["grupo", "proporcao"])["linha"]))
    fig, axs = plt.subplots(1, 2, figsize=(11, max(4, 0.2 * len(ordem_l) + 1.5)), sharey=True)
    lim = float(np.nanpercentile(np.abs(np.log(cel[["razao_BA", "razao_CA"]].to_numpy())), 98)) or 0.1
    for ax, col, tit in zip(axs, ["razao_BA", "razao_CA"], ["ln(B/A)", "ln(C/A)"]):
        m = cel.assign(v=np.log(cel[col])).pivot(index="linha", columns="grandeza", values="v").reindex(index=ordem_l, columns=ordem_g)
        im = ax.imshow(m.to_numpy(float), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(len(ordem_g)), ordem_g, rotation=60, ha="right", fontsize=8)
        ax.set_title(tit)
        det = cel.pivot(index="linha", columns="grandeza", values="detectada").reindex(index=ordem_l, columns=ordem_g)
        for i in range(len(ordem_l)):
            for j in range(len(ordem_g)):
                if det.iat[i, j] is True or det.iat[i, j] == True:   # noqa: E712 (pivot pode devolver object)
                    ax.text(j, i, "*", ha="center", va="center", fontsize=8)
    axs[0].set_yticks(range(len(ordem_l)), ordem_l, fontsize=7)
    fig.colorbar(im, ax=axs, shrink=0.6, label="ln(razão)  (* = deriva detectada)")
    fig.savefig(destino, dpi=p["dpi"], bbox_inches="tight")
    plt.close(fig)


def relatorio(cel, desc, ordem, var, p, amostras, concl, versao: str = "") -> str:
    L = ["DERIVA DE SESSÃO A/B/C POR PROPORÇÃO (B5, só análise; nenhum número do dataset foi alterado)", "=" * 78]
    L += [f"Amostras: {', '.join(amostras)}   células: {len(cel)}   descartadas: {len(desc)}   permutações: {p['n_permutacoes']} (semente {p['semente']})",
          f"Regra de sinal (medida, sem prop_*): inicio_corr mediano >= {p['limiar_inicio_corr']} e n >= {p['min_n_por_amostra']} regravações por amostra.",
          f"Deriva detectada na célula = p_perm < {p['alfa']} e eta² >= {p['eta2_minimo']} (eta² e d em log; n = 5 por grupo: ler o tamanho de efeito, não só o p).",
          "", "CONCLUSÃO (calculada dos CSVs)", "-" * 78, *concl, ""]
    L += ["(i) SENTIDO DA DERIVA: ordem decrescente das médias (* = deriva detectada)", "-" * 78]
    t = tabela_sentido(cel)
    L.append(t.fillna("").to_string())
    L += ["", "(0) POR GRANDEZA (medianas sobre as células)", "-" * 78]
    L.append(resumo_por_grandeza(cel, var, p).round(3).to_string(index=False))
    L += ["", "(ii) ORDEM A>B>C: monotônica (direta ou inversa) contra o acaso (2 de 6 ordens = 1/3)", "-" * 78]
    o = ordem.copy()
    o["mono"] = o["n_direta"] + o["n_inversa"]
    L.append(o[["grupo", "grandeza", "n_proporcoes", "n_direta", "n_inversa", "mono", "p_monotonica", "mediana_BA", "mediana_CA", "p_wilcoxon_BA", "p_wilcoxon_CA"]]
             .round(3).to_string(index=False))
    L += ["", "(iii) DECOMPOSIÇÃO DA VARIÂNCIA (em log): fração entre proporções / entre amostras / entre regravações", "-" * 78]
    L.append(var[["grupo", "grandeza", "n_proporcoes", "frac_proporcao", "frac_amostra", "frac_regravacao", "F_amostra", "p_F_amostra",
                  "frac_offset_comum", "p_F_sem_offset"]].round(3).to_string(index=False))
    L += ["", "frac_offset_comum: parte da variância entre amostras que é o MESMO fator em todas as proporções (normalizar por sessão removeria).",
          "F_amostra: variância entre amostras / variância entre regravações (1 = sem deriva além das regravações); p_F assume normalidade em log.", ""]
    L += ["Células descartadas (motivo):", "-" * 78]
    if len(desc):
        L.append(desc.groupby(["motivo"]).size().to_string())
        L.append(desc.head(15).to_string(index=False))
    else:
        L.append("nenhuma")
    L += ["", "Limites: n = 5 por grupo; as proporções não são independentes entre si (mesma sessão/dia por amostra); testes por célula não são corrigidos para",
          "comparações múltiplas (por isso o resumo compara a fração de células detectadas com a esperada ao acaso). Esta análise não separa 'sessão' de",
          "'preparação' (cada amostra é uma preparação gravada em uma sessão): é a deriva A/B/C total, não a deriva só da câmera."]
    return "\n".join(L) + "\n"


def rodar(cfg: dict, pasta_tabela=None, saida=None) -> int:
    p = parametros(cfg)
    amostras = list((cfg.get("protocolo", {}) or {}).get("amostras", ["A", "B", "C"]))
    base = Path(cfg.get("pastas", {}).get("saida", "output"))
    pasta = Path(pasta_tabela or base / "tabela")
    saida = Path(saida or base / "deriva")
    saida.mkdir(parents=True, exist_ok=True)
    df = carregar(pasta)
    cel, desc, dados = calcular_celulas(df, p, amostras)
    if cel.empty:
        raise ValueError("nenhuma célula (proporção x canal x grandeza) com dados suficientes")
    ordem, var = resumo_ordem(cel, amostras, p["alfa_global"]), resumo_variancia(dados)
    concl = conclusoes(cel, ordem, var, p, amostras)
    cel.to_csv(saida / "deriva_celulas.csv", index=False, encoding="utf-8-sig")
    tabela_sentido(cel).to_csv(saida / "deriva_sentido.csv", encoding="utf-8-sig")
    resumo_por_grandeza(cel, var, p).to_csv(saida / "deriva_por_grandeza.csv", index=False, encoding="utf-8-sig")
    ordem.to_csv(saida / "deriva_ordem.csv", index=False, encoding="utf-8-sig")
    var.to_csv(saida / "deriva_variancia.csv", index=False, encoding="utf-8-sig")
    desc.to_csv(saida / "deriva_descartadas.csv", index=False, encoding="utf-8-sig")
    figura(cel, saida / "deriva_mapa.png", p)
    txt = relatorio(cel, desc, ordem, var, p, amostras, concl)
    (saida / "relatorio_deriva.txt").write_text(txt, encoding="utf-8", newline="\n")
    print("\n".join(concl))
    print(f"\nSalvo em {saida}")
    return 0
