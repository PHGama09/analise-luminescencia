"""B13 (Q4, Fase 1): correlação e colinearidade das colunas que a IA poderia usar como entrada. Só análise.

As colunas candidatas vêm de entradas.py (colunas_de_entrada para o dataset.csv e colunas_de_descritores para o descritores.csv),
portanto nunca incluem meta, alvo, canal_ativo_*, usar_no_ajuste_*, sat_ate_s nem excluido_saturacao. Só numpy/pandas/scipy.

Por grupo (canal R, G, B; razões entre canais; todas): correlação de Spearman e de Pearson, pares com |r| acima do limite, VIF (regressão de
cada coluna nas outras, linhas completas), agrupamento hierárquico por 1-|r| (Spearman), número efetivo de dimensões (PCA por SVD com X
padronizado, fração de variância acumulada) e os casos específicos cl x c0 e tau x t50 (global e dentro da proporção, tirando a média da
proporção, que mostra se a relação sobrevive quando a composição é fixa). A proporção só agrupa a análise; nunca é coluna.
Propõe (sem aplicar) conjuntos mínimos de entradas: um representante por grupo de |r| alto e poda por VIF, dentro de cada reservatório
de colunas definido por padrões no config. Nenhuma lista de colunas ou conclusão está escrita aqui: tudo vem dos CSVs e do config.

Opção `colinearidade.amostras` (padrão None = todas as linhas, comportamento original) e flag --so-treino (v0.16.2): restringem a análise às
amostras de treino (divisao.treino, A+B) e tiram os vídeos com excluido_saturacao (D5), gravando em output/colinearidade_AB/ e comparando
com a análise de todas as linhas (output/colinearidade/) em comparacao_com_base.txt. Só análise; nada é aplicado.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

from .entradas import colunas_de_descritores, colunas_de_entrada, parametros as parametros_entradas

CANAIS = ["R", "G", "B"]


def parametros(cfg: dict) -> dict:
    p = {
        "limite_r": 0.95, "vif_maximo": 10.0, "vif_folgado": 30.0, "variancia_pca": 0.95, "fracao_nan_maxima": 0.30, "variancia_minima_relativa": 1e-12,
        "corte_cluster": 0.05,                      # distância 1-|r| (0,05 = |r| >= 0,95 no mesmo grupo)
        "excluir_dos_conjuntos": [r"_modelo$", r"_ok$", r"^sinal_presente_", r"_r2$", r"_rmse$", r"_n_pontos$", r"_inicio_s$", r"_fim_s$"],
        "preferencia": [r"_tau_medio$", r"_md_I0$", r"_md_t50$", r"_md_r_5s$", r"_md_auc_10s$"],
        "conjuntos": {
            "so_descritores": [r"^[RGB]_md_", r"^razao_"],
            "so_tau": [r"^[RGB]_(cl|c0)_(tau1|tau2|tau_medio|fracao_rapida|A1|A2|C)$"],
            "misto": [r"^[RGB]_(cl|c0)_(tau1|tau2|tau_medio|fracao_rapida|A1|A2|C)$", r"^[RGB]_md_", r"^razao_", r"^(inicio|final)_corr_[RGB]$"],
        },
        "casos": {"par_caminho": ["cl", "c0"], "par_tau_t50": ["tau_medio", "md_t50"]}, "dpi": 120, "min_linhas": 30,
        "amostras": None,                            # None = todas as amostras (original); lista = só estas (ex.: [A, B])
    }
    p.update(cfg.get("colinearidade", {}) or {})
    if not 0 < p["limite_r"] < 1 or not 0 < p["variancia_pca"] <= 1:
        raise ValueError("colinearidade.limite_r deve estar em (0,1) e variancia_pca em (0,1]")
    for nome, pads in p["conjuntos"].items():
        for pad in pads:
            re.compile(pad)
    am = p["amostras"]
    if am is not None:
        if isinstance(am, str) or not isinstance(am, (list, tuple)) or not am:
            raise ValueError("colinearidade.amostras deve ser null ou uma lista não vazia de amostras (ex.: [A, B])")
        p["amostras"] = [str(a) for a in am]
    return p


def _verdadeiro(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "1.0")


def filtrar_linhas(ds: pd.DataFrame, amostras=None, excluir_saturados: bool = False) -> pd.DataFrame:
    """Sem filtro devolve `ds` como veio. Com `amostras`, só essas; com `excluir_saturados`, sem as linhas excluido_saturacao (D5)."""
    if not amostras and not excluir_saturados:
        return ds
    if amostras:
        ds = ds[ds["amostra"].astype(str).isin([str(a) for a in amostras])]
    if excluir_saturados:
        if "excluido_saturacao" not in ds.columns:
            raise ValueError("dataset.csv sem a coluna excluido_saturacao: não dá para tirar os saturados")
        ds = ds[~ds["excluido_saturacao"].map(_verdadeiro)]
    return ds.reset_index(drop=True)


# ----------------------------------------------------------------------------------------------------------------- colunas
def candidatas(ds: pd.DataFrame, de: pd.DataFrame, dic: pd.DataFrame, dic_desc: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Colunas permitidas por entradas.py (os dois arquivos), numéricas. Devolve (tabela por vídeo, {coluna: origem})."""
    pe = parametros_entradas({**cfg, "entradas": {**(cfg.get("entradas", {}) or {}), "canais_sem_po": True, "caminhos": ["cl", "c0"], "limiar_inicio_corr": None}})
    c_ds = [c for c in colunas_de_entrada(dic, pe) if c in ds.columns]
    c_de = [c for c in colunas_de_descritores(dic_desc, cfg) if c in de.columns and c not in c_ds]
    tab = ds[["video", "proporcao", "amostra"] + c_ds].merge(de[["video"] + c_de], on="video", how="left", validate="one_to_one")
    origem = {**{c: "dataset" for c in c_ds}, **{c: "descritores" for c in c_de}}
    num = {}
    for c in origem:
        v = pd.to_numeric(tab[c], errors="coerce") if tab[c].dtype != bool else tab[c].astype(float)
        if v.notna().any():
            num[c] = v.astype(float)
    return pd.DataFrame(num, index=tab.index).join(tab[["video", "proporcao", "amostra"]]), {c: origem[c] for c in num}


def e_diagnostico(c: str, p: dict) -> bool:
    """Coluna que descreve o ajuste/a janela (qualidade, nº de pontos, fim da janela, flags), não o decaimento em si."""
    return any(re.search(x, c) for x in p["excluir_dos_conjuntos"])


def grupo_da_coluna(c: str) -> str:
    if c.startswith("razao_"):
        return "razoes"
    m = re.match(r"^([RGB])_", c) or re.search(r"_([RGB])$", c)
    return m.group(1) if m else "outros"


def separar_constantes(X: pd.DataFrame, p: dict) -> tuple[list[str], dict]:
    manter, fora = [], {}
    for c in X.columns:
        v = X[c].dropna()
        if len(v) < p["min_linhas"]:
            fora[c] = f"poucos valores ({len(v)})"
        elif X[c].isna().mean() > p["fracao_nan_maxima"]:
            fora[c] = f"{X[c].isna().mean() * 100:.0f}% vazio (> {p['fracao_nan_maxima'] * 100:.0f}%)"
        elif v.nunique() < 2 or v.std() <= p["variancia_minima_relativa"] * max(1.0, abs(v.mean())):
            fora[c] = "constante"
        else:
            manter.append(c)
    return manter, fora


# ----------------------------------------------------------------------------------------------------------------- números
def correlacoes(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return X.corr(method="spearman", min_periods=10), X.corr(method="pearson", min_periods=10)


def pares_altos(sp: pd.DataFrame, pe: pd.DataFrame, limite: float, grupo: str) -> pd.DataFrame:
    cols, linhas = list(sp.columns), []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rs, rp = sp.iat[i, j], pe.iat[i, j]
            m = np.nanmax([abs(rs), abs(rp)]) if not (np.isnan(rs) and np.isnan(rp)) else np.nan
            linhas.append({"grupo": grupo, "coluna_a": cols[i], "coluna_b": cols[j], "spearman": rs, "pearson": rp, "abs_max": m})
    d = pd.DataFrame(linhas, columns=["grupo", "coluna_a", "coluna_b", "spearman", "pearson", "abs_max"])
    d["acima_limite"] = d["abs_max"] > limite
    return d.sort_values("abs_max", ascending=False, ignore_index=True)


def _padronizar(X: pd.DataFrame) -> pd.DataFrame:
    """Linhas completas, padronizadas; colunas sem variância nessas linhas saem (não há o que explicar nelas)."""
    Z = X.dropna()
    Z = Z.loc[:, Z.std(ddof=0) > 0]
    return (Z - Z.mean()) / Z.std(ddof=0)


def vif(X: pd.DataFrame) -> pd.Series:
    """VIF de cada coluna: 1/(1-R²) da regressão (mínimos quadrados) nas demais, padronizando e usando só linhas completas. inf se R² ~ 1."""
    Z = _padronizar(X)
    out = {}
    for c in Z.columns:
        y, A = Z[c].to_numpy(), Z.drop(columns=c).to_numpy()
        if A.shape[1] == 0:
            out[c] = 1.0
            continue
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ beta
        r2 = 1 - (res @ res) / (y @ y)
        out[c] = float(1 / (1 - r2)) if r2 < 1 - 1e-12 else np.inf
    return pd.Series(out, name="vif")


def dimensoes_efetivas(X: pd.DataFrame, fracao: float) -> dict:
    Z = _padronizar(X)
    s = np.linalg.svd(Z.to_numpy(), compute_uv=False)
    var = s ** 2 / (s ** 2).sum()
    acum = np.cumsum(var)
    k = int(np.searchsorted(acum, fracao - 1e-12) + 1)
    return {"n_colunas": Z.shape[1], "n_linhas": int(len(Z)), "dimensoes_95": k, "pc1": float(var[0]),
            "razao_participacao": float(1 / (var ** 2).sum())}


def agrupar(sp: pd.DataFrame, corte: float):
    """Agrupamento hierárquico (ligação média) por 1-|r|. Devolve (linkage, ordem das folhas, rótulo do grupo por coluna)."""
    d = 1 - sp.abs().fillna(0).to_numpy()
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0)
    Z = linkage(squareform(np.clip(d, 0, None), checks=False), method="average")
    rot = fcluster(Z, t=corte, criterion="distance")
    return Z, leaves_list(Z), pd.Series(rot, index=sp.columns, name="cluster")


def casos_especificos(tab: pd.DataFrame, cols: set, p: dict) -> pd.DataFrame:
    """cl x c0 (cada medida do ajuste, por canal) e tau x t50 (por canal): global e dentro da proporção (média da proporção removida)."""
    linhas = []

    def par(a, b, rotulo, canal):
        if a not in tab or b not in tab:
            return
        ok = tab[[a, b, "proporcao"]].dropna()
        if len(ok) < 10:
            return
        dentro = ok[[a, b]] - ok.groupby("proporcao")[[a, b]].transform("mean")
        linhas.append({"caso": rotulo, "canal": canal, "coluna_a": a, "coluna_b": b, "n": len(ok),
                       "spearman": float(stats.spearmanr(ok[a], ok[b])[0]), "pearson": float(stats.pearsonr(ok[a], ok[b])[0]),
                       "spearman_dentro_proporcao": float(stats.spearmanr(dentro[a], dentro[b])[0]),
                       "pearson_dentro_proporcao": float(stats.pearsonr(dentro[a], dentro[b])[0])})

    c1, c2 = p["casos"]["par_caminho"]
    medidas = sorted({re.sub(rf"^[RGB]_{c1}_", "", c) for c in cols if re.match(rf"^[RGB]_{c1}_", c)})
    for canal in CANAIS:
        for m in medidas:
            if e_diagnostico(f"{canal}_{c1}_{m}", p):
                continue
            par(f"{canal}_{c1}_{m}", f"{canal}_{c2}_{m}", f"{c1} x {c2}", canal)
        for caminho in (c1, c2):
            t, d50 = p["casos"]["par_tau_t50"]
            par(f"{canal}_{caminho}_{t}", f"{canal}_{d50}", f"tau({caminho}) x t50", canal)
    return pd.DataFrame(linhas)


def conjunto_minimo(X: pd.DataFrame, origem: dict, padroes: list[str], p: dict, vif_max: float | None = None) -> dict:
    """Reservatório = colunas que casam com os padrões (menos as de diagnóstico do ajuste). Um representante por grupo de |r| >= limite
    (o que casa com `preferencia`; depois o mais completo) e depois poda de VIF > vif_maximo (tira o de maior VIF, que não seja preferido, até acabar)."""
    vif_max = p["vif_maximo"] if vif_max is None else vif_max
    excl = [re.compile(x) for x in p["excluir_dos_conjuntos"]]
    pref = [re.compile(x) for x in p["preferencia"]]
    pads = [re.compile(x) for x in padroes]
    pool = [c for c in X.columns if any(r.search(c) for r in pads) and not any(r.search(c) for r in excl)]
    base = {"reservatorio": len(pool)}
    if len(pool) < 2:
        return {**base, "colunas": pool, "vif_max": np.nan, "dimensoes_95": len(pool), "n_linhas": 0}
    Z = X[pool]
    comp = Z.dropna()
    if len(comp) < p["min_linhas"]:                                  # tira as colunas mais vazias até sobrar linhas completas
        while len(Z.dropna()) < p["min_linhas"] and Z.shape[1] > 2:
            Z = Z.drop(columns=Z.isna().sum().idxmax())
    sp = Z.corr(method="spearman", min_periods=10)
    _, _, rot = agrupar(sp, p["corte_cluster"])
    escolhidas = []
    for _, cols in rot.groupby(rot):
        cs = list(cols.index)
        cs.sort(key=lambda c: (not any(r.search(c) for r in pref), Z[c].isna().sum(), c))
        escolhidas.append(cs[0])
    while True:
        v = vif(Z[escolhidas])
        if len(escolhidas) <= 2 or v.max() <= vif_max:
            break
        ordem = [c for c in v.sort_values(ascending=False).index]
        sem_pref = [c for c in ordem if not any(r.search(c) for r in pref)]
        escolhidas.remove((sem_pref or ordem)[0])
    d = dimensoes_efetivas(Z[escolhidas], p["variancia_pca"])
    return {**base, "colunas": sorted(escolhidas), "vif_max": float(vif(Z[escolhidas]).max()), "dimensoes_95": d["dimensoes_95"], "n_linhas": d["n_linhas"]}


def risco_sobreajuste(tab: pd.DataFrame, colunas: list[str], treino: list[str]) -> dict:
    """Quantas unidades realmente independentes existem: (proporção, amostra) = uma preparação; 5 regravações dela são quase duplicatas.
    Só contam as linhas com todas as `colunas` preenchidas (as que um treino com essas entradas usaria)."""
    tr = tab[tab["amostra"].isin(treino)]
    if colunas:
        tr = tr.dropna(subset=colunas)
    return {"videos_treino": int(len(tr)), "preparacoes_treino": int(tr.groupby(["proporcao", "amostra"]).ngroups),
            "proporcoes": int(tr["proporcao"].nunique()), "colunas": len(colunas)}


# ----------------------------------------------------------------------------------------------------------------- saída
def figura(sp: pd.DataFrame, Z, ordem, rot: pd.Series, destino: Path, corte: float, dpi: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    n = len(sp)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, max(6, 0.17 * n + 2)), gridspec_kw={"width_ratios": [1, 1.4]})
    dendrogram(Z, labels=list(sp.columns), orientation="left", color_threshold=corte, leaf_font_size=6, ax=a1)
    a1.axvline(corte, color="gray", ls="--", lw=0.8)
    a1.set_title(f"1-|Spearman| (corte {corte})")
    m = sp.abs().to_numpy()[np.ix_(ordem, ordem)]
    a2.imshow(m, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    a2.set_xticks([]), a2.set_yticks(range(n), [sp.columns[i] for i in ordem], fontsize=6)
    a2.yaxis.tick_right()
    a2.set_title("|Spearman| (ordem das folhas)")
    fig.savefig(destino, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------- comparação
def _chave_par(a: str, b: str) -> str:
    return " | ".join(sorted([str(a), str(b)]))


def _conj(s) -> list[str]:
    return str(s).split() if isinstance(s, str) else []


def comparar_com_base(pasta_ab, pasta_base, grupo_pares: str = "todas|sem_diag") -> tuple[list[str], dict]:
    """Compara duas saídas de `rodar` (a restrita e a de todas as linhas) lendo só os CSVs gravados: dimensões efetivas, número de pares
    acima do limite, pares e colunas que entraram/saíram, e os conjuntos mínimos (mantidas / só na base / só na restrita)."""
    nomes = ["colinearidade_dimensoes.csv", "colinearidade.csv", "colinearidade_clusters.csv", "conjuntos_propostos.csv"]
    lidos = {}
    for n in nomes:
        for rot, pasta in (("ab", pasta_ab), ("base", pasta_base)):
            f = Path(pasta) / n
            if not f.exists():
                return [f"Comparação não feita: falta {f}. Rode antes 'luminescencia colinearidade' (todas as linhas) sobre a mesma tabela."], {}
            lidos[(rot, n)] = pd.read_csv(f, encoding="utf-8-sig")
    dim = lidos[("base", nomes[0])].merge(lidos[("ab", nomes[0])], on="grupo", how="outer", suffixes=("_base", "_ab"))
    dim = dim[["grupo", "n_colunas_base", "n_colunas_ab", "n_linhas_base", "n_linhas_ab", "dimensoes_95_base", "dimensoes_95_ab"]]
    pb, pa = lidos[("base", nomes[1])], lidos[("ab", nomes[1])]
    cont = pd.concat([pb.groupby("grupo", sort=False).size().rename("pares_base"), pa.groupby("grupo", sort=False).size().rename("pares_ab")], axis=1)
    cont = cont.fillna(0).astype(int).reset_index().rename(columns={"index": "grupo"})
    chaves = lambda df: {_chave_par(a, b) for a, b in zip(df.loc[df["grupo"] == grupo_pares, "coluna_a"], df.loc[df["grupo"] == grupo_pares, "coluna_b"])}
    kb, ka = chaves(pb), chaves(pa)
    cb = lidos[("base", nomes[2])]
    ca = lidos[("ab", nomes[2])]
    ub, ua = set(cb.loc[cb["grupo"] == "todas", "coluna"]), set(ca.loc[ca["grupo"] == "todas", "coluna"])
    cj_b, cj_a = lidos[("base", nomes[3])], lidos[("ab", nomes[3])]
    cj_b = {(r["conjunto"], float(r["vif_maximo_regra"])): _conj(r["colunas"]) for _, r in cj_b.iterrows()}
    cj_a = {(r["conjunto"], float(r["vif_maximo_regra"])): _conj(r["colunas"]) for _, r in cj_a.iterrows()}
    conjuntos = {}
    for k in cj_b:
        if k in cj_a:
            sb, sa = set(cj_b[k]), set(cj_a[k])
            conjuntos[f"{k[0]}|{k[1]:g}"] = {"mantidas": sorted(sb & sa), "so_base": sorted(sb - sa), "so_ab": sorted(sa - sb)}
    res = {"colunas_entraram": sorted(ua - ub), "colunas_sairam": sorted(ub - ua), "pares_entraram": sorted(ka - kb), "pares_sairam": sorted(kb - ka), "conjuntos": conjuntos}
    L = ["COMPARAÇÃO COM A ANÁLISE DE TODAS AS LINHAS (base = %s; restrita = %s)" % (pasta_base, pasta_ab), "-" * 78,
         "Dimensões efetivas e nº de colunas/linhas por grupo (colunas *_base = todas as linhas; *_ab = restrita):", dim.round(3).to_string(index=False), "",
         f"Pares com |r| > limite por grupo (base x restrita):", cont.to_string(index=False), "",
         f"Colunas analisadas ('todas'): {len(ub)} na base, {len(ua)} na restrita; entraram {len(res['colunas_entraram'])}: {', '.join(res['colunas_entraram']) or '-'}; saíram {len(res['colunas_sairam'])}: {', '.join(res['colunas_sairam']) or '-'}",
         f"Pares |r| alto ('{grupo_pares}'): {len(kb)} na base, {len(ka)} na restrita; só na restrita {len(res['pares_entraram'])}, só na base {len(res['pares_sairam'])}",
         "  só na restrita (até 25): " + ("; ".join(res["pares_entraram"][:25]) or "-"), "  só na base (até 25): " + ("; ".join(res["pares_sairam"][:25]) or "-"), "",
         "CONJUNTOS MÍNIMOS (propostos, não aplicados): mantidas / só na base / só na restrita"]
    for k, v in conjuntos.items():
        L += [f"[{k}] mantidas {len(v['mantidas'])}: {', '.join(v['mantidas']) or '-'}", f"   só na base {len(v['so_base'])}: {', '.join(v['so_base']) or '-'}",
              f"   só na restrita {len(v['so_ab'])}: {', '.join(v['so_ab']) or '-'}"]
    return L, res


def _tabela_txt(df: pd.DataFrame, n: int = 15) -> str:
    return df.head(n).round(3).to_string(index=False)


def rodar(cfg: dict, pasta_tabela=None, saida=None, so_treino: bool = False, pasta_base=None) -> int:
    p = parametros(cfg)
    amostras, sem_sat = p["amostras"], False
    if so_treino:                                                    # protocolo: treino = divisao.treino (A+B) e saturados fora (D5)
        amostras, sem_sat = [str(a) for a in (cfg.get("divisao", {}) or {}).get("treino", ["A", "B"])], True
    filtro = bool(amostras) or sem_sat
    base = Path(cfg.get("pastas", {}).get("saida", "output"))
    pasta = Path(pasta_tabela or base / "tabela")
    saida = Path(saida or base / ("colinearidade_" + ("AB" if so_treino else "".join(amostras)) if filtro else "colinearidade"))
    saida.mkdir(parents=True, exist_ok=True)
    ler = lambda nome: pd.read_csv(pasta / nome, encoding="utf-8-sig")
    ds, de, dic, dic_desc = ler("dataset.csv"), ler("descritores.csv"), ler("dicionario.csv"), ler("dicionario_descritores.csv")
    n_total = len(ds)
    ds = filtrar_linhas(ds, amostras, sem_sat)
    if filtro and len(ds) < p["min_linhas"]:
        raise ValueError(f"só {len(ds)} vídeos após o filtro (amostras {amostras}, sem saturados: {sem_sat}); mínimo colinearidade.min_linhas = {p['min_linhas']}")
    tab, origem = candidatas(ds, de, dic, dic_desc, cfg)
    cols = list(origem)
    manter, fora = separar_constantes(tab[cols], p)
    X = tab[manter]
    base_g = {g: [c for c in manter if grupo_da_coluna(c) == g] for g in [*CANAIS, "razoes", "outros"]}
    grupos = {}
    for g, c in {**base_g, "todas": manter}.items():
        grupos[g] = c
        sem = [x for x in c if not e_diagnostico(x, p)]
        grupos[f"{g}|sem_diag"] = sem
    grupos = {g: c for g, c in grupos.items() if len(c) >= 2}

    pares, vifs, dims, clusters = [], [], [], []
    graf = None
    for g, cs in grupos.items():
        sp, pe = correlacoes(X[cs])
        pares.append(pares_altos(sp, pe, p["limite_r"], g))
        v = vif(X[cs])
        vifs.append(pd.DataFrame({"grupo": g, "coluna": v.index, "vif": v.values}))
        d = dimensoes_efetivas(X[cs], p["variancia_pca"])
        dims.append({"grupo": g, **d})
        Z, ordem, rot = agrupar(sp, p["corte_cluster"])
        clusters.append(pd.DataFrame({"grupo": g, "coluna": rot.index, "cluster": rot.values}))
        if g == "todas|sem_diag":
            sp.to_csv(saida / "corr_spearman.csv", encoding="utf-8-sig")
            pe.to_csv(saida / "corr_pearson.csv", encoding="utf-8-sig")
            graf = (sp, Z, ordem, rot)
    pares = pd.concat(pares, ignore_index=True)
    vifs, dims, clusters = pd.concat(vifs, ignore_index=True), pd.DataFrame(dims), pd.concat(clusters, ignore_index=True)
    casos = casos_especificos(tab, set(manter), p)
    conj = {nome: conjunto_minimo(X, origem, pads, p) for nome, pads in p["conjuntos"].items()}
    conj_folgado = {nome: conjunto_minimo(X, origem, pads, p, p["vif_folgado"]) for nome, pads in p["conjuntos"].items()}
    treino = list((cfg.get("divisao", {}) or {}).get("treino", ["A", "B"]))
    risco = {nome: risco_sobreajuste(tab, c["colunas"], treino) for nome, c in conj.items()}

    pares[pares["acima_limite"]].to_csv(saida / "colinearidade.csv", index=False, encoding="utf-8-sig")
    vifs.to_csv(saida / "colinearidade_vif.csv", index=False, encoding="utf-8-sig")
    dims.to_csv(saida / "colinearidade_dimensoes.csv", index=False, encoding="utf-8-sig")
    clusters.to_csv(saida / "colinearidade_clusters.csv", index=False, encoding="utf-8-sig")
    casos.to_csv(saida / "colinearidade_casos.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"conjunto": n, "vif_maximo_regra": vm, **{k: v for k, v in c.items() if k != "colunas"}, "colunas": " ".join(c["colunas"])}
                  for vm, cj in ((p["vif_maximo"], conj), (p["vif_folgado"], conj_folgado)) for n, c in cj.items()]
                 ).to_csv(saida / "conjuntos_propostos.csv", index=False, encoding="utf-8-sig")
    figura(*graf, saida / "colinearidade_mapa.png", p["corte_cluster"], p["dpi"])

    todas = pares[pares["grupo"] == "todas|sem_diag"]
    n_prep = int(ds.groupby(["proporcao", "amostra"]).ngroups) if len(ds) else 0
    n_diag = sum(e_diagnostico(c, p) for c in manter)
    L = ["COLINEARIDADE DAS ENTRADAS (B13, só análise; nada foi aplicado)", "=" * 78,
         f"Candidatas (permitidas por entradas.py): {len(cols)} ({sum(o == 'dataset' for o in origem.values())} do dataset, {sum(o == 'descritores' for o in origem.values())} dos descritores); "
         f"usadas: {len(manter)}; fora: {len(fora)}; entre as usadas, {n_diag} são diagnóstico do ajuste/janela (grupos '|sem_diag' = sem elas).",
         f"Limite |r| > {p['limite_r']}; VIF máximo {p['vif_maximo']}; PCA {p['variancia_pca'] * 100:.0f}% da variância; corte do agrupamento 1-|r| = {p['corte_cluster']}.",
         "Fora da análise (motivo): " + ("; ".join(f"{c}: {m}" for c, m in list(fora.items())[:15]) + (" ..." if len(fora) > 15 else "") if fora else "nada"), "",
         "NÚMERO EFETIVO DE DIMENSÕES (PCA por SVD, linhas completas)", "-" * 78, dims.round(3).to_string(index=False), "",
         f"PARES COM |r| > {p['limite_r']} (Spearman ou Pearson), por grupo; lista = 'todas|sem_diag': " + ", ".join(f"{g} {int(s['acima_limite'].sum())}" for g, s in pares.groupby("grupo", sort=False)),
         "-" * 78, _tabela_txt(todas[todas["acima_limite"]][["coluna_a", "coluna_b", "spearman", "pearson"]], 15),
         "(lista completa em colinearidade.csv)", "",
         "VIF MAIS ALTOS (grupo 'todas|sem_diag')", "-" * 78, _tabela_txt(vifs[vifs["grupo"] == "todas|sem_diag"].sort_values("vif", ascending=False), 12),
         f"colunas com VIF > {p['vif_maximo']}: " + ", ".join(f"{g} {int((s['vif'] > p['vif_maximo']).sum())}/{len(s)}" for g, s in vifs.groupby("grupo", sort=False)), "",
         "AGRUPAMENTO: grupos de colunas redundantes (|r| >= %.2f), grupo 'todas|sem_diag'" % (1 - p["corte_cluster"]), "-" * 78]
    cl = clusters[clusters["grupo"] == "todas|sem_diag"]
    L.append(f"{cl['cluster'].nunique()} grupos para {len(cl)} colunas; grupos com mais de uma coluna: {int((cl.groupby('cluster').size() > 1).sum())}")
    for _, s in cl.groupby("cluster"):
        if len(s) > 1:
            L.append("  " + ", ".join(s["coluna"]))
    L += ["", "CASOS ESPECÍFICOS: cl x c0 e tau x t50 (global e dentro da proporção)", "-" * 78]
    if len(casos):
        L.append(casos.groupby("caso")[["spearman", "pearson", "spearman_dentro_proporcao", "pearson_dentro_proporcao"]].agg(["min", "median"]).round(3).to_string())
        sel = casos[casos["coluna_a"].str.endswith("tau_medio")]
        L.append(_tabela_txt(sel[["caso", "canal", "n", "spearman", "spearman_dentro_proporcao"]], 15))
    L += ["", f"CONJUNTOS MÍNIMOS PROPOSTOS (NÃO aplicados; representante por grupo de |r| alto + poda de VIF > {p['vif_maximo']}; folgado = VIF > {p['vif_folgado']})", "-" * 78]
    for n, c in conj.items():
        r = risco[n]
        cf = conj_folgado[n]
        L += [f"[{n}] {len(c['colunas'])} colunas (reservatório {c['reservatorio']}), VIF máx {c['vif_max']:.2f}, dimensões efetivas {c['dimensoes_95']}, linhas completas {c['n_linhas']}",
              "   " + ", ".join(c["colunas"]),
              f"   folgado (VIF <= {p['vif_folgado']:g}): {len(cf['colunas'])} colunas: " + ", ".join(cf["colunas"]),
              f"   treino {', '.join(treino)}: {r['videos_treino']} vídeos = {r['preparacoes_treino']} preparações (proporção x amostra) -> {len(c['colunas']) / max(r['preparacoes_treino'], 1):.2f} colunas por preparação"]
    L += ["", "RISCO DE SOBREAJUSTE: as regravações de uma mesma amostra são quase duplicatas (CV baixo); o número de unidades independentes é o de preparações,",
          "não o de vídeos. Colunas redundantes (|r| alto) não acrescentam informação, só graus de liberdade. Validar sempre agrupando por amostra (ver C17)."]
    if filtro:
        L.insert(2, f"Linhas usadas: {len(ds)} vídeos de {n_total} do dataset.csv (amostras {', '.join(amostras) if amostras else 'todas'}; "
                    f"{'saturados fora' if sem_sat else 'saturados mantidos'}) = {n_prep} preparações (proporção x amostra).")
        base_cmp = Path(pasta_base) if pasta_base else base / "colinearidade"
        if base_cmp.resolve() != saida.resolve():
            Lc, _ = comparar_com_base(saida, base_cmp)
            (saida / "comparacao_com_base.txt").write_text("\n".join(Lc) + "\n", encoding="utf-8", newline="\n")
            L += ["", *Lc]
    (saida / "relatorio_colinearidade.txt").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L[:12]))
    print(f"\nSalvo em {saida}")
    return 0
