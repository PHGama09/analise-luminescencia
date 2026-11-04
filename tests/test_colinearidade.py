"""Testes sintéticos da colinearidade (B13): respostas conhecidas, nada escrito no código. Rode: pytest"""
import numpy as np
import pandas as pd
import pytest

from luminescencia import colinearidade as col

PROIBIDAS = {"video", "amostra", "grupo", "num_video", "proporcao", "prop_R", "prop_G", "prop_B", "canal_ativo_R", "usar_no_ajuste_R",
             "sat_ate_s", "excluido_saturacao"}


def _ortogonais(n=60, k=3, semente=0):
    m = np.column_stack([np.ones(n), np.random.default_rng(semente).normal(size=(n, k))])
    q, _ = np.linalg.qr(m)                                        # colunas 1..k ortogonais entre si e à constante: média 0, desvio 1 após * sqrt(n)
    return q[:, 1:] * np.sqrt(n)


def test_colunas_identicas_dao_correlacao_um():
    x = _ortogonais(k=2)
    X = pd.DataFrame({"a": x[:, 0], "b": x[:, 0], "c": x[:, 1]})
    sp, pe = col.correlacoes(X)
    assert sp.loc["a", "b"] == pytest.approx(1.0) and pe.loc["a", "b"] == pytest.approx(1.0)
    pares = col.pares_altos(sp, pe, 0.95, "g")
    alto = pares[pares["acima_limite"]]
    assert len(alto) == 1 and {alto.iloc[0]["coluna_a"], alto.iloc[0]["coluna_b"]} == {"a", "b"}
    assert abs(pares[(pares["coluna_a"] == "a") & (pares["coluna_b"] == "c")].iloc[0]["pearson"]) < 1e-9


def test_vif_com_correlacao_conhecida():
    z = _ortogonais(k=2)
    r = 0.9
    X = pd.DataFrame({"x1": z[:, 0], "x2": r * z[:, 0] + np.sqrt(1 - r ** 2) * z[:, 1]})
    v = col.vif(X)
    assert v["x1"] == pytest.approx(1 / (1 - r ** 2), rel=1e-6) and v["x2"] == pytest.approx(1 / (1 - r ** 2), rel=1e-6)


def test_vif_independentes_e_um_e_soma_exata_e_infinito():
    z = _ortogonais(k=3)
    assert col.vif(pd.DataFrame(z, columns=list("abc"))).max() == pytest.approx(1.0, abs=1e-9)
    X = pd.DataFrame({"a": z[:, 0], "b": z[:, 1], "soma": z[:, 0] + z[:, 1]})
    assert np.isinf(col.vif(X)["soma"])


def test_dimensoes_efetivas_de_dois_fatores():
    z = _ortogonais(k=2)
    X = pd.DataFrame({"a": z[:, 0], "b": 3 * z[:, 0] + 1, "c": -z[:, 0], "d": z[:, 1], "e": 2 * z[:, 1], "f": z[:, 1] + 5})
    d = col.dimensoes_efetivas(X, 0.95)
    assert d["n_colunas"] == 6 and d["dimensoes_95"] == 2


def test_agrupamento_junta_duplicadas_e_separa_independentes():
    z = _ortogonais(k=2)
    X = pd.DataFrame({"a": z[:, 0], "a2": 5 * z[:, 0], "b": z[:, 1]})
    sp, _ = col.correlacoes(X)
    _, _, rot = col.agrupar(sp, 0.05)
    assert rot["a"] == rot["a2"] and rot["a"] != rot["b"]


def test_colunas_constantes_ou_vazias_ficam_fora():
    X = pd.DataFrame({"const": np.ones(40), "vazia": [np.nan] * 30 + [1.0] * 10, "ok": np.arange(40.0)})
    manter, fora = col.separar_constantes(X, col.parametros({}))
    assert manter == ["ok"] and set(fora) == {"const", "vazia"}


# ----------------------------------------------------------------------------------------------------------- ponta a ponta
def _tabelas(pasta):
    rng = np.random.default_rng(3)
    n, props, amostras = 40, [f"P{i}" for i in range(5)], ["A", "B"]
    base = [(p, s, k) for p in props for s in amostras for k in range(4)]
    t50 = rng.uniform(3, 8, n)
    ds = pd.DataFrame({"video": [f"v{i}" for i in range(n)], "amostra": [b[1] for b in base], "grupo": "treino", "num_video": [b[2] for b in base],
                       "proporcao": [b[0] for b in base], "prop_R": rng.random(n), "prop_G": 0.0, "prop_B": 0.0, "canal_ativo_R": True,
                       "usar_no_ajuste_R": True, "sat_ate_s": 0.0, "excluido_saturacao": False, "sinal_presente_R": True,
                       "inicio_corr_R": rng.uniform(80, 120, n), "R_cl_tau_medio": t50 * 2 + rng.normal(0, .01, n), "R_c0_tau_medio": rng.uniform(5, 9, n),
                       "R_cl_ok": True, "R_c0_ok": True})
    ds["R_c0_tau_medio"] = 1.3 * ds["R_cl_tau_medio"] + 0.5 * rng.normal(size=n)
    de = pd.DataFrame({"video": ds["video"], "amostra": ds["amostra"], "grupo": "treino", "num_video": ds["num_video"], "proporcao": ds["proporcao"],
                       "R_md_t50": t50, "R_md_r_5s": np.exp(-5 / t50), "R_md_I0": rng.uniform(20, 40, n)})
    def dic(df, resto):
        papel = {c: ("meta" if c in ("video", "amostra", "grupo", "num_video", "proporcao") else "alvo" if c.startswith("prop_") else
                     "flag" if c in resto else "feature") for c in df.columns}
        return pd.DataFrame({"coluna": list(papel), "papel": list(papel.values()), "descricao": ""})
    pasta.mkdir(parents=True)
    ds.to_csv(pasta / "dataset.csv", index=False)
    de.to_csv(pasta / "descritores.csv", index=False)
    dic(ds, {"canal_ativo_R", "usar_no_ajuste_R", "sat_ate_s", "excluido_saturacao", "sinal_presente_R", "R_cl_ok", "R_c0_ok"}).to_csv(pasta / "dicionario.csv", index=False)
    dic(de, set()).to_csv(pasta / "dicionario_descritores.csv", index=False)
    return pasta


def test_rodar_ponta_a_ponta_sem_vazamento_e_acha_a_redundancia(tmp_path):
    tab = _tabelas(tmp_path / "tabela")
    assert col.rodar({}, pasta_tabela=tab, saida=tmp_path / "s") == 0
    saida = tmp_path / "s"
    usados = set()
    for f in ("colinearidade_vif.csv", "colinearidade_clusters.csv", "colinearidade_dimensoes.csv", "conjuntos_propostos.csv", "colinearidade_casos.csv"):
        df = pd.read_csv(saida / f)
        for c in df.columns:
            if c in ("coluna", "coluna_a", "coluna_b"):
                usados |= set(df[c])
        if f == "conjuntos_propostos.csv":
            for s in df["colunas"].dropna():
                usados |= set(s.split())
    assert not usados & PROIBIDAS and {"R_md_t50", "R_md_r_5s"} <= usados
    pares = pd.read_csv(saida / "colinearidade.csv")
    p = pares[(pares["grupo"] == "todas|sem_diag") & (pares["coluna_a"].isin(["R_md_t50", "R_md_r_5s"])) & (pares["coluna_b"].isin(["R_md_t50", "R_md_r_5s"]))]
    assert len(p) == 1 and p.iloc[0]["spearman"] == pytest.approx(1.0)               # r_5s é função monotônica de t50
    casos = pd.read_csv(saida / "colinearidade_casos.csv")
    assert ((casos["caso"] == "cl x c0") & (casos["canal"] == "R")).any() and ((casos["caso"] == "tau(cl) x t50") & (casos["spearman"] > 0.99)).any()
    conj = pd.read_csv(saida / "conjuntos_propostos.csv")
    for s in conj["colunas"]:
        assert not {"R_md_t50", "R_md_r_5s"} <= set(s.split())                       # um representante por grupo redundante
    txt = (saida / "relatorio_colinearidade.txt").read_text(encoding="utf-8")
    assert "sinal_presente_R: constante" in txt and "preparações" in txt
    assert (saida / "colinearidade_mapa.png").stat().st_size > 1000


def test_risco_conta_preparacoes_e_nao_videos():
    tab = pd.DataFrame({"proporcao": ["P1"] * 10 + ["P2"] * 10, "amostra": (["A"] * 5 + ["B"] * 5) * 2})
    tab["x"] = [np.nan] * 5 + [1.0] * 15                           # as 5 primeiras linhas (P1, A) não têm a coluna
    assert col.risco_sobreajuste(tab, [], ["A", "B"])["preparacoes_treino"] == 4
    r = col.risco_sobreajuste(tab, ["x"], ["A", "B"])
    assert r["videos_treino"] == 15 and r["preparacoes_treino"] == 3 and r["proporcoes"] == 2 and r["colunas"] == 1


def test_parametros_invalidos():
    with pytest.raises(ValueError):
        col.parametros({"colinearidade": {"limite_r": 1.5}})
    with pytest.raises(Exception):
        col.parametros({"colinearidade": {"conjuntos": {"x": ["[abc"]}}})
