"""Testes da opção colinearidade.amostras e da flag --so-treino (B13 só com A+B). Dados sintéticos com resposta conhecida. Rode: pytest"""
import numpy as np
import pandas as pd
import pytest

from luminescencia import colinearidade as col

META = ("video", "amostra", "grupo", "num_video", "proporcao")


def _tabelas(pasta):
    """60 vídeos = 5 proporções x A,B,C x 4 regravações. R_md_auc_10s = 2 x R_md_I0 em A e B, independente em C. A primeira linha (P0, A) é saturada."""
    rng = np.random.default_rng(7)
    base = [(p, s, k) for p in [f"P{i}" for i in range(5)] for s in ["A", "B", "C"] for k in range(4)]
    n = len(base)
    amo = np.array([b[1] for b in base])
    i0 = rng.uniform(20, 40, n)
    t50 = rng.uniform(3, 8, n)
    auc = np.where(amo == "C", rng.uniform(20, 40, n), 2 * i0)
    ds = pd.DataFrame({"video": [f"v{i}" for i in range(n)], "amostra": amo, "grupo": "x", "num_video": [b[2] for b in base], "proporcao": [b[0] for b in base],
                       "prop_R": rng.random(n), "prop_G": 0.0, "prop_B": 0.0, "canal_ativo_R": True, "usar_no_ajuste_R": True, "sat_ate_s": 0.0,
                       "excluido_saturacao": False, "sinal_presente_R": True, "inicio_corr_R": rng.uniform(80, 120, n),
                       "R_cl_tau_medio": rng.uniform(5, 9, n), "R_c0_tau_medio": rng.uniform(5, 9, n), "R_cl_ok": True, "R_c0_ok": True})
    ds.loc[0, "excluido_saturacao"] = True
    de = pd.DataFrame({"video": ds["video"], "amostra": amo, "grupo": "x", "num_video": ds["num_video"], "proporcao": ds["proporcao"],
                       "R_md_t50": t50, "R_md_I0": i0, "R_md_auc_10s": auc})

    def dic(df, flags):
        papel = {c: ("meta" if c in META else "alvo" if c.startswith("prop_") else "flag" if c in flags else "feature") for c in df.columns}
        return pd.DataFrame({"coluna": list(papel), "papel": list(papel.values()), "descricao": ""})
    pasta.mkdir(parents=True)
    ds.to_csv(pasta / "dataset.csv", index=False)
    de.to_csv(pasta / "descritores.csv", index=False)
    dic(ds, {"canal_ativo_R", "usar_no_ajuste_R", "sat_ate_s", "excluido_saturacao", "sinal_presente_R", "R_cl_ok", "R_c0_ok"}).to_csv(pasta / "dicionario.csv", index=False)
    dic(de, set()).to_csv(pasta / "dicionario_descritores.csv", index=False)
    return pasta


def test_filtrar_linhas_contagens_conhecidas(tmp_path):
    ds = pd.read_csv(_tabelas(tmp_path / "t") / "dataset.csv")
    assert col.filtrar_linhas(ds) is ds and col.filtrar_linhas(ds, None, False) is ds           # sem filtro: nada muda
    ab = col.filtrar_linhas(ds, ["A", "B"])
    assert len(ab) == 40 and set(ab["amostra"]) == {"A", "B"}
    assert len(col.filtrar_linhas(ds, ["A", "B"], True)) == 39                                     # menos a saturada
    assert len(col.filtrar_linhas(ds, None, True)) == 59 and set(col.filtrar_linhas(ds, ["C"])["amostra"]) == {"C"}
    with pytest.raises(ValueError):
        col.filtrar_linhas(ds.drop(columns="excluido_saturacao"), ["A"], True)


def test_parametros_amostras():
    assert col.parametros({})["amostras"] is None and col.parametros({"colinearidade": {"amostras": None}})["amostras"] is None
    assert col.parametros({"colinearidade": {"amostras": ["A", "B"]}})["amostras"] == ["A", "B"]
    for ruim in ("AB", [], 3):
        with pytest.raises(ValueError):
            col.parametros({"colinearidade": {"amostras": ruim}})


def test_so_treino_acha_par_que_so_existe_em_AB_e_a_base_nao_muda(tmp_path):
    tab = _tabelas(tmp_path / "tabela")
    assert col.rodar({}, pasta_tabela=tab, saida=tmp_path / "cheio") == 0
    assert col.rodar({}, pasta_tabela=tab, saida=tmp_path / "ab", so_treino=True, pasta_base=tmp_path / "cheio") == 0
    chave = col._chave_par("R_md_I0", "R_md_auc_10s")
    g = "todas|sem_diag"

    def pares(pasta):
        d = pd.read_csv(pasta / "colinearidade.csv")
        d = d[d["grupo"] == g]
        return {col._chave_par(a, b) for a, b in zip(d["coluna_a"], d["coluna_b"])}
    assert chave in pares(tmp_path / "ab") and chave not in pares(tmp_path / "cheio")             # |r| = 1 só em A+B
    txt_cheio = (tmp_path / "cheio" / "relatorio_colinearidade.txt").read_text(encoding="utf-8")
    txt_ab = (tmp_path / "ab" / "relatorio_colinearidade.txt").read_text(encoding="utf-8")
    assert "Linhas usadas" not in txt_cheio and "COMPARAÇÃO" not in txt_cheio and not (tmp_path / "cheio" / "comparacao_com_base.txt").exists()
    assert "39 vídeos de 60" in txt_ab and "saturados fora" in txt_ab and "COMPARAÇÃO COM A ANÁLISE" in txt_ab
    assert (tmp_path / "ab" / "comparacao_com_base.txt").exists()
    assert (pd.read_csv(tmp_path / "ab" / "colinearidade_dimensoes.csv")["n_linhas"] <= 39).all()
    assert (pd.read_csv(tmp_path / "cheio" / "colinearidade_dimensoes.csv")["n_linhas"] > 39).any()
    _, d = col.comparar_com_base(tmp_path / "ab", tmp_path / "cheio")
    assert chave in d["pares_entraram"] and chave not in d["pares_sairam"]
    usados = set()
    for f in ("colinearidade_clusters.csv", "colinearidade_vif.csv"):
        usados |= set(pd.read_csv(tmp_path / "ab" / f)["coluna"])
    assert not usados & {"video", "amostra", "proporcao", "prop_R", "excluido_saturacao", "sat_ate_s", "canal_ativo_R", "usar_no_ajuste_R"}


def test_amostras_na_config_nao_sobrescreve_a_pasta_padrao(tmp_path):
    tab = _tabelas(tmp_path / "tabela")
    cfg = {"pastas": {"saida": str(tmp_path / "out")}, "colinearidade": {"amostras": ["A", "B"]}}
    assert col.rodar(cfg, pasta_tabela=tab) == 0
    assert (tmp_path / "out" / "colinearidade_AB" / "relatorio_colinearidade.txt").exists() and not (tmp_path / "out" / "colinearidade").exists()
    txt = (tmp_path / "out" / "colinearidade_AB" / "relatorio_colinearidade.txt").read_text(encoding="utf-8")
    assert "40 vídeos de 60" in txt and "saturados mantidos" in txt                                # sem --so-treino a saturada continua


def test_pouco_dado_apos_filtro_erra(tmp_path):
    tab = _tabelas(tmp_path / "tabela")
    with pytest.raises(ValueError):
        col.rodar({"colinearidade": {"amostras": ["C"], "min_linhas": 30}}, pasta_tabela=tab, saida=tmp_path / "s")   # só 20 linhas de C


def _csvs(pasta, dims, pares, clusters, conj):
    pasta.mkdir(parents=True)
    pd.DataFrame(dims, columns=["grupo", "n_colunas", "n_linhas", "dimensoes_95", "pc1", "razao_participacao"]).to_csv(pasta / "colinearidade_dimensoes.csv", index=False)
    pd.DataFrame(pares, columns=["grupo", "coluna_a", "coluna_b", "spearman", "pearson", "abs_max", "acima_limite"]).to_csv(pasta / "colinearidade.csv", index=False)
    pd.DataFrame(clusters, columns=["grupo", "coluna", "cluster"]).to_csv(pasta / "colinearidade_clusters.csv", index=False)
    pd.DataFrame(conj, columns=["conjunto", "vif_maximo_regra", "reservatorio", "vif_max", "dimensoes_95", "n_linhas", "colunas"]).to_csv(pasta / "conjuntos_propostos.csv", index=False)


def test_comparar_com_csvs_feitos_a_mao(tmp_path):
    g = "todas|sem_diag"
    _csvs(tmp_path / "base", [["todas", 3, 100, 2, .7, 1.5]], [[g, "a", "b", 1, 1, 1, True], [g, "b", "c", .99, .99, .99, True]],
          [["todas", "a", 0], ["todas", "b", 1], ["todas", "c", 2]], [["so_tau", 10.0, 3, 2.0, 2, 100, "a b c"], ["so_tau", 30.0, 3, 2.0, 2, 100, "a b c"]])
    _csvs(tmp_path / "ab", [["todas", 3, 60, 1, .9, 1.1]], [[g, "b", "a", 1, 1, 1, True], [g, "a", "d", .97, .97, .97, True]],
          [["todas", "a", 0], ["todas", "b", 1], ["todas", "d", 2]], [["so_tau", 10.0, 3, 2.0, 2, 60, "a b d"], ["so_tau", 30.0, 3, 2.0, 2, 60, "a b c"]])
    L, d = col.comparar_com_base(tmp_path / "ab", tmp_path / "base")
    assert d["pares_entraram"] == ["a | d"] and d["pares_sairam"] == ["b | c"]                       # (a,b) e (b,a) são o mesmo par
    assert d["colunas_entraram"] == ["d"] and d["colunas_sairam"] == ["c"]
    assert d["conjuntos"]["so_tau|10"] == {"mantidas": ["a", "b"], "so_base": ["c"], "so_ab": ["d"]}
    assert d["conjuntos"]["so_tau|30"] == {"mantidas": ["a", "b", "c"], "so_base": [], "so_ab": []}
    assert any("[so_tau|10]" in x for x in L)


def test_comparar_sem_base_avisa_e_nao_quebra(tmp_path):
    L, d = col.comparar_com_base(tmp_path / "ab", tmp_path / "nao_existe")
    assert d == {} and "falta" in L[0]
