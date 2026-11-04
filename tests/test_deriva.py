"""Testes sintéticos da deriva A/B/C (B5): respostas conhecidas, nada escrito no código. Rode: pytest"""
import hashlib

import numpy as np
import pandas as pd
import pytest

from luminescencia import deriva

PADRAO = np.array([0.98, 1.00, 1.02, 0.99, 1.01])          # 5 regravações com ~1,6% de CV
NIVEIS = {"P1": 100.0, "P2": 80.0, "P3": 60.0, "P4": 40.0}


def _rng():
    return np.random.default_rng(0)


def test_sem_deriva_nao_detecta_e_razao_e_um():
    v = {s: 100 * PADRAO for s in "ABC"}                     # as três amostras idênticas
    r = deriva.estatisticas_celula(v, list("ABC"), 500, _rng())
    assert r["razao_BA"] == pytest.approx(1.0) and r["razao_CA"] == pytest.approx(1.0)
    assert r["eta2"] == pytest.approx(0.0, abs=1e-12) and r["p_perm"] == pytest.approx(1.0) and r["p_kw"] == pytest.approx(1.0)


def test_deriva_injetada_de_10_por_cento_e_detectada():
    v = {"A": 100 * PADRAO, "B": 110 * PADRAO, "C": 100 * PADRAO}
    r = deriva.estatisticas_celula(v, list("ABC"), 2000, _rng())
    assert r["razao_BA"] == pytest.approx(1.10) and r["razao_CA"] == pytest.approx(1.0)
    assert r["eta2"] > 0.9 and r["p_perm"] < 0.01 and r["p_kw"] < 0.05
    assert r["ordem"].startswith("B")
    assert r["diferenca_rel"] > 0.09 and r["cv_reg"] == pytest.approx(np.std(PADRAO, ddof=1) / np.mean(PADRAO), rel=1e-6)
    assert r["d"] > 3 * r["d_nulo"]                            # muito acima do esperado sem deriva


def test_ordem_decrescente_das_medias():
    v = {"A": 100 * PADRAO, "B": 90 * PADRAO, "C": 80 * PADRAO}
    assert deriva.estatisticas_celula(v, list("ABC"), 200, _rng())["ordem"] == "A>B>C"


def _decomp_df(deslocamento_comum=0.0, deriva_so_na_p1=0.0):
    linhas = []
    for prop, nivel in NIVEIS.items():
        for s, desl in zip("ABC", [0.0, 0.0, 0.0]):
            extra = deslocamento_comum * (s == "B") + deriva_so_na_p1 * (s == "B" and prop == "P1")
            for x in PADRAO:
                linhas.append((prop, s, float(np.log(nivel * x) + extra)))
    return pd.DataFrame(linhas, columns=["prop", "amostra", "y"])


def test_decomposicao_soma_um_e_proporcao_domina():
    r = deriva.decomposicao(_decomp_df())
    assert r["frac_proporcao"] + r["frac_amostra"] + r["frac_regravacao"] == pytest.approx(1.0)
    assert r["frac_proporcao"] > 0.99 and r["frac_amostra"] == pytest.approx(0.0, abs=1e-12)


def test_deslocamento_comum_de_uma_amostra_e_removivel_por_normalizar():
    r = deriva.decomposicao(_decomp_df(deslocamento_comum=0.1))
    assert r["frac_amostra"] > 10 * r["frac_regravacao"]                   # a deriva supera de longe a variação entre regravações
    assert r["frac_offset_comum"] == pytest.approx(1.0, abs=1e-9)         # mesmo fator em todas as proporções


def test_deriva_especifica_de_uma_proporcao_nao_e_removivel():
    r = deriva.decomposicao(_decomp_df(deriva_so_na_p1=0.2))
    assert r["frac_amostra"] > 10 * r["frac_regravacao"] and r["frac_offset_comum"] < 0.5


# ----------------------------------------------------------------------------------------------------------- ponta a ponta
def _tabelas(pasta, deriva_b=0.0, embaralhar_prop=False):
    linhas = []
    for prop, nivel in NIVEIS.items():
        for s in "ABC":
            for k, x in enumerate(PADRAO, 1):
                linhas.append({"video": f"{prop}_{s}{k}", "amostra": s, "grupo": "treino" if s != "C" else "validacao", "num_video": k, "proporcao": prop,
                               "prop_R": 1.0, "inicio_corr_R": 150.0, "inicio_corr_G": 5.0, "inicio_corr_B": 5.0,
                               "R_md_I0": nivel * x * (1 + deriva_b * (s == "B")),         # com deriva só em B
                               "R_md_t50": nivel / 10 * x,                                  # idêntico entre amostras
                               "G_md_I0": 3.0 * x, "B_md_I0": 2.0 * x})
    ds = pd.DataFrame(linhas)
    if embaralhar_prop:
        ds["prop_R"] = np.random.default_rng(1).random(len(ds))
    pasta.mkdir(parents=True)
    ds.drop(columns=[c for c in ds if c.endswith("_md_I0") or c.endswith("_md_t50")]).to_csv(pasta / "dataset.csv", index=False)
    ds[["video", "amostra", "grupo", "num_video", "proporcao"] + [c for c in ds if "_md_" in c]].to_csv(pasta / "descritores.csv", index=False)
    return pasta


def _cfg(**extra):
    d = {"n_permutacoes": 500, "semente": 7}
    d.update(extra)
    return {"deriva": d, "protocolo": {"amostras": ["A", "B", "C"]}}


def test_rodar_detecta_a_deriva_injetada_e_so_ela(tmp_path):
    tab = _tabelas(tmp_path / "tabela", deriva_b=0.10)
    assert deriva.rodar(_cfg(), pasta_tabela=tab, saida=tmp_path / "saida") == 0
    cel = pd.read_csv(tmp_path / "saida" / "deriva_celulas.csv")
    assert set(cel["grupo"]) == {"R"}                                        # G e B ficam de fora pela regra medida (inicio_corr baixo)
    i0, t50 = cel[cel["grandeza"] == "I0"], cel[cel["grandeza"] == "t50"]
    assert len(i0) == 4 and i0["detectada"].all() and (i0["razao_BA"].round(6) == 1.1).all()
    assert len(t50) == 4 and not t50["detectada"].any()
    txt = (tmp_path / "saida" / "relatorio_deriva.txt").read_text(encoding="utf-8")
    assert "4 de 8 células" in txt and "Há deriva" in txt
    assert (tmp_path / "saida" / "deriva_mapa.png").stat().st_size > 1000
    desc = pd.read_csv(tmp_path / "saida" / "deriva_descartadas.csv")
    assert desc["motivo"].str.contains("ausente").any() and desc["motivo"].str.contains("limiar").any()


def test_sem_deriva_a_conclusao_diz_que_nao_ha(tmp_path):
    tab = _tabelas(tmp_path / "tabela", deriva_b=0.0)
    deriva.rodar(_cfg(), pasta_tabela=tab, saida=tmp_path / "saida")
    cel = pd.read_csv(tmp_path / "saida" / "deriva_celulas.csv")
    assert not cel["detectada"].any()
    assert "Não há (distinguível do acaso) deriva" in (tmp_path / "saida" / "relatorio_deriva.txt").read_text(encoding="utf-8")


def test_nao_usa_prop_e_nao_altera_as_entradas(tmp_path):
    a = _tabelas(tmp_path / "a" / "tabela", deriva_b=0.1)
    b = _tabelas(tmp_path / "b" / "tabela", deriva_b=0.1, embaralhar_prop=True)         # só prop_R muda
    hashes = {f: hashlib.md5((a / f).read_bytes()).hexdigest() for f in ("dataset.csv", "descritores.csv")}
    deriva.rodar(_cfg(), pasta_tabela=a, saida=tmp_path / "sa")
    deriva.rodar(_cfg(), pasta_tabela=b, saida=tmp_path / "sb")
    assert (tmp_path / "sa" / "deriva_celulas.csv").read_text() == (tmp_path / "sb" / "deriva_celulas.csv").read_text()
    assert hashes == {f: hashlib.md5((a / f).read_bytes()).hexdigest() for f in hashes}   # análise não muda os CSVs


def test_deterministico_com_a_mesma_semente(tmp_path):
    tab = _tabelas(tmp_path / "tabela", deriva_b=0.05)
    deriva.rodar(_cfg(), pasta_tabela=tab, saida=tmp_path / "s1")
    deriva.rodar(_cfg(), pasta_tabela=tab, saida=tmp_path / "s2")
    assert (tmp_path / "s1" / "deriva_celulas.csv").read_text() == (tmp_path / "s2" / "deriva_celulas.csv").read_text()


def test_celula_com_poucas_regravacoes_e_descartada():
    df = pd.DataFrame({"proporcao": ["P"] * 7, "amostra": list("AAABBCC"), "inicio_corr_R": [100.0] * 7, "R_md_I0": [1, 2, 3, 4, 5, 6, 7.0]})
    p = deriva.parametros({"deriva": {"grandezas": {"I0": "{c}_md_I0"}, "pares": []}})
    cel, desc, _ = deriva.calcular_celulas(df, p, list("ABC"))
    assert cel.empty and desc["motivo"].str.contains("poucas regravações").any()


def test_parametros_invalidos():
    with pytest.raises(ValueError):
        deriva.parametros({"deriva": {"min_n_por_amostra": 1}})
    with pytest.raises(ValueError):
        deriva.parametros({"deriva": {"n_permutacoes": 10}})
