"""Proteções das entradas da IA (v0.20.0): sem_intensidade e max_ausentes_treino, com dados de resposta conhecida."""
import numpy as np
import pandas as pd
import pytest

from luminescencia import entradas as en


def _df():
    n = 10
    am = ["A"] * 4 + ["B"] * 4 + ["C"] * 2
    return pd.DataFrame({
        "video": [f"v{i}" for i in range(n)], "amostra": am, "excluido_saturacao": [False] * 9 + [True],
        "R_c0_tau_medio": np.arange(n) + 1.0, "R_c0_A1": 1.0, "R_md_I0": 2.0, "razao_I0_R_G": 0.5, "razao_t50_R_G": 0.9,
        "inicio_corr_R": 100.0, "R_md_t50": 3.0,
        "R_md_r_40s": [np.nan, 1, 1, 1, 1, 1, 1, 1, 1, 1],          # 1 de 8 no treino = 12,5% ausente
        "R_md_r_20s": [1, 1, 1, 1, 1, 1, 1, 1, np.nan, np.nan],   # ausentes só em C e no saturado: não conta
    })


def _dic(df):
    return pd.DataFrame({"coluna": df.columns, "papel": ["meta", "meta", "flag"] + ["feature"] * (len(df.columns) - 3)})


CFG = {"divisao": {"treino": ["A", "B"]}, "entradas": {"caminhos": ["c0"]}}


def test_eh_intensidade_separa_intensidade_de_forma():
    sim = ["R_md_I0", "razao_I0_R_G", "inicio_corr_B", "final_corr_G", "R_c0_A1", "G_cl_A2", "B_cl_C"]
    nao = ["R_c0_tau_medio", "R_md_t50", "R_md_r_5s", "R_md_auc_10s", "razao_t50_R_G", "R_cl_fracao_rapida", "G_cl_tau2"]
    assert all(en.eh_intensidade(c) for c in sim) and not any(en.eh_intensidade(c) for c in nao)


def test_sem_intensidade_tira_so_as_colunas_de_intensidade():
    df = _df()
    cfg = {**CFG, "entradas": {**CFG["entradas"], "sem_intensidade": True}}
    cols = list(en.montar_entradas(df, _dic(df), cfg).columns)
    assert "R_md_I0" not in cols and "razao_I0_R_G" not in cols and "R_c0_A1" not in cols and "inicio_corr_R" not in cols
    assert {"R_c0_tau_medio", "R_md_t50", "razao_t50_R_G"} <= set(cols)
    assert "R_md_I0" in en.montar_entradas(df, _dic(df), CFG).columns                 # desligada: nada sai


def test_max_ausentes_usa_so_o_treino():
    df = _df()
    cfg = {**CFG, "entradas": {**CFG["entradas"], "max_ausentes_treino": 0.10}}
    cols = list(en.montar_entradas(df, _dic(df), cfg).columns)
    assert "R_md_r_40s" not in cols                 # 12,5% no treino > 10%
    assert "R_md_r_20s" in cols                     # ausente só no C e no saturado: o C não decide
    cfg["entradas"]["max_ausentes_treino"] = 0.20
    assert "R_md_r_40s" in en.montar_entradas(df, _dic(df), cfg).columns


def test_max_ausentes_nao_muda_com_a_amostra_c():
    df = _df()
    df2 = df.copy()
    df2.loc[df2["amostra"] == "C", ["R_c0_tau_medio", "R_md_t50"]] = np.nan          # C toda vazia nessas colunas
    cfg = {**CFG, "entradas": {**CFG["entradas"], "max_ausentes_treino": 0.10}}
    assert list(en.montar_entradas(df, _dic(df), cfg).columns) == list(en.montar_entradas(df2, _dic(df2), cfg).columns)


@pytest.mark.parametrize("valor", [1.5, -0.1, "0.1", True])
def test_max_ausentes_invalido(valor):
    with pytest.raises(ValueError, match="max_ausentes_treino"):
        en.parametros({"entradas": {"max_ausentes_treino": valor}})


def test_sem_intensidade_invalido():
    with pytest.raises(ValueError, match="sem_intensidade"):
        en.parametros({"entradas": {"sem_intensidade": "sim"}})
