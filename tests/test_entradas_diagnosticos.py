"""E1 (04/11/2026): colunas de diagnóstico do ajuste/janela não são entrada da IA (padrão), mas ficam disponíveis para controle de qualidade. Rode: pytest"""
import numpy as np
import pandas as pd
import pytest

from luminescencia.entradas import (colunas_de_diagnostico, colunas_de_descritores, colunas_de_entrada, eh_diagnostico,
                                    montar_entradas, parametros)

DIAG = ["R_cl_ok", "G_c0_ok", "B_md_ok", "R_cl_r2", "G_c0_rmse", "B_cl_n_pontos", "R_c0_fim_s", "G_cl_inicio_s"]
NAO_DIAG = ["R_cl_tau_medio", "G_c0_tau1", "B_cl_C", "R_cl_fracao_rapida", "R_md_t50", "G_md_I0", "razao_t50_R_G",
            "inicio_corr_G", "final_corr_B", "sinal_presente_G", "R_cl_modelo", "R_cl_A1"]


def _tabela():
    cols = ["video", "amostra", "prop_G", "canal_ativo_G", "inicio_corr_G", "sinal_presente_G", "G_cl_tau_medio", "G_cl_ok", "G_cl_r2",
            "G_cl_rmse", "G_cl_inicio_s", "G_cl_fim_s", "G_cl_n_pontos", "G_c0_tau_medio", "G_c0_ok", "G_c0_r2"]
    papel = ["meta", "meta", "alvo", "flag", "feature", "flag", "feature", "flag", "feature", "feature", "feature", "feature", "feature",
             "feature", "flag", "feature"]
    df = pd.DataFrame([["a", "A", 1.0, True, 200.0, True, 10.0, True, 0.999, 0.1, 2.5, 40.0, 450, 14.0, True, 0.998],
                       ["b", "A", 0.0, False, 120.0, True, 9.0, True, 0.99, 0.2, 2.5, 30.0, 300, 13.0, True, 0.99]], columns=cols)
    return df, pd.DataFrame({"coluna": cols, "papel": papel})


@pytest.mark.parametrize("c", DIAG)
def test_reconhece_diagnosticos(c):
    assert eh_diagnostico(c)


@pytest.mark.parametrize("c", NAO_DIAG)
def test_nao_confunde_feature_com_diagnostico(c):
    assert not eh_diagnostico(c)


def test_padrao_bloqueia_diagnosticos_e_mantem_as_features():
    df, dic = _tabela()
    assert parametros({})["bloquear_diagnosticos"] is True
    cols = colunas_de_entrada(dic, parametros({}))
    assert not set(colunas_de_diagnostico(df.columns)) & set(cols)
    assert {"G_cl_tau_medio", "G_c0_tau_medio", "inicio_corr_G", "sinal_presente_G"} <= set(cols)
    out = montar_entradas(df, dic, {})
    assert out["G_cl_tau_medio"].tolist() == [10.0, 9.0] and not colunas_de_diagnostico(out.columns)


def test_diagnosticos_continuam_disponiveis_para_controle_de_qualidade():
    df, _ = _tabela()
    assert set(colunas_de_diagnostico(df.columns)) == {"G_cl_ok", "G_cl_r2", "G_cl_rmse", "G_cl_inicio_s", "G_cl_fim_s", "G_cl_n_pontos", "G_c0_ok", "G_c0_r2"}


def test_chave_false_e_uma_variante_exploratoria_explicita():
    df, dic = _tabela()
    cfg = {"entradas": {"bloquear_diagnosticos": False}}
    assert {"G_cl_r2", "G_cl_ok", "G_cl_n_pontos"} <= set(colunas_de_entrada(dic, parametros(cfg)))
    assert "G_cl_fim_s" in montar_entradas(df, dic, cfg).columns


def test_valor_invalido_e_erro():
    with pytest.raises(ValueError):
        parametros({"entradas": {"bloquear_diagnosticos": "sim"}})


def test_bloquear_diagnosticos_nao_libera_meta_alvo_nem_canal_ativo():
    df, dic = _tabela()
    for cfg in ({}, {"entradas": {"bloquear_diagnosticos": False}}):
        cols = set(colunas_de_entrada(dic, parametros(cfg)))
        assert not {"video", "amostra", "prop_G", "canal_ativo_G"} & cols


def test_descritores_tambem_sem_diagnostico_por_padrao():
    desc = pd.DataFrame({"video": ["a", "b"], "amostra": ["A", "A"], "R_md_ok": [True, True], "R_md_t50": [3.0, 4.0], "razao_t50_R_G": [0.6, 0.7]})
    dic = pd.DataFrame({"coluna": desc.columns, "papel": ["meta", "meta", "flag", "feature", "feature"]})
    assert colunas_de_descritores(dic, {}) == ["R_md_t50", "razao_t50_R_G"]
    assert "R_md_ok" in colunas_de_descritores(dic, {"entradas": {"bloquear_diagnosticos": False}})


def test_regra_de_sinal_continua_valendo_com_o_bloqueio():
    df, dic = _tabela()
    cfg = {"entradas": {"canais_sem_po": False, "limiar_inicio_corr": {"G": 150}}}
    out = montar_entradas(df, dic, cfg)                                  # b tem inicio_corr_G = 120 < 150
    assert np.isnan(out.loc[1, "G_cl_tau_medio"]) and out.loc[0, "G_cl_tau_medio"] == 10.0
    assert not colunas_de_diagnostico(out.columns)
