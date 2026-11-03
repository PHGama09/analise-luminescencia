"""Testes da escolha de entradas da IA (D1): nunca meta, alvo ou canal_ativo; a regra de sinal não olha prop_*. Rode: pytest"""
import numpy as np
import pandas as pd
import pytest

from luminescencia.entradas import aplicar_regra_de_sinal, colunas_de_entrada, montar_entradas, parametros


def _tabela():
    df = pd.DataFrame({
        "video": ["a", "b"], "amostra": ["A", "A"], "prop_G": [1.0, 0.0], "canal_ativo_G": [True, False],
        "sinal_presente_G": [True, True], "inicio_corr_G": [200.0, 120.0], "G_cl_ok": [True, True], "G_c0_ok": [True, True],
        "G_cl_tau_medio": [10.0, 9.0], "G_c0_tau_medio": [14.0, 13.0], "excluido_saturacao": [False, False],
        "usar_no_ajuste_G": [True, True], "sat_ate_s": [0.0, 0.0]})
    dic = pd.DataFrame({"coluna": df.columns, "papel": [
        "meta", "meta", "alvo", "flag", "flag", "feature", "flag", "flag", "feature", "feature", "flag", "flag", "flag"]})
    return df, dic


def test_nunca_meta_alvo_ou_canal_ativo():
    df, dic = _tabela()
    cols = colunas_de_entrada(dic, parametros({}))
    assert not {"video", "amostra", "prop_G", "canal_ativo_G", "excluido_saturacao", "usar_no_ajuste_G"} & set(cols)
    assert {"sinal_presente_G", "G_cl_tau_medio", "G_c0_tau_medio", "G_cl_ok"} <= set(cols)


def test_so_um_caminho_tira_tambem_as_flags_do_outro():
    df, dic = _tabela()
    cols = colunas_de_entrada(dic, parametros({"entradas": {"caminhos": ["cl"]}}))
    assert "G_cl_tau_medio" in cols and "G_c0_tau_medio" not in cols and "G_c0_ok" not in cols


def test_desligar_canais_sem_po_exige_regra_medida():
    with pytest.raises(ValueError):
        parametros({"entradas": {"canais_sem_po": False}})


def test_regra_de_sinal_usa_o_nivel_medido_e_nao_o_alvo():
    df, dic = _tabela()
    cfg = {"entradas": {"canais_sem_po": False, "limiar_inicio_corr": {"R": 0, "G": 150, "B": 0}}}
    out = montar_entradas(df, dic, cfg)
    assert out.loc[0, "G_cl_tau_medio"] == 10.0 and np.isnan(out.loc[1, "G_cl_tau_medio"])      # b: sinal fraco
    # trocar o alvo (prop_G / canal_ativo_G) não muda quem é esvaziado
    df2 = df.copy(); df2["prop_G"] = [0.0, 1.0]; df2["canal_ativo_G"] = [False, True]
    out2 = montar_entradas(df2, dic, cfg)
    assert out.equals(out2)
    assert not bool(aplicar_regra_de_sinal(df, parametros(cfg))["G_cl_ok"].iloc[1])


def test_padrao_deixa_tudo_como_esta():
    df, dic = _tabela()
    out = montar_entradas(df, dic, {})
    assert out["G_cl_tau_medio"].tolist() == [10.0, 9.0]
