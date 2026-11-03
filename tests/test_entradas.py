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


# ---- Q3 (03/11/2026): descritores sem modelo como opção de entrada (chave desligada por padrão)
def _descritores():
    desc = pd.DataFrame({"video": ["b", "a"], "amostra": ["A", "A"], "R_md_ok": [True, True], "R_md_t50": [3.0, 4.0], "R_md_t10": [15.0, 16.0],
                         "G_md_ok": [True, True], "G_md_t50": [5.0, 6.0], "razao_t50_R_G": [0.6, 0.7], "razao_t50_G_B": [0.9, 0.95]})
    dic = pd.DataFrame({"coluna": desc.columns, "papel": ["meta", "meta", "flag", "feature", "feature", "flag", "feature", "feature", "feature"]})
    return desc, dic


def test_descritores_ficam_desligados_por_padrao():
    df, dic = _tabela()
    desc, dic_desc = _descritores()
    out = montar_entradas(df, dic, {}, desc, dic_desc)
    assert not any("_md_" in c or c.startswith("razao_") for c in out.columns)
    assert out.equals(montar_entradas(df, dic, {}))


def test_ligar_descritores_junta_por_video_sem_meta_nem_alvo():
    df, dic = _tabela()
    desc, dic_desc = _descritores()                          # note: linhas em ordem diferente (b, a); a junção é por video
    out = montar_entradas(df, dic, {"entradas": {"descritores": True}}, desc, dic_desc)
    assert out.loc[0, "R_md_t50"] == 4.0 and out.loc[1, "R_md_t50"] == 3.0            # a = linha 0 do dataset
    assert {"G_md_t50", "razao_t50_G_B", "R_md_ok"} <= set(out.columns)
    assert not {"video", "amostra", "prop_G", "canal_ativo_G"} & set(out.columns)
    df2 = df.copy(); df2["prop_G"] = [0.0, 1.0]; df2["canal_ativo_G"] = [False, True]
    assert out.equals(montar_entradas(df2, dic, {"entradas": {"descritores": True}}, desc, dic_desc))      # mudar o alvo não muda nada


def test_regra_de_sinal_esvazia_tambem_os_descritores_e_as_razoes_do_canal_fraco():
    df, dic = _tabela()
    desc, dic_desc = _descritores()
    cfg = {"entradas": {"descritores": True, "canais_sem_po": False, "limiar_inicio_corr": {"R": 0, "G": 150, "B": 0}}}
    out = montar_entradas(df, dic, cfg, desc, dic_desc)                                # b tem inicio_corr_G = 120 < 150
    assert np.isnan(out.loc[1, "G_md_t50"]) and out.loc[0, "G_md_t50"] == 6.0
    assert np.isnan(out.loc[1, "razao_t50_R_G"]) and np.isnan(out.loc[1, "razao_t50_G_B"]) and not bool(out.loc[1, "G_md_ok"])
    assert out.loc[1, "R_md_t50"] == 3.0                                               # R não é fraco
    assert out.loc[0, "razao_t50_R_G"] == 0.7


def test_tirar_os_descritores_nao_confiaveis_do_R_e_opcional():
    df, dic = _tabela()
    desc, dic_desc = _descritores()
    cheio = montar_entradas(df, dic, {"entradas": {"descritores": True}}, desc, dic_desc)
    enxuto = montar_entradas(df, dic, {"entradas": {"descritores": True, "descritores_sem_nao_confiaveis_R": True}}, desc, dic_desc)
    assert "R_md_t10" in cheio.columns and "R_md_t10" not in enxuto.columns and "R_md_t50" in enxuto.columns


def test_descritores_ligados_sem_tabela_ou_com_valor_invalido_e_erro():
    df, dic = _tabela()
    desc, dic_desc = _descritores()
    with pytest.raises(ValueError):
        montar_entradas(df, dic, {"entradas": {"descritores": True}})
    with pytest.raises(ValueError):
        parametros({"entradas": {"descritores": "sim"}})
    with pytest.raises(ValueError):                                                    # vídeo do dataset sem descritor
        montar_entradas(df, dic, {"entradas": {"descritores": True}}, desc[desc["video"] != "a"], dic_desc)


def test_carregar_entradas_le_a_pasta_da_tabela(tmp_path):
    df, dic = _tabela()
    desc, dic_desc = _descritores()
    for nome, t in (("dataset.csv", df), ("dicionario.csv", dic), ("descritores.csv", desc), ("dicionario_descritores.csv", dic_desc)):
        t.to_csv(tmp_path / nome, index=False, encoding="utf-8-sig")
    from luminescencia.entradas import carregar_entradas
    assert "R_md_t50" not in carregar_entradas({}, tmp_path).columns
    assert "R_md_t50" in carregar_entradas({"entradas": {"descritores": True}}, tmp_path).columns
