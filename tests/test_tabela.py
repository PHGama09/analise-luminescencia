"""Testes da tabela final: dois caminhos lado a lado, flags e dicionário. Rode: pytest"""
import pandas as pd

from luminescencia.tabela import dicionario, montar


def _entradas():
    vpos = pd.DataFrame([
        dict(video="v1", amostra="A", grupo="treino", num_video=1, proporcao="G1", prop_R=0.0, prop_G=1.0, prop_B=0.0, sat_ate_s=0.0,
             canal_ativo_R=False, canal_ativo_G=True, canal_ativo_B=False, sinal_presente_R=False, sinal_presente_G=True, sinal_presente_B=False,
             usar_no_ajuste_R=False, usar_no_ajuste_G=True, usar_no_ajuste_B=False),
        dict(video="v2", amostra="A", grupo="treino", num_video=2, proporcao="G1", prop_R=0.0, prop_G=1.0, prop_B=0.0, sat_ate_s=3.0,
             canal_ativo_R=False, canal_ativo_G=True, canal_ativo_B=False, sinal_presente_R=False, sinal_presente_G=True, sinal_presente_B=False,
             usar_no_ajuste_R=False, usar_no_ajuste_G=False, usar_no_ajuste_B=False)])
    linha = lambda cam, **k: dict(video="v1", canal="G", caminho=cam, modelo="exp_dupla", A1=1, tau1=2, A2=3, tau2=9, tau_medio=7, r2=.99, rmse=.1,
                                  inicio_s=2.5, fim_s=59, n_pontos=400, **k)
    aj = pd.DataFrame([linha("c_livre", C=4.0), linha("principal")])
    vcons = pd.DataFrame({"video": ["v1", "v2"], **{f"{k}_corr_{c}": 1.0 for k in ("inicio", "final") for c in "RGB"}})
    return aj, vpos, vcons


def test_dois_caminhos_lado_a_lado_e_flags():
    d = montar(*_entradas()).set_index("video")
    assert d.loc["v1", "G_cl_C"] == 4.0 and "G_c0_C" not in d.columns
    assert d.loc["v1", "G_cl_tau_medio"] == d.loc["v1", "G_c0_tau_medio"] == 7
    assert bool(d.loc["v1", "G_cl_ok"]) and not bool(d.loc["v2", "G_c0_ok"])
    assert bool(d.loc["v2", "excluido_saturacao"]) and not bool(d.loc["v1", "excluido_saturacao"])


def test_dicionario_marca_papeis_e_o_vazamento():
    dic = dicionario(montar(*_entradas())).set_index("coluna")
    assert dic.loc["video", "papel"] == "meta" and dic.loc["prop_G", "papel"] == "alvo" and dic.loc["G_cl_tau1", "papel"] == "feature"
    assert dic.loc["canal_ativo_G", "papel"] == "flag" and "vaza" in dic.loc["canal_ativo_G", "descricao"]
    assert "pode ser entrada" in dic.loc["sinal_presente_G", "descricao"] and "pode ser entrada" in dic.loc["G_cl_ok", "descricao"]
