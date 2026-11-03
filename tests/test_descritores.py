"""Testes dos descritores sem modelo (P3, 03/11/2026): valores conhecidos, vazio no ruído, sem uso de prop_*. Rode: pytest"""
import numpy as np
import pandas as pd
import pytest

from luminescencia.descritores import PREFIXO, calcular, descritores_serie, dicionario, parametros
from luminescencia.modelos import exp_simples_c0

T = np.arange(0, 57, 1 / 7.5) + 2.5
P = parametros({})


def _y(tau=10.0, A=100.0, ruido=0.1, seed=0):
    return exp_simples_c0(T - T[0], A, tau) + np.random.default_rng(seed).normal(0, ruido, len(T))


def test_valores_da_exponencial_simples():
    d = descritores_serie(T, _y(10.0), P, 3.0, 31, 5.0)
    assert abs(d["t50"] - 10 * np.log(2)) < 0.3 and abs(d["t10"] - 10 * np.log(10)) < 0.5
    assert abs(d["r_5s"] - np.exp(-0.5)) < 0.03 and abs(d["r_10s"] - np.exp(-1)) < 0.03
    assert abs(d["auc_10s"] - 10 * (1 - np.exp(-1))) < 0.4
    assert abs(d["I0"] - 100) < 3


def test_vazio_quando_o_nivel_ou_o_tempo_ja_sumiu_no_ruido():
    d = descritores_serie(T, _y(tau=2.0, A=20.0), P, 3.0, 31, 5.0)      # cai ao ruído em poucos segundos
    assert np.isnan(d["r_40s"]) and np.isnan(d["auc_30s"])
    ruidoso = descritores_serie(T, _y(tau=10.0, A=1.0, ruido=0.2), P, 3.0, 31, 5.0)   # 10% de I0 = 0,1 < 3 x ruído
    assert ruidoso is None or np.isnan(ruidoso["t10"])


def test_sinal_curto_ou_nulo_nao_tem_descritor():
    assert descritores_serie(T, _y(tau=0.3, A=5.0), P, 3.0, 31, 5.0) is None
    assert descritores_serie(T[:5], _y()[:5], P, 3.0, 31, 5.0) is None


def _entradas(prop):
    s = pd.DataFrame({"video": "v1", "tempo_s": T, "corr_R": _y(6.0), "corr_G": _y(14.0, seed=1), "corr_B": _y(18.0, A=150, seed=2)})
    v = {"video": "v1", "amostra": "A", "grupo": "treino", "num_video": 1, "proporcao": "G1",
         "satura_apos_2s": False, "satura_apos_3s": False}
    for i, c in enumerate("RGB"):
        v.update({f"usar_no_ajuste_{c}": True, f"prop_{c}": prop[i], f"canal_ativo_{c}": prop[i] > 0})
    return s, pd.DataFrame([v])


def test_nao_olha_a_proporcao_e_calcula_razoes_entre_canais():
    s, a = _entradas((0.0, 1.0, 0.0))
    _, b = _entradas((1.0, 0.0, 0.5))
    da, db = calcular(s, a, {}), calcular(s, b, {})
    assert da.equals(db)                                          # mudar o alvo não muda nenhum descritor
    r = da.iloc[0]
    assert abs(r["razao_t50_G_B"] - (14 * np.log(2)) / (18 * np.log(2))) < 0.1
    assert abs(r["razao_I0_G_B"] - 100 / 150) < 0.05
    assert bool(r[f"R_{PREFIXO}_ok"])


def test_canal_sem_ajuste_fica_vazio_e_dicionario_marca_papeis():
    s, a = _entradas((0.0, 1.0, 0.0))
    a["usar_no_ajuste_R"] = False
    d = calcular(s, a, {})
    assert not bool(d.loc[0, f"R_{PREFIXO}_ok"]) and np.isnan(d.loc[0, f"R_{PREFIXO}_t50"]) and np.isnan(d.loc[0, "razao_t50_R_G"])
    dic = dicionario(d, P, 2.5).set_index("coluna")
    assert dic.loc["video", "papel"] == "meta" and dic.loc[f"G_{PREFIXO}_t50", "papel"] == "feature"
    assert dic.loc[f"G_{PREFIXO}_ok", "papel"] == "flag" and dic.loc["razao_I0_G_B", "papel"] == "feature"
    assert not any(c.startswith("prop_") or c.startswith("canal_ativo_") for c in d.columns)


def test_nivel_de_queda_invalido_e_erro():
    with pytest.raises(ValueError):
        parametros({"descritores": {"niveis_queda": [1.5]}})


def test_dicionario_avisa_que_alguns_descritores_nao_servem_no_R_mas_mantem_a_coluna():
    s, a = _entradas((0.0, 1.0, 0.0))
    d = calcular(s, a, {})
    dic = dicionario(d, P, 2.5).set_index("coluna")
    for k in ("t10", "r_20s", "r_40s"):
        assert "AVISO" in dic.loc[f"R_{PREFIXO}_{k}", "descricao"] and f"R_{PREFIXO}_{k}" in d.columns
        assert "AVISO" not in dic.loc[f"G_{PREFIXO}_{k}", "descricao"]
    assert "AVISO" not in dic.loc[f"R_{PREFIXO}_t50", "descricao"]
    desligado = dicionario(d, parametros({"descritores": {"nao_confiaveis_R": []}}), 2.5).set_index("coluna")
    assert "AVISO" not in desligado.loc[f"R_{PREFIXO}_t10", "descricao"]
