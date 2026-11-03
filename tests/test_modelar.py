"""Testes do ajuste em lote (C = 0): fim da janela no ruído, ruído do canal, escolha de caminhos. Rode: pytest"""
import numpy as np

from luminescencia.modelar import _um_ajuste, fim_util, ruido_do_canal
from luminescencia.modelos import ajustar_serie, exp_simples_c0

T = np.arange(0, 57, 1 / 7.5)


def test_ruido_do_canal_perto_do_sigma():
    y = np.random.default_rng(0).normal(0, 0.5, len(T))
    assert abs(ruido_do_canal(y, T) - 0.5) < 0.1


def test_fim_util_corta_onde_o_sinal_some():
    rng = np.random.default_rng(1)
    y = exp_simples_c0(T, 20, 3.0) + rng.normal(0, 0.2, len(T))       # cai a ~0 depois de ~15 s
    fim, ruido = fim_util(T, y, 3.0, 31)
    assert 8 < fim < 30 and ruido < 0.3


def test_sinal_forte_usa_a_serie_inteira():
    y = exp_simples_c0(T, 200, 30.0) + np.random.default_rng(2).normal(0, 0.2, len(T))
    fim, _ = fim_util(T, y, 3.0, 31)
    assert fim > 55


def test_c0_recupera_tau_simples():
    y = exp_simples_c0(T, 30, 12) + np.random.default_rng(3).normal(0, 0.3, len(T))
    r = ajustar_serie(T, y, ("exp_simples_c0", "exp_dupla_c0"))
    assert r["modelo"] == "exp_simples_c0" and abs(r["tau"] - 12) < 1


def test_sinal_curto_nao_e_ajustado():
    y = exp_simples_c0(T, 5, 0.3) + np.random.default_rng(4).normal(0, 0.3, len(T))
    r = _um_ajuste(({"video": "x", "canal": "R", "caminho": "principal"}, T, y, 2.5, ("exp_simples_c0",), 3.0, 31, 5.0))
    assert r["modelo"] == "" and "ruído" in r["observacao"]


def test_janela_comeca_no_instante_pedido():
    y = exp_simples_c0(T, 200, 30.0)
    r = _um_ajuste(({"video": "x", "canal": "G", "caminho": "inicio_3s"}, T, y, 3.0, ("exp_simples_c0",), 3.0, 31, 5.0))
    assert abs(r["inicio_s"] - 3.0) < 1e-6 and abs(r["tau"] - 30) < 0.5
