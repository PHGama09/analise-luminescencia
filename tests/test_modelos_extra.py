"""Testes dos modelos extras (D3): só comparação, o principal não muda. Rode: pytest"""
import numpy as np

from luminescencia.modelos import MODELOS, PADRAO, ajustar_modelo, ajustar_serie, exp_esticada_c0, exp_simples_c0, potencia_c0

T = np.arange(0, 57, 1 / 7.5)


def test_esticada_recupera_tau_e_beta():
    y = exp_esticada_c0(T, 100, 10, 0.6) + np.random.default_rng(0).normal(0, 0.3, len(T))
    a = ajustar_modelo(MODELOS["exp_esticada_c0"], T, y, PADRAO)
    assert a.ok and abs(a.params["tau"] - 10) < 1 and abs(a.params["beta"] - 0.6) < 0.05


def test_esticada_com_beta_um_e_a_simples():
    y = exp_simples_c0(T, 50, 12) + np.random.default_rng(1).normal(0, 0.2, len(T))
    a = ajustar_modelo(MODELOS["exp_esticada_c0"], T, y, PADRAO)
    assert a.params["beta"] > 0.95 and abs(a.params["tau"] - 12) < 1


def test_potencia_recupera_t0_e_alpha():
    y = potencia_c0(T, 100, 5, 1.5) + np.random.default_rng(2).normal(0, 0.3, len(T))
    a = ajustar_modelo(MODELOS["potencia_c0"], T, y, PADRAO)
    assert a.ok and abs(a.params["t0"] - 5) < 0.5 and abs(a.params["alpha"] - 1.5) < 0.1


def test_modelos_extras_nao_mudam_o_ajuste_padrao():
    y = exp_simples_c0(T, 30, 12) + np.random.default_rng(3).normal(0, 0.3, len(T))
    r = ajustar_serie(T, y, ("exp_simples_c0", "exp_dupla_c0"))
    assert r["modelo"] == "exp_simples_c0" and abs(r["tau"] - 12) < 1


def test_ajustar_serie_aceita_modelo_sem_tau():
    y = potencia_c0(T, 100, 5, 1.5)
    r = ajustar_serie(T, y, ("potencia_c0",), r2_minimo=0.9)
    assert r["modelo"] == "potencia_c0" and np.isnan(r["tau_medio"])
