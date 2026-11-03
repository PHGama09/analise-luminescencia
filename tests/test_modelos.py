"""Testes do ajuste de decaimento, com dados sintéticos. Rode: pytest  (ou python tests/test_modelos.py)"""
import numpy as np

from luminescencia.modelos import ajustar_serie, exp_dupla, exp_simples


def _t():
    return np.arange(0, 20, 1 / 11)


def test_exponencial_simples_e_escolhida():
    rng = np.random.default_rng(1)
    y = exp_simples(_t(), 200, 3.0, 5) + rng.normal(0, 0.8, len(_t()))
    r = ajustar_serie(_t(), y)
    assert r["modelo"] == "exp_simples"
    assert abs(r["tau"] - 3.0) < 0.3


def test_dupla_quando_simples_nao_serve():
    rng = np.random.default_rng(2)
    y = exp_dupla(_t(), 150, 0.8, 100, 6.0, 5) + rng.normal(0, 0.8, len(_t()))
    r = ajustar_serie(_t(), y)
    assert r["modelo"] == "exp_dupla"
    assert abs(r["tau1"] - 0.8) < 0.2 and abs(r["tau2"] - 6.0) < 1.0


def test_poucos_pontos():
    r = ajustar_serie([0, 1, 2], [3, 2, 1])
    assert r["modelo"] == "" and "poucos pontos" in r["observacao"]


if __name__ == "__main__":
    test_exponencial_simples_e_escolhida()
    test_dupla_quando_simples_nao_serve()
    test_poucos_pontos()
    print("ok")