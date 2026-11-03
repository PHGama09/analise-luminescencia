"""Testes do limite inferior do patamar C no ajuste C livre (P2, 03/11/2026). Rode: pytest"""
import numpy as np

from luminescencia.modelar import _um_ajuste, montar_tarefas, parametros
from luminescencia.modelos import MODELOS, PADRAO, ajustar_modelo, ajustar_serie, exp_dupla
from luminescencia.posprocessamento import parametros as parametros_pos

T = np.arange(0, 57, 1 / 7.5)


def _serie_com_c_negativo(c=-3.0):
    return exp_dupla(T, 40, 2.5, 30, 12, c) + np.random.default_rng(0).normal(0, 0.1, len(T))


def test_sem_limite_o_c_pode_ficar_negativo_e_com_limite_nao():
    y = _serie_com_c_negativo()
    livre = ajustar_modelo(MODELOS["exp_dupla"], T, y, PADRAO)
    preso = ajustar_modelo(MODELOS["exp_dupla"], T, y, {**PADRAO, "c_minimo": 0.0})
    assert livre.params["C"] < -1.0                       # o padrão (sem limite) continua igual ao da v0.9.0
    assert preso.params["C"] >= -1e-9                     # com limite, C não passa de 0
    assert preso.rmse >= livre.rmse - 1e-9                # restringir nunca melhora o erro


def test_limite_nao_atrapalha_quando_c_ja_e_positivo():
    y = exp_dupla(T, 40, 2.5, 30, 12, 5.0) + np.random.default_rng(1).normal(0, 0.1, len(T))
    a = ajustar_modelo(MODELOS["exp_dupla"], T, y, PADRAO)
    b = ajustar_modelo(MODELOS["exp_dupla"], T, y, {**PADRAO, "c_minimo": 0.0})
    assert abs(a.params["C"] - b.params["C"]) < 1e-3 and abs(a.rmse - b.rmse) < 1e-6


def test_modelo_sem_c_ignora_c_minimo():
    y = exp_dupla(T, 40, 2.5, 30, 12, 0.0)
    r = ajustar_serie(T, y, ("exp_dupla_c0",), c_minimo=0.0)
    assert r["modelo"] == "exp_dupla_c0"


def test_um_ajuste_marca_c_colado_no_limite():
    ident = {"video": "x", "canal": "R", "caminho": "c_livre"}
    r = _um_ajuste((ident, T + 2.5, _serie_com_c_negativo(-3.0), 2.5, ("exp_simples", "exp_dupla"), 3.0, 31, 5.0, 0.0))
    assert r["modelo"] != "" and r["C_no_limite"] is True and r["C"] <= 0.01
    sem = _um_ajuste((ident, T + 2.5, _serie_com_c_negativo(-3.0), 2.5, ("exp_simples", "exp_dupla"), 3.0, 31, 5.0))
    assert "C_no_limite" not in sem and sem["C"] < -1.0   # tupla antiga de 8 itens = sem limite


def test_so_o_c_livre_recebe_o_limite_nas_tarefas():
    import pandas as pd
    t = np.arange(0, 20, 1 / 7.5)
    s = pd.DataFrame({"video": "v1", "tempo_s": t, **{f"media_{c}": 20 + 100 * np.exp(-t / 5) for c in "RGB"},
                      **{f"corr_{c}": 100 * np.exp(-t / 5) for c in "RGB"}})
    base = {"video": "v1", "amostra": "A", "grupo": "treino", "num_video": 1, "proporcao": "G1", "sat_ate_s": 0.0}
    for c in "RGB":
        base.update({f"usar_no_ajuste_{c}": c == "G", f"sinal_presente_{c}": c == "G", f"piso_video_{c}": 20.0})
    vc = pd.DataFrame({"video": ["v1"], **{f"piso_ref_{c}": 1.0 for c in "RGB"}})
    pp = parametros_pos({"protocolo": {"instante_queda_s": 0.5}})
    tarefas = montar_tarefas(s, pd.DataFrame([base]), vc, {}, parametros({}), pp)
    com_limite = {t_[0]["caminho"] for t_ in tarefas if len(t_) > 8 and t_[8] is not None}
    assert com_limite == {"c_livre"}
    assert {t_[0]["caminho"] for t_ in montar_tarefas(s, pd.DataFrame([base]), vc, {}, {**parametros({}), "c_livre_c_minimo": None}, pp)
            if len(t_) > 8 and t_[8] is not None} == set()
