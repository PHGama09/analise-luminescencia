"""Testes da comparação de modelos: descritor t_meio e escolha do vencedor. Rode: pytest"""
import numpy as np
import pandas as pd

from luminescencia.comparacao_modelos import CONJUNTOS, _tarefa, resumir, t_meio
from luminescencia.modelos import exp_esticada_c0, exp_simples_c0

T = np.arange(0, 57, 1 / 7.5) + 2.5


def test_t_meio_da_exponencial_simples():
    # cai à metade em tau*ln2
    assert abs(t_meio("exp_simples_c0", {"A": 10, "tau": 10}, 50, False) - 10 * np.log(2)) < 0.05


def test_t_meio_com_patamar_usa_a_metade_acima_de_C():
    assert abs(t_meio("exp_simples", {"A": 10, "tau": 10, "C": 5}, 50, True) - 10 * np.log(2)) < 0.05


def test_tarefa_ajusta_todos_os_modelos_dos_dois_conjuntos():
    y = exp_esticada_c0(T - T[0], 100, 10, 0.6) + np.random.default_rng(0).normal(0, 0.2, len(T))
    linhas = _tarefa(({"video": "v", "amostra": "A", "grupo": "treino", "num_video": 1, "proporcao": "G1", "canal": "G", "canal_ativo": True},
                      T, y, 2.5, 3.0, 31, 5.0))
    assert len(linhas) == sum(len(v) for v in CONJUNTOS.values())
    df = pd.DataFrame(linhas)
    c0 = df[df["conjunto"] == "c0"].set_index("modelo")
    assert c0.loc["exp_esticada_c0", "bic"] < c0.loc["exp_simples_c0", "bic"]      # a forma verdadeira vence a simples


def test_relatorio_conta_vitorias():
    y = exp_simples_c0(T - T[0], 100, 8) + np.random.default_rng(1).normal(0, 0.2, len(T))
    linhas = _tarefa(({"video": "v", "amostra": "A", "grupo": "treino", "num_video": 1, "proporcao": "G1", "canal": "G", "canal_ativo": True},
                      T, y, 2.5, 3.0, 31, 5.0))
    df = pd.DataFrame(linhas); df["motivo"] = df["motivo"].fillna("")
    texto = resumir(df)
    assert "CONJUNTO C0" in texto and "exp_esticada_c0" in texto and "potencia_c0" in texto
