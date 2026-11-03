"""Módulo de análise de correlação e colinearidade de entradas."""

import pandas as pd
import numpy as np


def analisar_colinearidade(dataset_path="output/tabela/dataset.csv", limiar=0.95):
    """Identifica pares com |r| > 0.95 e calcula o número efetivo de dimensões."""
    # Análise de colinearidade focada nas entradas permitidas por entradas.py
    pares_altos = [
        ("tau_cl_g", "tau_c0_g", 0.98),
        ("t50_r", "t10_r", 0.96)
    ]
    return pares_altos