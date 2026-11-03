"""
Testes de integração para a Fase 2 (C15)
"""
import pytest
import pandas as pd
import numpy as np

def test_integracao_fluxo(tmp_path):
    # Teste sintético rápido
    df = pd.DataFrame({'t': [0, 1, 2], 'R': [100, 50, 25], 'G': [80, 40, 20], 'B': [60, 30, 15]})
    file_path = tmp_path / "B_15_v1a.csv"
    df.to_csv(file_path, index=False)
    assert file_path.exists()
