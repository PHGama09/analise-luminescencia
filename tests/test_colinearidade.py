"""Testes para o módulo de colinearidade."""
from luminescencia import colinearidade

def test_colinearidade_basico():
    pares = colinearidade.analisar_colinearidade()
    assert isinstance(pares, list)