"""Testes do leitor de nomes. Rode: pytest"""
from luminescencia.auditoria import _rotulo_da_pasta
from luminescencia.nomes import ler_nome

CFG = {"controle": "referencia", "soma_esperada": 1.5, "soma_tolerancia": 0.01}


def _ler(nome: str, cfg: dict = CFG) -> dict:
    """Lê um nome que deve ser válido (o assert também tranquiliza o verificador de tipos do editor)."""
    campos, erro = ler_nome(nome, cfg)
    assert campos is not None, erro
    return campos


def test_nome_correto_nao_tem_correcao():
    c = _ler("GB_1_05_v1a.avi")
    assert c["proporcao"] == "G1_B0.5" and c["amostra"] == "A" and c["video"] == 1
    assert "correcoes" not in c and "aviso_caixa" not in c


def test_erros_de_digitacao_sao_lidos_com_nome_sugerido():
    esperado = {
        "GB _05_1_v2a.avi": "GB_05_1_v2a.avi",
        "RGB_1_025__025_v1b.avi": "RGB_1_025_025_v1b.avi",
        "RGB_05_05_05v3c.avi": "RGB_05_05_05_v3c.avi",
    }
    for nome, certo in esperado.items():
        c = _ler(nome)
        assert c["nome_sugerido"] == certo and c["correcoes"], nome


def test_so_maiuscula_vira_aviso_e_nao_revisao():
    c = _ler("B_15_v4A.avi")
    assert c["amostra"] == "A" and "aviso_caixa" in c
    assert "correcoes" not in c and "nome_sugerido" not in c


def test_so_maiuscula_pode_voltar_a_ser_erro():
    c = _ler("B_15_v4A.avi", {**CFG, "maiusculas_sao_erro": True})
    assert c["nome_sugerido"] == "B_15_v4a.avi" and "aviso_caixa" not in c


def test_nome_invalido_e_controle():
    c, erro = ler_nome("qualquer_coisa.avi", CFG)
    assert c is None and erro
    assert _ler("Amostra_referencia.avi") == {"tipo": "controle"}


def test_soma_diferente_gera_aviso():
    assert "soma" in _ler("GB_1_1_v1a.avi")["aviso"]


def test_rotulo_da_pasta():
    assert _rotulo_da_pasta("Amostras C/4/B_15_v4c.avi") == ("C", 4)
    assert _rotulo_da_pasta("Amostra_referencia.avi") == (None, None)