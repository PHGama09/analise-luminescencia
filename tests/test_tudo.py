"""Testes do comando `tudo`: carimbo idempotente, ordem da cadeia, --de/--ate e parada no erro. Rode: pytest"""
import hashlib

import pytest

from luminescencia import __version__, cli, tudo

ORDEM = ["consolidar", "pos", "modelar", "tabela", "comparar_modelos", "descritores", "luz", "graficos"]


@pytest.fixture
def ambiente(tmp_path, monkeypatch):
    """Config falso em disco, pasta de saída em tmp e `rodar` falsos que registram a chamada e escrevem o relatório."""
    monkeypatch.chdir(tmp_path)
    cfg_arq = tmp_path / "config.yaml"
    cfg_arq.write_bytes(b"pastas:\n  saida: output\n")
    cfg = {"pastas": {"saida": str(tmp_path / "output")}}
    chamadas, falhas = [], {}

    def obter(nome):
        def falso(c, **kw):
            chamadas.append(nome)
            assert kw == {}                            # só cfg: padrões do config
            if nome in falhas:
                if falhas[nome] == "excecao":
                    raise FileNotFoundError("falta a pasta anterior")
                return falhas[nome]
            rel = tudo._caminho_relatorio(c, nome)
            rel.parent.mkdir(parents=True, exist_ok=True)
            rel.write_text(f"relatório de {nome}\n", encoding="utf-8")
            return 0
        return falso

    monkeypatch.setattr(tudo, "_obter_rodar", obter)
    return cfg, str(cfg_arq), chamadas, falhas


def test_carimbo_idempotente(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_bytes(b"a: 1\n")
    rel = tmp_path / "relatorio.txt"
    rel.write_text("linha 1\nlinha 2\n", encoding="utf-8")
    assert tudo.carimbar(rel, cfg) is True
    primeiro = rel.read_bytes()
    assert tudo.carimbar(rel, cfg) is True and tudo.carimbar(rel, cfg) is True
    texto = rel.read_text(encoding="utf-8")
    assert texto.count(tudo.MARCA_INI) == 1 and texto.count(tudo.MARCA_FIM) == 1
    assert texto.startswith("linha 1\nlinha 2\n\n# --- carimbo luminescencia ---\n") and texto.endswith("# --- fim carimbo ---\n")
    assert rel.read_bytes()[:len("linha 1\nlinha 2\n")] == primeiro[:len("linha 1\nlinha 2\n")]
    assert b"\r" not in rel.read_bytes()


def test_carimbo_conteudo_e_substituicao(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_bytes(b"a: 1\n")
    rel = tmp_path / "relatorio.txt"
    rel.write_text("corpo\n", encoding="utf-8")
    tudo.carimbar(rel, cfg)
    esperado = hashlib.sha256(b"a: 1\n").hexdigest()[:16]
    texto = rel.read_text(encoding="utf-8")
    assert f"versao_codigo: {__version__}" in texto and f"config_sha256: {esperado}" in texto and "data: " in texto
    cfg.write_bytes(b"a: 2\n")                         # outro config: o hash do bloco é substituído, não acrescentado
    tudo.carimbar(rel, cfg)
    texto = rel.read_text(encoding="utf-8")
    assert esperado not in texto and texto.count("config_sha256:") == 1
    assert tudo.carimbar(tmp_path / "nao_existe.txt", cfg) is False


def test_cadeia_na_ordem_padrao(ambiente):
    cfg, cfg_arq, chamadas, _ = ambiente
    assert tudo.rodar(cfg, config_path=cfg_arq) == 0
    assert chamadas == ORDEM
    rel = tudo._caminho_relatorio(cfg, "graficos").read_text(encoding="utf-8")
    assert rel.count(tudo.MARCA_INI) == 1                # cada relatório recebeu o carimbo
    tudo.rodar(cfg, config_path=cfg_arq)                 # rodar de novo não duplica
    assert tudo._caminho_relatorio(cfg, "graficos").read_text(encoding="utf-8").count(tudo.MARCA_INI) == 1


def test_com_offset_depois_de_descritores(ambiente):
    cfg, cfg_arq, chamadas, _ = ambiente
    assert tudo.rodar(cfg, com_offset=True, config_path=cfg_arq) == 0
    assert chamadas == ["consolidar", "pos", "modelar", "tabela", "comparar_modelos", "descritores", "offset", "luz", "graficos"]


def test_de_ate(ambiente):
    cfg, cfg_arq, chamadas, _ = ambiente
    assert tudo.rodar(cfg, de="tabela", ate="descritores", config_path=cfg_arq) == 0
    assert chamadas == ["tabela", "comparar_modelos", "descritores"]
    chamadas.clear()
    assert tudo.rodar(cfg, de="descritores", ate="descritores", config_path=cfg_arq) == 0
    assert chamadas == ["descritores"]
    chamadas.clear()
    assert tudo.rodar(cfg, de="graficos", ate="pos", config_path=cfg_arq) == 1 and chamadas == []   # de depois de ate
    assert tudo.rodar(cfg, de="auditar", config_path=cfg_arq) == 1 and chamadas == []               # fora da cadeia


def test_para_no_erro(ambiente, capsys):
    cfg, cfg_arq, chamadas, falhas = ambiente
    falhas["modelar"] = 2
    assert tudo.rodar(cfg, config_path=cfg_arq) == 1
    assert chamadas == ["consolidar", "pos", "modelar"]
    saida = capsys.readouterr().out
    assert "FALHOU (código 2)" in saida and "FALHA em modelar" in saida and "tabela" in saida and "não executada" in saida
    assert not tudo._caminho_relatorio(cfg, "modelar").exists()                                     # etapa falha não é carimbada


def test_continua_sem_parar_no_erro(ambiente):
    cfg, cfg_arq, chamadas, falhas = ambiente
    falhas["pos"] = 1
    assert tudo.rodar(cfg, parar_no_erro=False, config_path=cfg_arq) == 1
    assert chamadas == ORDEM


def test_excecao_conta_como_falha(ambiente, capsys):
    cfg, cfg_arq, chamadas, falhas = ambiente
    falhas["pos"] = "excecao"
    assert tudo.rodar(cfg, config_path=cfg_arq) == 1
    assert chamadas == ["consolidar", "pos"]
    assert "FileNotFoundError" in capsys.readouterr().out


def test_cli_registra_tudo(ambiente, monkeypatch):
    _, cfg_arq, _, _ = ambiente
    visto = {}
    monkeypatch.setattr(tudo, "rodar", lambda cfg, **kw: visto.update(kw) or 0)
    assert cli.main(["tudo", "--config", cfg_arq, "--de", "tabela", "--ate", "descritores", "--com-offset", "--no-parar-no-erro"]) == 0
    assert visto["de"] == "tabela" and visto["ate"] == "descritores" and visto["com_offset"] is True
    assert visto["parar_no_erro"] is False and visto["config_path"] == cfg_arq and visto["conferir"] is False
    assert cli.main(["tudo", "--config", cfg_arq]) == 0 and visto["parar_no_erro"] is True
