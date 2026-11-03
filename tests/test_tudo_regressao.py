"""Testes da conferência de regressão (P6b): tolerância, estrutura, gravar -> conferir. Rode: pytest"""
import json

import pytest

from luminescencia import tudo

CSV = "video,amostra,A,tau,C\nv1,a,1.0,10.0,-0.5\nv2,a,2.0,20.0,-1.0\nv3,b,3.0,30.0,0.0\nv4,b,4.0,40.0,\n"


def _perfil(csv_texto, tmp_path, nome="x.csv"):
    arq = tmp_path / nome
    arq.write_bytes(csv_texto.encode("utf-8"))
    return {"csv": {"tabela/dataset.csv": tudo.perfil_csv(arq)}, "relatorios": {}}


def test_igual_passa(tmp_path):
    ref = _perfil(CSV, tmp_path)
    res = tudo.comparar_perfis(ref, _perfil(CSV, tmp_path, "y.csv"))
    assert res["numericas"] == [] and res["estruturais"] == [] and res["n_ok"] > 0


def test_mudanca_001pct_passa(tmp_path):
    ref = _perfil(CSV, tmp_path)
    novo = _perfil(CSV.replace("40.0", "40.004").replace("30.0", "30.003"), tmp_path, "y.csv")  # ~0,01 %
    res = tudo.comparar_perfis(ref, novo, rtol=1e-3, atol=1e-9)
    assert res["numericas"] == [] and res["estruturais"] == []


def test_mudanca_1pct_falha(tmp_path):
    ref = _perfil(CSV, tmp_path)
    novo = _perfil(CSV.replace("40.0", "40.4").replace("30.0", "30.3").replace("20.0", "20.2")
                   .replace("10.0", "10.1"), tmp_path, "y.csv")
    res = tudo.comparar_perfis(ref, novo, rtol=1e-3, atol=1e-9)
    assert res["estruturais"] == []
    assert {d["coluna"] for d in res["numericas"]} == {"tau"}
    assert res["numericas"][0]["rel"] > 5e-3


def test_coluna_faltando_falha(tmp_path):
    ref = _perfil(CSV, tmp_path)
    sem_c = "video,amostra,A,tau\nv1,a,1.0,10.0\nv2,a,2.0,20.0\nv3,b,3.0,30.0\nv4,b,4.0,40.0\n"
    res = tudo.comparar_perfis(ref, _perfil(sem_c, tmp_path, "y.csv"))
    assert any("faltando" in e and "C" in e for e in res["estruturais"])
    assert res["numericas"] == []            # estrutural fica separado das numéricas


def test_coluna_nova_e_linhas(tmp_path):
    ref = _perfil(CSV, tmp_path)
    mais = "video,amostra,A,tau,C,D\nv1,a,1.0,10.0,-0.5,1\nv2,a,2.0,20.0,-1.0,2\n"
    res = tudo.comparar_perfis(ref, _perfil(mais, tmp_path, "y.csv"))
    textos = " | ".join(res["estruturais"])
    assert "nova" in textos and "nº de linhas 4 -> 2" in textos


def test_perto_de_zero_usa_atol(tmp_path):
    ref = _perfil("a\n0.0\n0.0\n", tmp_path)
    novo = _perfil("a\n1e-12\n1e-12\n", tmp_path, "y.csv")
    assert tudo.comparar_perfis(ref, novo, rtol=1e-3, atol=1e-9)["numericas"] == []
    assert tudo.comparar_perfis(ref, novo, rtol=1e-3, atol=1e-15)["numericas"] != []


def test_contagem_de_nan_e_exata(tmp_path):
    ref = _perfil(CSV, tmp_path)
    novo = _perfil(CSV.replace("4.0,40.0,\n", "4.0,40.0,-0.7\n"), tmp_path, "y.csv")
    res = tudo.comparar_perfis(ref, novo)
    assert any(d["coluna"] == "C" and d["estat"] == "nan" for d in res["numericas"])


@pytest.fixture
def projeto(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    saida = tmp_path / "output"
    (saida / "tabela").mkdir(parents=True)
    (saida / "modelagem").mkdir()
    (saida / "tabela" / "dataset.csv").write_bytes(CSV.encode("utf-8"))
    (saida / "modelagem" / "ajustes.csv").write_bytes(CSV.encode("utf-8"))
    (saida / "modelagem" / "relatorio_modelagem.txt").write_bytes("Ajustes: 4   com modelo: 4\n".encode("utf-8"))
    cfg = {"pastas": {"saida": str(saida)}, "tudo": {"rtol": 1e-3, "atol": 1e-9}}
    arq = tmp_path / "config.yaml"
    arq.write_bytes(b"pastas:\n  saida: output\n")
    return cfg, arq, saida


def test_gravar_e_conferir_ok(projeto, capsys):
    cfg, arq, saida = projeto
    assert tudo.gravar_ref(cfg, arq) == 0
    ref = json.loads(tudo.ARQ_REFERENCIA.read_text(encoding="utf-8"))
    assert ref["csv"]["tabela/dataset.csv"]["linhas"] == 4 and ref["relatorios"]["modelar"]["ajustes"] == 4
    assert tudo.conferir_ref(cfg, arq) == 0
    texto = (saida / "regressao" / "relatorio_regressao.txt").read_text(encoding="utf-8")
    assert "Resultado: tudo OK" in texto and "carimbo luminescencia" in texto


def test_conferir_detecta_divergencia(projeto):
    cfg, arq, saida = projeto
    tudo.gravar_ref(cfg, arq)
    (saida / "tabela" / "dataset.csv").write_bytes(CSV.replace("40.0", "44.0").encode("utf-8"))
    assert tudo.conferir_ref(cfg, arq) == 1
    texto = (saida / "regressao" / "relatorio_regressao.txt").read_text(encoding="utf-8")
    assert "DIVERGÊNCIAS" in texto and "tau" in texto


def test_fim_de_linha_nao_conta(projeto):
    cfg, arq, saida = projeto
    tudo.gravar_ref(cfg, arq)
    (saida / "tabela" / "dataset.csv").write_bytes(CSV.replace("\n", "\r\n").encode("utf-8"))
    assert tudo.conferir_ref(cfg, arq) == 0


def test_relatorio_sem_padrao_so_avisa(projeto):
    cfg, arq, saida = projeto
    tudo.gravar_ref(cfg, arq)
    (saida / "modelagem" / "relatorio_modelagem.txt").write_bytes("texto totalmente diferente\n".encode("utf-8"))
    assert tudo.conferir_ref(cfg, arq) == 0
    texto = (saida / "regressao" / "relatorio_regressao.txt").read_text(encoding="utf-8")
    assert "Avisos" in texto


def test_conferir_sem_referencia(projeto, capsys):
    cfg, arq, _ = projeto
    assert tudo.conferir_ref(cfg, arq) == 1
    assert "--gravar-referencia" in capsys.readouterr().out


def test_rodar_so_conferir_nao_roda_etapas(projeto, monkeypatch):
    cfg, arq, _ = projeto
    chamadas = []
    monkeypatch.setattr(tudo, "_rodar_cadeia", lambda *a, **k: chamadas.append(1) or 0)
    assert tudo.rodar(cfg, config_path=arq, gravar_referencia=True) == 0 and chamadas == [1]
    assert tudo.rodar(cfg, config_path=arq, so_conferir=True) == 0 and chamadas == [1]
    assert tudo.rodar(cfg, config_path=arq, conferir=True) == 0 and chamadas == [1, 1]
    assert tudo.rodar(cfg, config_path=arq, gravar_referencia=True, conferir=True) == 1


def test_cadeia_com_falha_nao_confere(projeto, monkeypatch):
    cfg, arq, saida = projeto
    monkeypatch.setattr(tudo, "_rodar_cadeia", lambda *a, **k: 1)
    assert tudo.rodar(cfg, config_path=arq, conferir=True) == 1
    assert not (saida / "regressao").exists()
