"""Testes de preregistro.py e da trava do C (avaliacao.abrir_c) com tabelas sintéticas. Nada aqui calcula em C: a trava só entrega linhas."""
import copy
import hashlib
import json
from datetime import datetime

import pandas as pd
import pytest

from luminescencia import avaliacao as av
from luminescencia import preregistro as pr


def _dataset():
    linhas = []
    for prop, (r, g, b) in {"R1.5": (1.5, 0, 0), "G1.5": (0, 1.5, 0), "R0.75_G0.75": (0.75, 0.75, 0)}.items():
        for am in "ABC":
            for k in range(2):
                linhas.append({"video": f"{prop}_{am}_{k}", "amostra": am, "proporcao": prop, "prop_R": r, "prop_G": g, "prop_B": b,
                               "excluido_saturacao": False})
    df = pd.DataFrame(linhas)
    df.loc[df.video == "G1.5_A_1", "excluido_saturacao"] = True
    return df


def _montar(tmp_path):
    """Cria tabela + config em disco; devolve (cfg, caminho do config.yaml)."""
    pasta = tmp_path / "output" / "tabela"
    pasta.mkdir(parents=True)
    _dataset().to_csv(pasta / "dataset.csv", index=False, encoding="utf-8-sig")
    (pasta / "descritores.csv").write_text("video,R_md_I0\na,1\n", encoding="utf-8")
    (pasta / "dicionario.csv").write_text("coluna,papel\nx,feature\n", encoding="utf-8")
    (pasta / "dicionario_descritores.csv").write_text("coluna,papel\nR_md_I0,feature\n", encoding="utf-8")
    cfg = {
        "pastas": {"saida": str(tmp_path / "output")},
        "avaliacao": {"semente": 1},
        "graficos": {"dpi": 100},                                    # seção SEM relação com o teste (fora de secoes_config)
        "preregistro": {"arquivo_json": str(tmp_path / "docs" / "preregistro.json"), "arquivo_md": str(tmp_path / "docs" / "preregistro.md"),
                        "secoes_config": ["avaliacao", "preregistro"],
                        "declaracoes": {"variante_primaria": {"id": "teste"}, "criterio_d3": {"nivel_minimo_razao_max": 0.7}}},
    }
    caminho = tmp_path / "config.yaml"
    caminho.write_text("avaliacao:\n  semente: 1\n", encoding="utf-8")
    return cfg, caminho


def test_sha256_arquivo_bate_com_hashlib(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc" * 100000)
    assert pr.sha256_arquivo(f) == hashlib.sha256(b"abc" * 100000).hexdigest()


def test_secao_hash_independe_da_ordem_das_chaves_e_muda_com_valor():
    assert pr.sha256_secao({"a": 1, "b": [1, 2]}) == pr.sha256_secao({"b": [1, 2], "a": 1})
    assert pr.sha256_secao({"a": 1}) != pr.sha256_secao({"a": 2})


def test_preregistrar_grava_json_e_md_com_hashes_e_treino(tmp_path):
    cfg, caminho = _montar(tmp_path)
    assert pr.rodar(cfg, config_path=caminho, agora=datetime(2026, 11, 4, 10, 30)) == 0
    reg = json.loads((tmp_path / "docs" / "preregistro.json").read_text(encoding="utf-8"))
    pasta = tmp_path / "output" / "tabela"
    for nome, h in reg["hashes_tabela"].items():
        assert h == hashlib.sha256((pasta / nome).read_bytes()).hexdigest()
    assert set(reg["hashes_tabela"]) == {"dataset.csv", "descritores.csv", "dicionario.csv", "dicionario_descritores.csv"}
    assert reg["data"] == "2026-11-04T10:30:00" and reg["versao_codigo"]
    assert reg["config"]["sha256"] == hashlib.sha256(caminho.read_bytes()).hexdigest() and len(reg["config"]["sha256_16"]) == 16
    assert set(reg["config"]["secoes_sha256"]) == {"avaliacao", "preregistro"}
    assert reg["declaracoes"] == cfg["preregistro"]["declaracoes"]
    # treino A+B sem saturados: 3 proporções x 2 amostras x 2 vídeos = 12, menos 1 saturado; 6 preparações
    assert reg["treino"]["n_videos"] == 11 and reg["treino"]["n_preparacoes"] == 6
    assert {(p["proporcao"], p["amostra"]) for p in reg["treino"]["preparacoes"]} == {(a, b) for a in ("R1.5", "G1.5", "R0.75_G0.75") for b in "AB"}
    assert next(p for p in reg["treino"]["preparacoes"] if (p["proporcao"], p["amostra"]) == ("G1.5", "A"))["n_videos"] == 1
    md = (tmp_path / "docs" / "preregistro.md").read_text(encoding="utf-8")
    assert "Mudar qualquer item depois de abrir C invalida o teste" in md and all(h in md for h in reg["hashes_tabela"].values())


def test_arquivos_gravados_com_lf(tmp_path):
    cfg, caminho = _montar(tmp_path)
    pr.rodar(cfg, config_path=caminho)
    for nome in ("preregistro.json", "preregistro.md"):
        assert b"\r" not in (tmp_path / "docs" / nome).read_bytes()


def test_recusa_sobrescrever_sem_forcar_e_guarda_antigo_com_forcar(tmp_path):
    cfg, caminho = _montar(tmp_path)
    assert pr.rodar(cfg, config_path=caminho, agora=datetime(2026, 11, 4, 10, 30)) == 0
    alvo = tmp_path / "docs" / "preregistro.json"
    antes = alvo.read_bytes()
    assert pr.rodar(cfg, config_path=caminho, agora=datetime(2026, 11, 5, 9, 0)) == 1          # recusa
    assert alvo.read_bytes() == antes and not list((tmp_path / "docs").glob("*anterior*"))
    assert pr.rodar(cfg, forcar=True, config_path=caminho, agora=datetime(2026, 11, 5, 9, 0)) == 0
    antigo = tmp_path / "docs" / "preregistro_anterior_20261104-103000.json"
    assert antigo.is_file() and antigo.read_bytes() == antes                                   # o antigo ficou, com a data dele
    assert (tmp_path / "docs" / "preregistro_anterior_20261104-103000.md").is_file()
    assert json.loads(alvo.read_text(encoding="utf-8"))["data"] == "2026-11-05T09:00:00"


# ------------------------------------------------------------------------------------------------------- a trava

def test_trava_recusa_sem_preregistro(tmp_path):
    cfg, _ = _montar(tmp_path)
    with pytest.raises(RuntimeError, match="não há pré-registro"):
        av.abrir_c(cfg)


def test_trava_abre_com_preregistro_valido_e_so_devolve_linhas_de_c(tmp_path):
    cfg, caminho = _montar(tmp_path)
    pr.rodar(cfg, config_path=caminho)
    c = av.abrir_c(cfg)
    assert set(c.amostra) == {"C"} and len(c) == 6


def test_trava_recusa_com_dataset_alterado(tmp_path):
    cfg, caminho = _montar(tmp_path)
    pr.rodar(cfg, config_path=caminho)
    arq = tmp_path / "output" / "tabela" / "dataset.csv"
    arq.write_bytes(arq.read_bytes() + b"\n")                                  # 1 byte muda o hash
    with pytest.raises(RuntimeError, match="dataset.csv: hash diferente"):
        av.abrir_c(cfg)


def test_trava_recusa_com_descritores_ou_dicionario_alterado(tmp_path):
    for nome in ("descritores.csv", "dicionario.csv", "dicionario_descritores.csv"):
        d = tmp_path / nome.replace(".csv", "")
        d.mkdir()
        cfg, caminho = _montar(d)
        pr.rodar(cfg, config_path=caminho)
        (d / "output" / "tabela" / nome).write_text("outra coisa\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match=f"{nome}: hash diferente"):
            av.abrir_c(cfg)


def test_trava_recusa_com_secao_relevante_do_config_alterada(tmp_path):
    cfg, caminho = _montar(tmp_path)
    pr.rodar(cfg, config_path=caminho)
    mudou = copy.deepcopy(cfg)
    mudou["avaliacao"]["semente"] = 2
    with pytest.raises(RuntimeError, match="seção 'avaliacao'"):
        av.abrir_c(mudou)
    mudou = copy.deepcopy(cfg)
    mudou["preregistro"]["declaracoes"]["criterio_d3"]["nivel_minimo_razao_max"] = 0.9        # mexer no critério depois de registrar
    with pytest.raises(RuntimeError, match="seção 'preregistro'"):
        av.abrir_c(mudou)


def test_trava_nao_e_burlada_tirando_secao_da_lista_do_config(tmp_path):
    cfg, caminho = _montar(tmp_path)
    pr.rodar(cfg, config_path=caminho)
    mudou = copy.deepcopy(cfg)
    mudou["avaliacao"]["semente"] = 2
    mudou["preregistro"]["secoes_config"] = ["preregistro"]                    # tenta "esquecer" a seção alterada
    with pytest.raises(RuntimeError, match="travada"):
        av.abrir_c(mudou)                                                       # a lista valendo é a do JSON, não a do config atual


def test_secao_fora_da_lista_pode_mudar_sem_travar(tmp_path):
    cfg, caminho = _montar(tmp_path)
    pr.rodar(cfg, config_path=caminho)
    cfg["graficos"]["dpi"] = 300
    assert len(av.abrir_c(cfg)) == 6


def test_trava_recusa_json_corrompido_ou_sem_formato(tmp_path):
    cfg, caminho = _montar(tmp_path)
    pr.rodar(cfg, config_path=caminho)
    arq = tmp_path / "docs" / "preregistro.json"
    original = arq.read_text(encoding="utf-8")
    arq.write_text("{não é json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="JSON legível"):
        av.abrir_c(cfg)
    reg = json.loads(original)
    reg["hashes_tabela"]["dataset.csv"] = "123"
    arq.write_text(json.dumps(reg), encoding="utf-8")
    with pytest.raises(RuntimeError, match="formato"):
        av.abrir_c(cfg)
    arq.write_text(json.dumps({"formato": 1}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="formato"):
        av.abrir_c(cfg)


def test_secao_listada_mas_ausente_do_config_recusa_preregistrar(tmp_path):
    cfg, caminho = _montar(tmp_path)
    cfg["preregistro"]["secoes_config"] = ["avaliacao", "nao_existe"]
    with pytest.raises(ValueError, match="ausentes"):
        pr.rodar(cfg, config_path=caminho)
    assert not (tmp_path / "docs" / "preregistro.json").exists()
