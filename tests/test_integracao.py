"""Integração (Q4/C15): consolidar -> pos -> modelar -> tabela sobre vídeos SINTÉTICOS, sem dados reais.

O gerador escreve, em tmp_path, um CSV por vídeo no formato exato de B_15_v1a.csv (mesmas 18 colunas, mesma ordem, fase
excitacao/decaimento, quadros repetidos descartados) com decaimento duplo exponencial + ruído, mais o extracao_resumo.csv (o que
a extração entregaria) e o vídeo de controle. Cenário: 2 amostras (A, B) x 2 proporções (B_15, GB_05_1) x 2 regravações + controle.
A verdade (parâmetros, preto de cada sessão) é conhecida, então o teste confere formas, vazamento, τ recuperado e determinismo.
Nada aqui altera código de produção. Rode: pytest tests/test_integracao.py
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from luminescencia import consolidacao, modelar, posprocessamento, tabela
from luminescencia.config import carregar_config

RAIZ_REPO = Path(__file__).resolve().parent.parent
CANAIS = "RGB"
FPS, DURACAO_S, T_QUEDA_S, RUIDO = 15, 60, 0.3333, 0.04

# formato exato do B_15_v1a.csv (extração)
COLUNAS_EXTRACAO = ["frame", "tempo_s", "fase", "media_R", "media_G", "media_B", "mediana_R", "mediana_G", "mediana_B",
                    "desvio_cinza", "saturados_pct", "fundo_R", "fundo_G", "fundo_B", "canto_R", "canto_G", "canto_B", "repeticoes"]

# proporção -> canal -> (A1, tau1, A2, tau2) do decaimento duplo exponencial (níveis de 0 a 255; segundos)
SINAIS = {"B_15": {"B": (60.0, 3.0, 140.0, 20.0)},
          "GB_05_1": {"G": (50.0, 2.5, 100.0, 18.0), "B": (60.0, 3.0, 140.0, 22.0)}}
FATOR_TAU = {1: 1.00, 2: 1.03}                       # regravação 2 com τ 3% maior (variação entre regravações)
PRETO_SESSAO = {"A": 0.0, "B": 0.5}                  # o preto da câmera muda de sessão para sessão (níveis)
PISO_ROI_BASE = {"R": 0.7, "G": 1.0, "B": 1.1}       # preto dentro da ROI
PISO_REF_CONTROLE = {"R": 0.6, "G": 0.9, "B": 1.0}   # piso do controle dentro da ROI
OFFSET_CANTO = 0.9                                   # diferença cantos - ROI (o pipeline a mede no controle)

N_COLUNAS_DATASET, N_COLUNAS_VIDEOS_POS = 106, 39

PROIBIDAS = [r"^prop_[RGB]$", r"^canal_ativo_[RGB]$", r"^usar_no_ajuste_[RGB]$", r"^sat_ate_s$", r"^excluido_saturacao$",
             r"^(video|amostra|grupo|num_video|proporcao)$"]


# ----------------------------------------------------------------------------- gerador sintético
def _serie(verdade: dict, rng: np.random.Generator) -> pd.DataFrame:
    frames = np.arange(0, FPS * DURACAO_S)
    manter = (frames < 30) | (rng.random(len(frames)) < 0.5) | (frames == frames[-1])      # ~metade dos quadros: os repetidos saem
    frames = frames[manter]
    t = np.round(frames / FPS, 4)
    decai = t >= T_QUEDA_S
    d = {"frame": frames, "tempo_s": t, "fase": np.where(decai, "decaimento", "excitacao")}
    for c in CANAIS:
        v = verdade["canais"].get(c)
        sinal = np.zeros(len(t))
        if v:
            A1, t1, A2, t2 = v
            sinal = A1 * np.exp(-(t - T_QUEDA_S) / t1) + A2 * np.exp(-(t - T_QUEDA_S) / t2)
        media = verdade["piso"][c] + sinal + rng.normal(0, RUIDO, len(t))
        d[f"media_{c}"] = np.where(decai, media, 254.0)
    for c in CANAIS:
        d[f"mediana_{c}"] = np.round(d[f"media_{c}"])
    d["desvio_cinza"] = np.where(decai, 10.0 + rng.normal(0, 0.2, len(t)), 0.0)
    d["saturados_pct"] = np.where(decai, 0.0, 100.0)
    for c in CANAIS:
        d[f"fundo_{c}"] = np.where(decai, 1.0 + rng.normal(0, 0.05, len(t)), 254.0)
    for c in CANAIS:
        d[f"canto_{c}"] = np.where(decai, verdade["piso"][c] + OFFSET_CANTO + rng.normal(0, 0.05, len(t)), 254.0)
    d["repeticoes"] = rng.integers(0, 3, len(t))
    return pd.DataFrame(d)[COLUNAS_EXTRACAO]


def _linha_resumo(nome, caminho, csv, controle, serie, piso_canto, piso_ref, ativos) -> dict:
    dec = serie[serie["fase"] == "decaimento"]
    L = {"caminho": caminho, "arquivo_csv": csv, "chave": "sha1:" + hashlib.sha1(caminho.encode()).hexdigest(),
         "controle": controle, "incluir": "pendente", "status_auditoria": "ok", "roi_cx": 320.0, "roi_cy": 240.0, "roi_r": 90.0,
         "roi_origem": "auto", "roi_suspeito": False, "roi_motivos": "", "fps": float(FPS), "n_quadros_novos": len(serie),
         "n_quadros_total": FPS * DURACAO_S, "repetidos_pct": 48.0, "duracao_s": float(DURACAO_S),
         "primeiro_decaimento_s": float(dec["tempo_s"].iloc[0]), "quadros_decaimento": len(dec), "saturacao_ate_s": float(T_QUEDA_S),
         "saturados_pct_inicio": 100.0, "versao_extracao": "sintetico", "extraido_em": "2026-01-01 00:00:00",
         "tempo_s_processamento": 0.0, "piso_fonte": "controle", "preto_difere_da_ref": False}
    for c in CANAIS:
        ruido = (not ativos.get(c, False)) and not controle
        L.update({f"inicio_{c}": float(dec[f"media_{c}"].iloc[0]), f"final_{c}": float(dec[f"media_{c}"].tail(150).mean()),
                  f"piso_canto_{c}": round(piso_canto[c], 4), f"fundo_anel_{c}": 1.0, f"perto_do_ruido_{c}": bool(ruido),
                  f"piso_ref_{c}": PISO_REF_CONTROLE[c], f"perto_do_ruido_canto_{c}": False,
                  f"deriva_preto_{c}": round(piso_canto[c] - PISO_REF_CONTROLE[c], 4)})
    return L


def gerar_extracao(pasta: Path, semente: int = 7) -> dict:
    """Escreve pasta/<video>.csv e pasta/extracao_resumo.csv. Devolve a verdade: {video: {amostra, proporcao, canais, piso, ...}}."""
    pasta.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(semente)
    linhas, verdade = [], {}
    for amostra in ("A", "B"):
        for prop, canais in SINAIS.items():
            for k in (1, 2):
                nome = f"{prop}_v{k}{amostra.lower()}"
                v = {"amostra": amostra, "proporcao": prop, "regravacao": k,
                     "canais": {c: (a1, t1 * FATOR_TAU[k], a2, t2 * FATOR_TAU[k]) for c, (a1, t1, a2, t2) in canais.items()},
                     "piso": {c: PISO_ROI_BASE[c] + PRETO_SESSAO[amostra] for c in CANAIS}}
                serie = _serie(v, rng)
                serie.to_csv(pasta / f"{nome}.csv", index=False, encoding="utf-8-sig")
                canto = {c: v["piso"][c] + OFFSET_CANTO for c in CANAIS}
                linhas.append(_linha_resumo(nome, f"Amostras {amostra}/{k}/{nome}.avi", f"{nome}.csv", False, serie, canto,
                                            PISO_REF_CONTROLE, {c: True for c in canais}))
                verdade[nome] = v
    ctrl = {"canais": {}, "piso": dict(PISO_REF_CONTROLE)}
    sc = _serie(ctrl, rng)
    sc.to_csv(pasta / "Amostra_referencia.csv", index=False, encoding="utf-8-sig")
    canto_c = {c: PISO_REF_CONTROLE[c] + OFFSET_CANTO for c in CANAIS}
    linhas.append(_linha_resumo("Amostra_referencia", "Amostra_referencia.avi", "Amostra_referencia.csv", True, sc, canto_c,
                                PISO_REF_CONTROLE, {}))
    pd.DataFrame(linhas).to_csv(pasta / "extracao_resumo.csv", index=False, encoding="utf-8-sig")
    return verdade


def _cfg(raiz: Path) -> dict:
    cfg = carregar_config(RAIZ_REPO / "config.yaml")
    cfg["pastas"] = {**(cfg.get("pastas") or {}), "saida": str(raiz / "output")}
    cfg["curadoria"] = {**(cfg.get("curadoria") or {}), "arquivo": str(raiz / "curadoria.csv")}     # não existe: tudo "pendente"
    cfg["modelagem"] = {**(cfg.get("modelagem") or {}), "workers": 1}
    return cfg


def rodar_cadeia(raiz: Path) -> dict:
    cfg = _cfg(raiz)
    saida = raiz / "output"
    verdade = gerar_extracao(saida / "extracao")
    assert consolidacao.rodar(cfg) in (0, None)
    assert posprocessamento.rodar(cfg) in (0, None)
    modelar.rodar(cfg)
    tabela.rodar(cfg)
    ler = lambda *p: pd.read_csv(saida.joinpath(*p), encoding="utf-8-sig")        # noqa: E731
    return {"verdade": verdade, "saida": saida, "videos": ler("consolidacao", "videos.csv"), "series": ler("consolidacao", "series.csv"),
            "videos_pos": ler("posprocessamento", "videos_pos.csv"), "series_pos": ler("posprocessamento", "series_pos.csv"),
            "ajustes": ler("modelagem", "ajustes.csv"), "dataset": ler("tabela", "dataset.csv"),
            "dicionario": ler("tabela", "dicionario.csv")}


@pytest.fixture(scope="module")
def cadeia(tmp_path_factory):
    return rodar_cadeia(tmp_path_factory.mktemp("exec1"))


@pytest.fixture(scope="module")
def cadeia_repetida(tmp_path_factory):
    return rodar_cadeia(tmp_path_factory.mktemp("exec2"))


def _tau_medio_esperado(v: dict, canal: str, t0: float) -> float:
    """τ médio (ponderado pela amplitude em t0, como o pipeline) do decaimento conhecido, com a janela começando em t0."""
    A1, t1, A2, t2 = v["canais"][canal]
    a1, a2 = A1 * np.exp(-(t0 - T_QUEDA_S) / t1), A2 * np.exp(-(t0 - T_QUEDA_S) / t2)
    return float((a1 * t1 + a2 * t2) / (a1 + a2))


# ----------------------------------------------------------------------------- testes
def test_gerador_no_formato_do_b15_v1a(cadeia):
    arq = cadeia["saida"] / "extracao" / "B_15_v1a.csv"
    d = pd.read_csv(arq, encoding="utf-8-sig")
    assert list(d.columns) == COLUNAS_EXTRACAO and set(d["fase"]) == {"excitacao", "decaimento"}
    assert d["tempo_s"].is_monotonic_increasing and d["tempo_s"].iloc[-1] == pytest.approx(DURACAO_S - 1 / FPS, abs=0.01)
    assert (d.loc[d["fase"] == "excitacao", "saturados_pct"] == 100).all()


def test_formas_das_tabelas(cadeia):
    assert 2 * len(SINAIS) * len(FATOR_TAU) == 8                                     # 2 amostras x 2 proporções x 2 regravações
    assert len(cadeia["videos"]) == 8 and cadeia["videos"]["video"].is_unique
    assert not cadeia["videos"]["controle"].astype(bool).any()                       # o controle não entra nos vídeos aceitos
    assert set(cadeia["videos"]["video"]) == set(cadeia["verdade"])
    assert set(cadeia["videos"]["grupo"]) == {"treino"}
    esperado_linhas = sum(len(pd.read_csv(cadeia["saida"] / "extracao" / f"{v}.csv")) for v in cadeia["verdade"])
    assert len(cadeia["series"]) == esperado_linhas and {"corr_R", "corr_G", "corr_B"} <= set(cadeia["series"].columns)
    assert len(cadeia["videos_pos"]) == 8 and set(cadeia["series_pos"]["video"]) == set(cadeia["verdade"])
    assert cadeia["series_pos"]["tempo_s"].min() >= min(posprocessamento.parametros(_cfg(Path(".")))["janelas_sensibilidade"]) - 1e-9
    ds = cadeia["dataset"]
    assert len(ds) == 8 and ds["video"].is_unique
    assert set(ds.columns) == set(cadeia["dicionario"]["coluna"])                    # o dicionário cobre todas as colunas
    for c in CANAIS:
        for caminho in ("cl", "c0"):
            assert f"{c}_{caminho}_tau_medio" in ds.columns


def test_ajustes_so_nos_canais_com_sinal(cadeia):
    aj, v = cadeia["ajustes"], cadeia["verdade"]
    pares = {(nome, c) for nome, x in v.items() for c in x["canais"]}
    assert {(a, c) for a, c in zip(aj["video"], aj["canal"])} == pares            # canal sem pó (perto do ruído) não é ajustado
    assert set(aj["caminho"]) == {"principal", "inicio_2s", "inicio_3s", "piso_fixo", "gama", "c_livre"}   # sem saturação: sem a_parte
    assert len(aj) == 6 * len(pares)
    assert (aj["modelo"].fillna("") != "").all()
    ds = cadeia["dataset"].set_index("video")
    for nome, x in v.items():
        for c in CANAIS:
            ok = c in x["canais"]
            assert pd.notna(ds.loc[nome, f"{c}_c0_tau_medio"]) == ok
            assert bool(ds.loc[nome, f"usar_no_ajuste_{c}"]) == ok


def test_preto_por_video_recuperado(cadeia):
    """piso_video = piso dos cantos - (cantos - ROI do controle) tem de dar o preto verdadeiro da sessão."""
    vd = cadeia["videos"].set_index("video")
    for nome, x in cadeia["verdade"].items():
        for c in CANAIS:
            assert vd.loc[nome, f"piso_video_{c}"] == pytest.approx(x["piso"][c], abs=0.01)


def test_tau_recuperado(cadeia):
    ds = cadeia["dataset"].set_index("video")
    erros = []
    for nome, x in cadeia["verdade"].items():
        for c in x["canais"]:
            t0 = float(ds.loc[nome, f"{c}_c0_inicio_s"])
            assert t0 >= 2.5 - 1e-9 and t0 < 2.5 + 0.7                             # janela fixa a partir de inicio_ajuste_s
            esp = _tau_medio_esperado(x, c, t0)
            for caminho, tol in (("c0", 0.05), ("cl", 0.08)):
                got = float(ds.loc[nome, f"{c}_{caminho}_tau_medio"])
                erros.append((nome, c, caminho, got / esp - 1))
                assert got == pytest.approx(esp, rel=tol), (nome, c, caminho, got, esp)
            tau2 = float(ds.loc[nome, f"{c}_c0_tau2"])
            assert tau2 == pytest.approx(x["canais"][c][3], rel=0.10), (nome, c, tau2)     # componente lenta
    assert len(erros) == 2 * sum(len(x["canais"]) for x in cadeia["verdade"].values())


def test_regravacao_mais_lenta_da_tau_maior(cadeia):
    """A regravação 2 foi gerada com τ 3% maior: o pipeline deve ordenar igual em cada amostra/proporção/canal."""
    ds = cadeia["dataset"].set_index("video")
    for amostra in "ab":
        for prop, canais in SINAIS.items():
            for c in canais:
                assert ds.loc[f"{prop}_v2{amostra}", f"{c}_c0_tau_medio"] > ds.loc[f"{prop}_v1{amostra}", f"{c}_c0_tau_medio"]


def _proibidas(cadeia):
    return [c for c in cadeia["dataset"].columns if any(re.match(p, c) for p in PROIBIDAS)]


def test_sem_vazamento_no_dicionario(cadeia):
    """Alvo (prop_*), canal_ativo_*, usar_no_ajuste_*, sat_ate_s, excluido_saturacao e meta nunca têm papel 'feature'."""
    proibidas = _proibidas(cadeia)
    assert {"prop_R", "prop_G", "prop_B", "canal_ativo_R", "usar_no_ajuste_G", "sat_ate_s", "excluido_saturacao", "video"} <= set(proibidas)
    papel = cadeia["dicionario"].set_index("coluna")["papel"]
    vazando = papel.loc[proibidas][papel.loc[proibidas] == "feature"]
    assert vazando.empty, vazando.to_dict()
    assert (papel == "feature").any()                                              # há o que a IA pode usar


def test_sem_vazamento_nas_entradas(cadeia):
    """A seleção de entradas (entradas.py) bloqueia as colunas proibidas em todas as combinações de caminho."""
    entradas = pytest.importorskip("luminescencia.entradas")
    proibidas = set(_proibidas(cadeia))
    for caminhos in (["cl", "c0"], ["cl"], ["c0"]):
        cols = set(entradas.colunas_de_entrada(cadeia["dicionario"], entradas.parametros({"entradas": {"caminhos": caminhos}})))
        assert cols and not cols & proibidas, cols & proibidas
        assert cols <= set(cadeia["dataset"].columns)


def test_numero_de_colunas_de_referencia(cadeia):
    """Mesmo layout das tabelas reais (236 x 106 e 236 x 39). Se mudar de propósito, atualize estes números e o reproduzir.md."""
    assert cadeia["dataset"].shape[1] == N_COLUNAS_DATASET and cadeia["videos_pos"].shape[1] == N_COLUNAS_VIDEOS_POS


def test_determinismo_duas_execucoes_iguais(cadeia, cadeia_repetida):
    for chave in ("videos", "series", "videos_pos", "series_pos", "ajustes", "dataset", "dicionario"):
        pd.testing.assert_frame_equal(cadeia[chave], cadeia_repetida[chave], check_exact=True, obj=chave)
    a = (cadeia["saida"] / "tabela" / "dataset.csv").read_bytes()
    b = (cadeia_repetida["saida"] / "tabela" / "dataset.csv").read_bytes()
    assert a == b


def test_etapa_sem_a_anterior_falha_com_mensagem(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(FileNotFoundError):
        consolidacao.rodar(cfg)
    with pytest.raises(FileNotFoundError):
        posprocessamento.rodar(cfg)
    with pytest.raises(FileNotFoundError):
        tabela.rodar(cfg)
