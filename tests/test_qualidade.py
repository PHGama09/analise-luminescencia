"""Testes de qualidade.py com dados SINTÉTICOS de resposta conhecida (nada é lido dos CSVs reais)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from luminescencia import qualidade as Q

CFG = {"divisao": {"treino": ["A", "B"], "validacao": ["C"]},
       "qualidade": {"n_permutacoes": 400, "semente": 7, "min_linhas": 30, "min_proporcoes": 4}}


def _sintetico(seed: int = 0, n_prop: int = 16, amostras=("A", "B", "C"), n_vid: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    linhas = []
    for i in range(n_prop):
        for am in amostras:
            off = rng.normal(0, 1)                      # efeito da preparação (não da proporção)
            for v in range(n_vid):
                linhas.append({"video": f"P{i:02d}_{am}_v{v}", "amostra": am, "grupo": "treino" if am != "C" else "validacao",
                               "num_video": v, "proporcao": f"P{i:02d}", "excluido_saturacao": False, "_i": i,
                               "R_c0_r2": 0.90 + 0.01 * i + rng.normal(0, 0.002),     # depende da proporção
                               "G_c0_r2": rng.normal(0.95, 0.01),                      # independente
                               "B_c0_rmse": 5 + off + rng.normal(0, 0.3),              # só da preparação
                               "R_c0_ok": True,                                        # constante
                               "G_c0_n_pontos": float(i)})                             # idêntica ao índice da proporção
    return pd.DataFrame(linhas)


def _res(df, cols):
    return Q.analisar(df, cols, Q.parametros(CFG))


def _ver(res, col):
    return res["eta2"].set_index("diagnostico").loc[col]


def test_dependente_da_proporcao_e_vies_e_independente_e_usavel():
    df = _sintetico()
    r = _res(df, ["R_c0_r2", "G_c0_r2"])
    assert _ver(r, "R_c0_r2")["veredito"] == "viés"
    assert _ver(r, "G_c0_r2")["veredito"] == "usável como peso"


def test_coluna_identica_a_proporcao_tem_eta2_um():
    r = _res(_sintetico(), ["G_c0_n_pontos"])
    assert _ver(r, "G_c0_n_pontos")["eta2_prop"] == pytest.approx(1.0)
    assert _ver(r, "G_c0_n_pontos")["veredito"] == "viés"


def test_efeito_so_da_preparacao_nao_vira_vies():
    # variância grande ENTRE preparações, nenhuma da proporção: a permutação por preparação deve absorvê-la
    r = _res(_sintetico(), ["B_c0_rmse"])
    assert _ver(r, "B_c0_rmse")["veredito"] != "viés"


def test_constante_nao_serve_de_peso():
    r = _res(_sintetico(), ["R_c0_ok"])
    assert _ver(r, "R_c0_ok")["veredito"] == "sem variação"
    assert r["candidatos"].empty


def test_amostra_c_e_saturados_ficam_fora():
    df = _sintetico()
    df.loc[df["video"] == "P00_A_v0", "excluido_saturacao"] = True
    r = _res(df, ["G_c0_r2"])
    assert r["info"]["amostras"] == ["A", "B"]
    assert r["info"]["n_videos"] == 16 * 2 * 5 - 1
    assert set(r["dist_amostra"]["amostra"]) == {"A", "B"}
    assert "P00_A_v0" not in set(r["candidatos"]["video"])


def test_semente_fixa_reproduz():
    df = _sintetico()
    a, b = _res(df, ["G_c0_r2"])["eta2"], _res(df, ["G_c0_r2"])["eta2"]
    pd.testing.assert_frame_equal(a, b)


def test_candidatos_ok_falso_resposta_conhecida():
    df = _sintetico()
    df["B_c0_ok"] = True
    alvo = ["P03_A_v0", "P03_A_v1"]                      # 2 de 5 vídeos da MESMA preparação
    df.loc[df["video"].isin(alvo), "B_c0_ok"] = False
    r = _res(df, ["B_c0_ok"])
    assert sorted(r["candidatos"]["video"]) == alvo
    s = r["candidatos_resumo"].iloc[0]
    assert (s["preparacoes_com_candidato"], s["preparacoes_maioria_candidata"], s["n_proporcoes_com_candidato"]) == (1, 0, 1)
    assert s["preparacoes_total"] == 32 and s["proporcao_mais_afetada"] == "P03"
    df.loc[df["video"] == "P03_A_v2", "B_c0_ok"] = False  # 3 de 5: a preparação passa a ter maioria
    assert _res(df, ["B_c0_ok"])["candidatos_resumo"].iloc[0]["preparacoes_maioria_candidata"] == 1


def test_candidatos_continuo_pior_lado_conforme_direcao():
    df = _sintetico()
    df["G_c0_rmse"] = 1.0 + np.arange(len(df)) * 1e-4
    df.loc[df["video"] == "P05_B_v2", "G_c0_rmse"] = 99.0   # rmse: menor é melhor, então o 99 é o pior
    r = _res(df, ["G_c0_rmse"])
    assert "P05_B_v2" in set(r["candidatos"]["video"])
    assert r["candidatos"].set_index("video").loc["P05_B_v2", "valor"] == 99.0


def test_amostra_depois_da_proporcao_detecta_deriva():
    df = _sintetico()
    df["R_c0_rmse"] = np.where(df["amostra"] == "B", 3.0, 0.0) + np.random.default_rng(1).normal(0, 0.2, len(df))
    r = _res(df, ["R_c0_rmse"])
    assert _ver(r, "R_c0_rmse")["aviso_amostra"] != ""
    assert _ver(r, "R_c0_rmse")["veredito"] == "usável como peso"


def test_ausencia_que_depende_da_proporcao_e_sinalizada():
    df = _sintetico()
    df["R_c0_rmse"] = np.random.default_rng(2).normal(1, 0.1, len(df))
    df.loc[df["_i"] < 8, "R_c0_rmse"] = np.nan               # sem ajuste nas 8 primeiras proporções
    row = _ver(_res(df, ["R_c0_rmse"]), "R_c0_rmse")
    assert row["n_ausentes"] == 8 * 2 * 5 and row["eta2_prop_ausencia_ajustado"] > 0.5 and row["p_prop_ausencia"] < 0.05


def test_poucos_dados_ou_proporcoes():
    df = _sintetico()
    df["B_c0_rmse"] = np.nan
    df.loc[df.index[:10], "B_c0_rmse"] = np.arange(10.0)
    assert _ver(_res(df, ["B_c0_rmse"]), "B_c0_rmse")["veredito"] == "dados insuficientes"


def test_parametros_invalidos():
    with pytest.raises(ValueError):
        Q.parametros({"qualidade": {"alfa": 2}})
    with pytest.raises(ValueError):
        Q.parametros({"qualidade": {"direcao": {"r2": "alto"}}})
    assert Q.parametros({"divisao": {"treino": ["A"]}})["amostras"] == ["A"]


def test_rodar_grava_saidas_e_relatorio(tmp_path):
    df = _sintetico()
    tab = tmp_path / "output" / "tabela"
    tab.mkdir(parents=True)
    df.drop(columns=["_i", "R_c0_ok"]).to_csv(tab / "dataset.csv", index=False)
    desc = df[["video", "amostra", "grupo", "num_video", "proporcao"]].copy()
    desc["R_md_ok"] = np.where(df["video"] == "P02_B_v1", False, True)
    desc.to_csv(tab / "descritores.csv", index=False)
    cfg = dict(CFG, pastas={"saida": str(tmp_path / "output")})
    assert Q.rodar(cfg) == 0
    out = tmp_path / "output" / "qualidade"
    for nome in ("diagnosticos_eta2.csv", "distribuicao_por_proporcao.csv", "distribuicao_por_amostra.csv",
                 "candidatos_baixo_peso.csv", "candidatos_resumo.csv", "relatorio_qualidade.txt"):
        assert (out / nome).exists(), nome
    e = pd.read_csv(out / "diagnosticos_eta2.csv").set_index("diagnostico")
    assert "R_md_ok" in e.index and e.loc["R_c0_r2", "veredito"] == "viés" and e.loc["G_c0_r2", "veredito"] == "usável como peso"
    assert "P02_B_v1" in set(pd.read_csv(out / "candidatos_baixo_peso.csv")["video"])
    assert "nenhum peso foi aplicado" in (out / "relatorio_qualidade.txt").read_text(encoding="utf-8")
    assert b"\r\n" not in (out / "relatorio_qualidade.txt").read_bytes()


def test_cli_registra_o_comando(capsys):
    from luminescencia.cli import main
    with pytest.raises(SystemExit) as exc:
        main(["qualidade", "--help"])
    assert exc.value.code == 0
    assert "--tabela" in capsys.readouterr().out
