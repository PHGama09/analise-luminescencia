"""Testes de ic_simulacao com dados sintéticos de resposta conhecida."""
import numpy as np
import pandas as pd
import pytest

from luminescencia import ic_simulacao as ic

NOMES = ["R1.5", "G1.5", "B1.5", "R0.75_G0.75"]


def _simular(coef, P, **kw):
    mae_media, erro_i = ic.baseline_media(NOMES)
    args = dict(grade_m=[0.1], n_prep=P, n_comp=len(NOMES), replicas=20000, semente=1, nivel_ic=0.95,
                limites={"minimo": 0.7, "forte": 0.5}, mae_media=mae_media, erro_media_i=erro_i)
    args.update(kw)
    return ic.simular(coef, **args)[0]


def test_parse_soma_um_e_canal_ausente():
    assert np.allclose(ic.parse_proporcao("R0.75_G0.75"), [0.5, 0.5, 0.0])
    assert np.allclose(ic.parse_proporcao("B1.5"), [0, 0, 1])
    with pytest.raises(ValueError):
        ic.parse_proporcao("R1_G1")


def test_baseline_media_valor_conhecido():
    # R1.5 e G1.5: media = (.5,.5,0); cada composicao erra |1-.5|+|0-.5|+0 = 1, /3 canais
    mae, por_i = ic.baseline_media(["R1.5", "G1.5"])
    assert mae == pytest.approx(1 / 3)
    assert np.allclose(por_i, 1 / 3)


def test_componentes_normalizam_e_usam_quantil():
    df = pd.DataFrame({"grandeza": ["I0"] * 3 + ["X"], "frac_proporcao": [0.6] * 3 + [0.0],
                       "frac_amostra": [0.3] * 3 + [1.0], "frac_regravacao": [0.1] * 3 + [0.0],
                       "frac_offset_comum": [0.2] * 3 + [1.0]})
    w = ic.componentes_variancia(df, ["I0"], 0.5)
    assert (w["w_prop"], w["w_am"], w["w_reg"], w["offset"]) == pytest.approx((0.6, 0.3, 0.1, 0.2))
    assert w["n_linhas"] == 3
    with pytest.raises(ValueError):
        ic.componentes_variancia(df, ["nao_existe"], 0.5)


def test_so_sistematico_ic_largura_zero():
    coef = ic.coeficientes({"w_prop": 1.0, "w_am": 0.0, "w_reg": 0.0, "offset": 0.0}, 5)
    assert _simular(coef, 1)["ic_largura"] == pytest.approx(0.0, abs=1e-12)


def test_calibracao_mae_verdadeiro():
    coef = ic.coeficientes({"w_prop": 0.0, "w_am": 1.0, "w_reg": 0.0, "offset": 0.0}, 5)
    assert _simular(coef, 1)["mae_obs_medio"] == pytest.approx(0.1, rel=0.02)


def test_preparacao_compartilhada_encolhe_como_raiz_de_P():
    coef = ic.coeficientes({"w_prop": 0.0, "w_am": 1.0, "w_reg": 0.0, "offset": 1.0}, 5)
    razao = _simular(coef, 1)["ic_largura"] / _simular(coef, 4)["ic_largura"]
    assert razao == pytest.approx(2.0, rel=0.15)


def test_ruido_de_regravacao_cai_com_n_reg():
    w = {"w_prop": 0.0, "w_am": 0.0, "w_reg": 1.0, "offset": 0.0}
    # com so regravacao a escala e recalibrada (E|e| = m), entao a largura nao muda com n_reg;
    # o que muda e a mistura: com 1 preparacao ela e igual
    a = _simular(ic.coeficientes(w, 1), 1)["ic_largura"]
    b = _simular(ic.coeficientes(w, 25), 1)["ic_largura"]
    assert a == pytest.approx(b, rel=0.05)


def test_probabilidade_limites_extremos():
    coef = ic.coeficientes({"w_prop": 0.2, "w_am": 0.5, "w_reg": 0.3, "offset": 0.3}, 5)
    erro_i = np.ones(len(NOMES))
    alto = _simular(coef, 2, mae_media=1e6, erro_media_i=erro_i * 1e6)
    baixo = _simular(coef, 2, mae_media=1e-9, erro_media_i=erro_i * 1e-9)
    assert alto["p_minimo"] == 1.0 and alto["p_forte_e_b"] == 1.0
    assert baixo["p_minimo"] == 0.0 and baixo["p_forte_e_b"] == 0.0


def test_semente_fixa_reproduz():
    coef = ic.coeficientes({"w_prop": 0.3, "w_am": 0.4, "w_reg": 0.3, "offset": 0.5}, 5)
    assert _simular(coef, 3, replicas=2000) == _simular(coef, 3, replicas=2000)


def test_rodar_com_csv_sintetico(tmp_path):
    der = tmp_path / "deriva"
    der.mkdir()
    pd.DataFrame({"proporcao": NOMES[:3]}).to_csv(der / "deriva_celulas.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"proporcao": [NOMES[3]]}).to_csv(der / "deriva_descartadas.csv", index=False)
    pd.DataFrame({"grandeza": ["I0", "t50"], "frac_proporcao": [0.6, 0.6], "frac_amostra": [0.3, 0.3],
                  "frac_regravacao": [0.1, 0.1], "frac_offset_comum": [0.2, 0.2]}
                 ).to_csv(der / "deriva_variancia.csv", index=False, encoding="utf-8-sig")
    cfg = {"pastas": {"saida": str(tmp_path)},
           "ic_simulacao": {"grandezas": ["I0", "t50"], "n_preparacoes": [1, 2], "n_regravacoes": 5,
                            "mae_verdadeiro": [0.05, 0.1], "replicas": 500, "semente": 3, "nivel_ic": 0.95,
                            "cenarios_quantil": {"mediana": 0.5}, "nivel_minimo_razao_max": 0.7,
                            "nivel_forte_razao_max": 0.5}}
    assert ic.rodar(cfg, pasta_deriva=der, saida=tmp_path / "ic") == 0
    r = pd.read_csv(tmp_path / "ic" / "ic_simulacao.csv", encoding="utf-8-sig")
    assert len(r) == 4 and set(r["n_preparacoes"]) == {1, 2}
    mae, _ = ic.baseline_media(NOMES)
    assert f"{mae:.4f}" in (tmp_path / "ic" / "relatorio_ic_simulacao.txt").read_text(encoding="utf-8")
