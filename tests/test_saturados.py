"""Testes sintéticos do Q4/B11 (saturados.py): respostas conhecidas, nada de dado real."""
import math

import numpy as np
import pandas as pd
import pytest

from luminescencia import saturados as S

CFG = {"saturados": {"min_irmaos": 3, "cv_piso": 0.02, "z_comparavel": 2.0, "z_duvidoso": 3.0}, "modelagem": {"workers": 1}}
DT = 0.2


def serie(video, amostra, prop, tau, sat_ate=0.0, canais=("G",), amp=200.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.round(np.arange(0.5, 60.0 + 1e-9, DT), 4)
    d = {"video": video, "amostra": amostra, "proporcao": prop, "tempo_s": t, "saturados_pct": np.where(t <= sat_ate, 1.0, 0.0)}
    for c in "RGB":
        d[f"corr_{c}"] = amp * np.exp(-(t - 0.5) / tau) + rng.normal(0, 0.05, len(t)) if c in canais else np.zeros(len(t))
    sat = {"video": video, "amostra": amostra, "proporcao": prop, "sat_ate_s": sat_ate}
    for c in "RGB":
        sat[f"sinal_presente_{c}"] = c in canais
        sat[f"usar_no_ajuste_{c}"] = (c in canais) and sat_ate < 2.5
    return pd.DataFrame(d), sat


def montar(specs, canais=("G",)):
    """specs: lista de (video, amostra, proporcao, tau, sat_ate)."""
    ss, vs = [], []
    for i, (v, a, pr, tau, sa) in enumerate(specs):
        s, vv = serie(v, a, pr, tau, sa, canais, seed=i)
        ss.append(s); vs.append(vv)
    return pd.concat(ss, ignore_index=True), pd.DataFrame(vs)


def grupo(prop, amostra, tau_base, n=4, fator=1.0, sat_ate=0.0, prefixo=None):
    pre = prefixo or f"{prop}_{amostra}"
    return [(f"{pre}_v{k + 1}", amostra, prop, tau_base * fator * (1 + 0.01 * (k - 1.5)), sat_ate) for k in range(n)]


# ----------------------------------------------------------------------------- funções puras
def test_descritores_exponencial_conhecida():
    t = np.arange(0, 60, DT)
    r = S.descritores_janela(t, 100 * np.exp(-t / 10), S.parametros(CFG))
    p = S.parametros(CFG)
    assert r["I0"] == pytest.approx(100 * np.exp(-np.median(t[:p["t50_pontos_i0"]]) / 10), rel=1e-6)
    assert r["t50"] == pytest.approx(10 * math.log(2) + 10 * math.log(100 / r["I0"]), abs=0.3)   # t50 mede a queda a I0/2


def test_t50_nan_se_nao_chega_na_metade():
    t = np.arange(0, 5, DT)
    assert math.isnan(S.descritores_janela(t, 100 * np.exp(-t / 100), S.parametros(CFG))["t50"])


def test_classe_z_limiares_e_calibracao():
    p = S.parametros(CFG)
    assert S.classe_z(1.5, p) == 0 and S.classe_z(-2.5, p) == 1 and S.classe_z(3.5, p) == 2
    assert S.classe_z(float("nan"), p) is None
    assert S.classe_z(4.0, p, z_irmaos=5.0) == 0          # irmãos destoam entre si até 5: 4 não é pior que eles
    assert S.classe_z(4.0, {**p, "calibrar_pelos_irmaos": False}, z_irmaos=5.0) == 2


def test_razao_grande_nunca_e_comparavel():
    p = S.parametros(CFG)
    assert S.limitar_por_razao(0, 0.24, p) == 1 and S.limitar_por_razao(0, 1.19, p) == 0
    assert S.limitar_por_razao(2, 1.0, p) == 2 and S.limitar_por_razao(0, float("nan"), p) == 0


def test_comparar_valor_conhecido():
    p = S.parametros(CFG)
    r = S.comparar_valor(11.0, [10.0, 10.2, 9.8, 10.0], p)
    assert r["n"] == 4 and r["razao"] == pytest.approx(1.1, rel=1e-6)
    cv = np.std([10.0, 10.2, 9.8, 10.0], ddof=1) / 10.0
    assert r["z"] == pytest.approx(0.1 / max(cv, p["cv_piso"]))
    assert S.comparar_valor(10.0, [10.0, 10.0, 10.0], p)["z"] == pytest.approx(0.0)      # colunas idênticas: razão 1, z 0
    assert math.isnan(S.comparar_valor(float("nan"), [1, 2, 3], p)["z"])


def test_janela_comeca_no_primeiro_quadro_depois_da_saturacao():
    s, v = serie("a", "A", "P", 10.0, sat_ate=3.0)
    j = S.janela_saturado(s, "G", 3.0, S.parametros(CFG))
    primeiro = float(s["tempo_s"][s["tempo_s"] > 3.0].iloc[0])
    assert j["motivo"] == "" and j["inicio_s"] == pytest.approx(primeiro) and 5 < j["fim_s"] <= 60


def test_recorte_rejeita_quadro_saturado_e_janela_nao_coberta():
    p = S.parametros(CFG)
    s, _ = serie("a", "A", "P", 10.0, sat_ate=6.0)
    assert S.recorte(s, "G", 5.0, 30.0, p) is None            # há quadro saturado dentro da janela
    assert S.recorte(s, "G", 7.0, 30.0, p) is not None
    assert S.recorte(s, "G", 7.0, 90.0, p) is None            # vídeo acaba antes do fim da janela


# ----------------------------------------------------------------------------- veredito
def _veredito(fator, extra=()):
    specs = grupo("P", "A", 10.0, n=4) + [("P_sat", "A", "P", 10.0 * fator, 3.0)] + list(extra)
    s, v = montar(specs)
    A = S.analisar(s, v, None, CFG)
    return A, A["veredito_video"].set_index("video").loc["P_sat"]


def test_mesma_curva_e_comparavel():
    A, r = _veredito(1.0)
    assert r["veredito"] == "comparável"
    comp = A["comparacao"].query("video == 'P_sat' and base == 'mesma_amostra' and grandeza == 'tau_medio'").iloc[0]
    assert comp["n"] == 4 and comp["razao"] == pytest.approx(1.0, abs=0.03)


def test_deriva_injetada_de_10_por_cento_e_detectada():
    A, r = _veredito(1.10)
    comp = A["comparacao"].query("video == 'P_sat' and base == 'mesma_amostra' and grandeza == 'tau_medio'").iloc[0]
    assert comp["razao"] == pytest.approx(1.10, abs=0.03)
    assert r["veredito"] == "não usar"


def test_deriva_de_2_por_cento_nao_e_marcada():
    assert _veredito(1.02)[1]["veredito"] == "comparável"


def test_irmao_saturado_dentro_da_janela_nao_entra():
    extra = [("P_outro_sat", "A", "P", 10.0, 20.0)]            # satura até 20 s: não cobre a janela do P_sat
    A, _ = _veredito(1.0, extra)
    assert "P_outro_sat" not in set(A["irmaos"].query("video == 'P_sat'")["irmao"])


def test_so_outras_amostras_no_maximo_duvidoso():
    specs = grupo("P", "B", 10.0, n=4) + [("P_sat", "A", "P", 10.0, 3.0)]
    s, v = montar(specs)
    r = S.analisar(s, v, None, CFG)["veredito_video"].iloc[0]
    assert r["veredito"] == "duvidoso" and "outras_amostras" in r["motivo"]


def test_sem_irmaos_nao_usar():
    s, v = montar([("P_sat", "A", "P", 10.0, 3.0)])
    r = S.analisar(s, v, None, CFG)["veredito_video"].iloc[0]
    assert r["veredito"] == "não usar"


def test_sem_saturados_nao_quebra():
    s, v = montar(grupo("P", "A", 10.0, n=4))
    A = S.analisar(s, v, None, CFG)
    assert A["veredito_video"].empty and A["janelas"].empty


# ----------------------------------------------------------------------------- efeito no grupo e saída
def _cenario(fator_alvo):
    specs = grupo("T", "A", 10.0, fator=fator_alvo, sat_ate=3.0) + grupo("T", "B", 10.0)
    for k in range(4):
        specs += grupo(f"R{k}", "A", 10.0 + k) + grupo(f"R{k}", "B", 10.0 + k)       # razão A/B = 1 nas referências
    return montar(specs)


@pytest.mark.parametrize("fator, classe", [(1.0, "comparável"), (1.5, "não usar")])
def test_efeito_grupo_contra_deriva_das_outras_proporcoes(fator, classe):
    s, v = _cenario(fator)
    A = S.analisar(s, v, None, CFG)
    e = S.efeito_grupo(A, v, s)
    g = e[(e["grandeza"] == "tau_medio") & (e["amostra_ref"] == "B")].iloc[0]
    assert (g["proporcao"], g["amostra_alvo"]) == ("T", "A") and g["n_outras_proporcoes"] == 4
    assert g["razao_alvo"] == pytest.approx(fator, abs=0.03)
    assert g["classe"] == classe
    t = S.treino_por_grupo(v, A["veredito_video"]).set_index(["proporcao", "amostra"]).loc[("T", "A")]
    assert t["n_treino_hoje"] == 0 and t["n_a_parte_comparavel"] + t["n_a_parte_duvidoso"] + t["n_a_parte_nao_usar"] == 4


def test_rodar_grava_saidas(tmp_path):
    s, v = _cenario(1.0)
    pos = tmp_path / "pos"; pos.mkdir()
    s.to_csv(pos / "series_pos.csv", index=False); v.to_csv(pos / "videos_pos.csv", index=False)
    assert S.rodar(CFG, pasta_pos=pos, saida=tmp_path / "out") == 0
    for f in ("saturados_janelas", "irmaos_ajustes", "comparacao", "veredito_canal", "veredito_video", "treino_por_grupo", "efeito_grupo"):
        assert (tmp_path / "out" / f"{f}.csv").exists()
    txt = (tmp_path / "out" / "relatorio_saturados.txt").read_text(encoding="utf-8")
    assert "Veredito por vídeo" in txt and "T_A_v1" in txt


def test_rodar_sem_arquivos_falha(tmp_path):
    with pytest.raises(FileNotFoundError):
        S.rodar(CFG, pasta_pos=tmp_path, saida=tmp_path / "o")
