"""Testes do pós-processamento: janela fixa, saturação, canais sem pó. Rode: pytest"""
import numpy as np
import pandas as pd
import pytest

from luminescencia.posprocessamento import parametros, recortar, resumir_videos

CFG = {"protocolo": {"instante_queda_s": 0.5}}


def _serie(video, sat_ate=0.0, corr_R=5.0):
    t = np.arange(0, 20, 1 / 7.5)
    return pd.DataFrame({
        "video": video, "tempo_s": t, "fase": np.where(t < 0.5, "excitacao", "decaimento"),
        "saturados_pct": np.where(t <= sat_ate, 3.0, 0.0), "corr_R": corr_R, "corr_G": 8.0, "corr_B": 9.0})


def _videos():
    base = {"amostra": "A", "grupo": "treino", "num_video": 1, "prop_R": 0.0, "prop_G": 1.0, "prop_B": 0.5,
            "proporcao": "G1_B0.5", "canal_ativo_R": False, "canal_ativo_G": True, "canal_ativo_B": True,
            "perto_do_ruido_R": True, "perto_do_ruido_G": False, "perto_do_ruido_B": False}
    return pd.DataFrame([{**base, "video": "ok"}, {**base, "video": "sat"}])


def test_janela_comeca_no_mesmo_instante_para_todos():
    p = parametros(CFG)
    s = pd.concat([_serie("ok"), _serie("sat", sat_ate=3.2)])
    rec = recortar(s, p)
    assert rec["tempo_s"].min() >= 2.0 - 1e-9 and (rec["fase"] == "decaimento").all()
    vp = resumir_videos(_videos(), rec, p).set_index("video")
    assert vp.loc["ok", "tempo_inicial_s"] == vp.loc["sat", "tempo_inicial_s"]


def test_saturacao_depois_do_inicio_tira_do_ajuste_principal():
    p = parametros(CFG)
    s = pd.concat([_serie("ok", sat_ate=1.0), _serie("sat", sat_ate=3.2)])
    vp = resumir_videos(_videos(), recortar(s, p), p).set_index("video")
    assert bool(vp.loc["ok", "usar_no_ajuste_G"]) and not bool(vp.loc["ok", "satura_apos_2s"])
    assert not bool(vp.loc["sat", "usar_no_ajuste_G"]) and "saturacao" in vp.loc["sat", "motivo_fora_G"]
    assert vp.loc["sat", "satura_apos_3s"] and not vp.loc["sat", "satura_apos_3_5s"] if "satura_apos_3_5s" in vp else True


def test_canal_sem_po_mas_com_sinal_entra_e_so_o_ruido_tira():
    p = parametros(CFG)
    v = _videos()
    v["perto_do_ruido_R"] = [True, False]                      # ok: R perto do ruído; sat: R com sinal (vazamento de cor)
    s = pd.concat([_serie("ok"), _serie("sat")])
    vp = resumir_videos(v, recortar(s, p), p).set_index("video")
    assert vp.loc["ok", "motivo_fora_R"] == "perto_do_ruido" and not bool(vp.loc["ok", "usar_no_ajuste_R"])
    assert bool(vp.loc["sat", "usar_no_ajuste_R"]) and not bool(vp.loc["sat", "canal_ativo_R"])    # sem pó, mas medido: entra
    assert bool(vp.loc["sat", "sinal_presente_R"]) and not bool(vp.loc["ok", "sinal_presente_R"])


def test_a_decisao_nao_olha_a_proporcao():
    p = parametros(CFG)
    a, b = _videos(), _videos()
    b[["prop_R", "prop_G", "prop_B"]] = [1.0, 0.0, 0.0]; b[["canal_ativo_R", "canal_ativo_G", "canal_ativo_B"]] = [True, False, False]
    s = pd.concat([_serie("ok"), _serie("sat")])
    ua = resumir_videos(a, recortar(s, p), p)[[f"usar_no_ajuste_{c}" for c in "RGB"]]
    ub = resumir_videos(b, recortar(s, p), p)[[f"usar_no_ajuste_{c}" for c in "RGB"]]
    assert ua.equals(ub)                                        # mudar o alvo não muda quem é ajustado


def test_aviso_de_cauda_no_ruido_nao_exclui():
    p = parametros(CFG)
    v = _videos()
    v["perto_do_ruido_R"] = False
    vp = resumir_videos(v, recortar(_serie("ok", corr_R=-0.5), p), p).set_index("video")
    assert bool(vp.loc["ok", "aviso_cauda_no_ruido_R"]) and bool(vp.loc["ok", "usar_no_ajuste_R"])


def test_inicio_antes_do_corte_e_erro():
    with pytest.raises(ValueError):
        parametros({**CFG, "posprocessamento": {"inicio_ajuste_s": 0.3}})
