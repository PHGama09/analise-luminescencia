"""Testes da sensibilidade de gama (D5): só um caminho a mais, o principal não muda. Rode: pytest"""
import numpy as np
import pandas as pd

from luminescencia.modelar import linearizar_gama, montar_tarefas, parametros
from luminescencia.posprocessamento import parametros as parametros_pos


def test_linearizar_gama_zero_no_preto_e_monotonico():
    media = np.array([2.0, 10.0, 100.0, 255.0])
    lin = linearizar_gama(media, 2.0, 2.2)
    assert abs(lin[0]) < 1e-9 and np.all(np.diff(lin) > 0) and abs(lin[-1] - 255 * (1 - (2 / 255) ** 2.2)) < 1e-6


def test_linearizar_gama_um_e_so_subtrair_o_preto():
    media = np.array([5.0, 50.0, 120.0])
    assert np.allclose(linearizar_gama(media, 3.0, 1.0), media - 3.0)


def _entradas():
    t = np.arange(0, 20, 1 / 7.5)
    s = pd.DataFrame({"video": "v1", "tempo_s": t, **{f"media_{c}": 20 + 100 * np.exp(-t / 5) for c in "RGB"},
                      **{f"corr_{c}": 100 * np.exp(-t / 5) for c in "RGB"}})
    base = {"video": "v1", "amostra": "A", "grupo": "treino", "num_video": 1, "proporcao": "G1", "sat_ate_s": 0.0}
    for c in "RGB":
        base.update({f"usar_no_ajuste_{c}": c == "G", f"sinal_presente_{c}": c == "G", f"piso_video_{c}": 20.0})
    vcons = pd.DataFrame({"video": ["v1"], **{f"piso_ref_{c}": 1.0 for c in "RGB"}})
    return s, pd.DataFrame([base]), vcons


def test_gama_vira_um_caminho_e_pode_ser_desligado():
    s, vp, vc = _entradas()
    pp = parametros_pos({"protocolo": {"instante_queda_s": 0.5}})
    p = parametros({})
    cams = [t[0]["caminho"] for t in montar_tarefas(s, vp, vc, {}, p, pp)]
    assert "gama" in cams and "principal" in cams
    cams = [t[0]["caminho"] for t in montar_tarefas(s, vp, vc, {}, {**p, "gama": None}, pp)]
    assert "gama" not in cams and "principal" in cams
