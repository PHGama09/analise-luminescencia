"""Teste de fumaça dos gráficos com dados sintéticos (sem vídeos). Rode: pytest"""
import numpy as np
import pandas as pd

from luminescencia.graficos import curva, grafico_normalizada, grafico_residuos, grafico_semilog, grafico_tau, parametros, residuos
from luminescencia.modelos import exp_dupla_c0

T = np.arange(0, 40, 1 / 7.5)


def _dados():
    series, aj, vp = [], [], []
    rng = np.random.default_rng(0)
    for i, am in enumerate("AAABBC"):
        v = f"G_15_v{i + 1}{am.lower()}"
        y = exp_dupla_c0(T, 60, 3, 40, 20) + rng.normal(0, 0.1, len(T))
        series.append(pd.DataFrame({"video": v, "tempo_s": T + 2.5, "corr_G": y, "corr_R": 0.0, "corr_B": 0.0}))
        for cam in ("principal", "c_livre"):
            aj.append(dict(video=v, amostra=am, proporcao="G1.5", canal="G", caminho=cam, modelo="exp_dupla_c0" if cam == "principal" else "exp_dupla",
                           A1=60, tau1=3, A2=40, tau2=20, C=np.nan if cam == "principal" else 0.0, tau_medio=10.4, inicio_s=2.5,
                           fim_s=float(T[-1] + 2.5), ruido=0.1))
        vp.append({"video": v, "proporcao": "G1.5", **{f"canal_ativo_{c}": c == "G" for c in "RGB"}})
    return pd.concat(series), pd.DataFrame(aj), pd.DataFrame(vp)


def test_curva_reproduz_o_modelo():
    s, aj, vp = _dados()
    l = aj.iloc[0]
    assert np.allclose(curva(l, T), exp_dupla_c0(T, 60, 3, 40, 20))


def test_residuos_pequenos_quando_o_modelo_e_o_certo():
    s, aj, vp = _dados()
    r = residuos(aj, s)
    assert abs(r["residuo"].median()) < 0.05 and set(r["caminho"]) == {"principal", "c_livre"}


def test_graficos_geram_arquivos(tmp_path):
    s, aj, vp = _dados()
    p = {**parametros({}), "proporcoes_tipicas": ["G1.5"], "dpi": 40}
    assert grafico_semilog(aj, s, p, tmp_path) and (tmp_path / "1_decaimento_semilog.png").exists()
    assert grafico_tau(aj, vp, p, tmp_path) and (tmp_path / "2_tau_cl_vs_c0.png").exists()
    assert grafico_residuos(aj, s, p, tmp_path) and (tmp_path / "3_residuos.png").exists()
    assert grafico_normalizada(s, vp, aj, p, tmp_path) and (tmp_path / "4_normalizada.png").exists()
