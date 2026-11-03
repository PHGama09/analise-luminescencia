"""Testes dos gráficos 5 a 7 (P4, v0.11.0): as inclinações dizem qual transformação é reta; fumaça com dados sintéticos. Rode: pytest"""
import numpy as np
import pandas as pd

from luminescencia.graficos import (_ticks_log, grafico_transformacoes, inclinacoes, parametros, sinais_normalizados)

T = np.arange(0, 57, 1 / 7.5) + 2.5          # tempo do vídeo (s); a janela começa em 2,5 s


def _entradas(forma, C=0.0, n=3):
    series, aj, vp = [], [], []
    rng = np.random.default_rng(0)
    for i in range(n):
        v = f"G_15_v{i + 1}a"
        tr = T - T[0]
        y = {"exp": 100 * np.exp(-tr / 10), "pot": 100 * (1 + tr / 5) ** -1.5, "2a": 100 / (1 + tr / 8)}[forma] + C
        y = y + rng.normal(0, 0.05, len(T))
        series.append(pd.DataFrame({"video": v, "tempo_s": T, "corr_G": y, "corr_R": 0.0, "corr_B": 0.0}))
        for cam in ("principal", "c_livre"):
            aj.append(dict(video=v, amostra="A", proporcao="G1.5", canal="G", caminho=cam, modelo="exp_dupla", inicio_s=2.5,
                           fim_s=float(T[-1]), ruido=0.05, C=C if cam == "c_livre" else np.nan))
        vp.append({"video": v, "proporcao": "G1.5", **{f"canal_ativo_{c}": c == "G" for c in "RGB"}})
    return pd.concat(series), pd.DataFrame(aj), pd.DataFrame(vp)


def _razoes(forma, **k):
    s, aj, vp = _entradas(forma, **k)
    n = sinais_normalizados(s, vp, aj, 0.5)
    inc = inclinacoes(n, parametros({}))
    return inc[(inc["versao"] == "c0")].median(numeric_only=True)


def test_exponencial_simples_e_reta_em_ln_I_contra_t():
    r = _razoes("exp")
    assert abs(r["ln_t_razao"] - 1) < 0.05 and abs(r["loglog_razao"] - 1) > 0.3


def test_lei_de_potencia_nao_e_reta_em_ln_I_contra_t():
    # A(1 + t/t0)^-a: a taxa de queda em ln I x t cai com o tempo (razão tarde/cedo bem abaixo de 1)
    r = _razoes("pot")
    assert r["ln_t_razao"] < 0.6


def test_segunda_ordem_e_reta_em_1_sobre_I():
    r = _razoes("2a")
    assert abs(r["inverso_razao"] - 1) < 0.05 and abs(r["ln_t_razao"] - 1) > 0.2


def test_versao_cl_tira_o_patamar_C():
    # sinal exponencial + patamar 5: com C = 0 deixa de ser reta em ln I; tirando o C do ajuste cl volta a ser reta
    s, aj, vp = _entradas("exp", C=5.0)
    inc = inclinacoes(sinais_normalizados(s, vp, aj, 0.5), parametros({}))
    c0, cl = (inc[inc["versao"] == v]["ln_t_razao"].median() for v in ("c0", "cl"))
    assert abs(cl - 1) < 0.1 and c0 < 0.8


def test_sinal_no_ruido_nao_entra_na_inclinacao():
    s, aj, vp = _entradas("exp")
    aj["ruido"] = 80.0                           # ruído enorme: nenhum ponto passa de 3 x ruído -> NaN, sem erro
    inc = inclinacoes(sinais_normalizados(s, vp, aj, 0.5), parametros({}))
    assert inc["ln_t_razao"].isna().all()


def test_graficos_5_a_7_geram_arquivos_e_resumo(tmp_path):
    s, aj, vp = _entradas("exp")
    p = {**parametros({}), "dpi": 40}
    L = grafico_transformacoes(s, vp, aj, p, tmp_path)
    for arq in ("5_log_log.png", "6_ln_I_vs_t.png", "7_inverso_1_sobre_I.png"):
        assert (tmp_path / arq).exists()
    assert any("ln_t" in l and "G c0" in l for l in L)


def test_zero_do_log_log_vem_do_protocolo():
    assert parametros({"protocolo": {"instante_queda_s": 0.5}})["instante_queda_s"] == 0.5
    assert parametros({"protocolo": {"instante_queda_s": 0.4}})["instante_queda_s"] == 0.4


def test_rotulos_log_legiveis():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(3, 30); ax.set_ylim(3, 30)
    _ticks_log(ax, [3, 30])
    assert [t.get_text() for t in ax.get_xticklabels()] == ["3", "4", "5", "6", "8", "10", "15", "20", "30"]
    plt.close(fig)
