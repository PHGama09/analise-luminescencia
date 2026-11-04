"""Q4/B12 - janela_r: dados sintéticos com resposta conhecida (nada de resultado escrito à mão no módulo)."""
import numpy as np
import pandas as pd
import pytest

from luminescencia import janela_r as J


def _dupla(t, a1=60.0, t1=3.0, a2=40.0, t2=8.0):
    return a1 * np.exp(-t / t1) + a2 * np.exp(-t / t2)


def test_normalizar_regras_poe_a_base_primeiro_e_tira_repetidas():
    r = J.normalizar_regras([{"k": 4}, {"k": 3}, {"k": 4}, {"k": 3, "duracao_max_s": 20}], k_base=3.0)
    assert [x["nome"] for x in r] == ["k3", "k4", "k3_d20s"]


def test_fim_cresce_com_k_menor_e_duracao_maxima_corta_exatamente():
    t = np.arange(2.5, 60.0, 1 / 15)
    rng = np.random.default_rng(1)
    y = _dupla(t - 2.5) + rng.normal(0, 0.05, len(t))
    f3, _ = J.fim_da_regra(t, y, {"k": 3.0}, 31)
    f6, _ = J.fim_da_regra(t, y, {"k": 6.0}, 31)
    assert f6 < f3                                   # limiar mais alto termina antes
    fd, _ = J.fim_da_regra(t, y, {"k": 3.0, "duracao_max_s": 20.0}, 31)
    assert fd == pytest.approx(t[0] + 20.0)          # corte exato
    fcurto, _ = J.fim_da_regra(t, y, {"k": 3.0, "duracao_max_s": 1000.0}, 31)
    assert fcurto == f3                              # duração maior que o sinal não estica a janela


def test_sinal_so_ruido_nao_tem_fim():
    t = np.arange(2.5, 60.0, 1 / 15)
    y = np.random.default_rng(2).normal(0, 1.0, len(t))
    fim, _ = J.fim_da_regra(t, y, {"k": 3.0}, 31)
    assert fim != fim or fim < t[0] + 5               # nan ou sinal inexistente


def test_t50_de_exponencial_simples_conhecida():
    t = np.arange(2.5, 60.0, 1 / 15)
    y = 100 * np.exp(-(t - 2.5) / 10.0)
    assert J.t50_livre(t, y) == pytest.approx(10.0 * np.log(2), abs=0.15)
    assert np.isnan(J.t50_livre(t[:40], y[:40]))     # janela curta demais: não chega a 50%


def test_cv_duas_colunas_iguais_e_valor_conhecido():
    df = pd.DataFrame({"proporcao": ["a"] * 5 + ["b"] * 5, "amostra": ["A"] * 10,
                       "x": [10.0] * 5 + [9.0, 10.0, 11.0, 10.0, 10.0]})
    cv, n = J.cv_medio_por_grupo(df[df.proporcao == "a"], "x", 3)
    assert cv == 0 and n == 1
    cv_b, _ = J.cv_medio_por_grupo(df[df.proporcao == "b"], "x", 3)
    esperado = np.std([9, 10, 11, 10, 10], ddof=1) / 10
    assert cv_b == pytest.approx(esperado)
    assert np.isnan(J.cv_medio_por_grupo(df, "x", 6)[0])    # grupos com menos regravações que o mínimo saem


def _projeto(tmp_path, n_prop=2, n_vid=5):
    """Série R sintética: tau fixo por proporção; ruído conhecido; um vídeo cujo R some no ruído logo no início."""
    rng = np.random.default_rng(0)
    t = np.arange(0.5, 60.0, 1 / 15)
    linhas, vids = [], []
    for ip in range(n_prop):
        for iv in range(1, n_vid + 1):
            nome = f"P{ip}_v{iv}"
            y = _dupla(t - 0.5, 60.0, 2.0 + ip, 40.0, 12.0 + 4 * ip) + rng.normal(0, 0.05, len(t))
            linhas.append(pd.DataFrame({"video": nome, "amostra": "A", "grupo": "treino", "num_video": iv, "proporcao": f"p{ip}",
                                        "tempo_s": t, "corr_R": y}))
            vids.append({"video": nome, "usar_no_ajuste_R": True})
    y = 0.3 * np.exp(-(t - 0.5) / 1.0) + rng.normal(0, 0.05, len(t))      # some no ruído em poucos segundos
    linhas.append(pd.DataFrame({"video": "morto", "amostra": "A", "grupo": "treino", "num_video": 9, "proporcao": "p0", "tempo_s": t, "corr_R": y}))
    vids.append({"video": "morto", "usar_no_ajuste_R": True})
    pos = tmp_path / "output" / "posprocessamento"
    pos.mkdir(parents=True)
    pd.concat(linhas).to_csv(pos / "series_pos.csv", index=False)
    pd.DataFrame(vids).to_csv(pos / "videos_pos.csv", index=False)
    return {"pastas": {"saida": str(tmp_path / "output")}, "modelagem": {"workers": 1, "c_livre_c_minimo": -1.35},
            "janela_r": {"regras": [{"k": 6}, {"k": 3, "duracao_max_s": 20}], "workers": 1}}


def test_rodar_ponta_a_ponta_recupera_tau_e_acusa_sinal_inajustavel(tmp_path):
    cfg = _projeto(tmp_path)
    assert J.rodar(cfg) == 0
    out = tmp_path / "output" / "janela_r"
    assert {"sinais.csv", "resumo_regras.csv", "relatorio_janela_r.txt"} <= {f.name for f in out.iterdir()}
    sin = pd.read_csv(out / "sinais.csv", encoding="utf-8-sig")
    res = pd.read_csv(out / "resumo_regras.csv", encoding="utf-8-sig").set_index("regra")
    assert list(res.index)[0] == "k3"
    # o vídeo cujo sinal some no ruído não é ajustável em nenhuma regra; os demais, sim
    assert not sin[sin.video == "morto"]["ajustavel"].any()
    assert res.loc["k3", "n_ajustaveis"] == len(sin[sin.regra == "k3"]) - 1
    assert res.loc["k3", "frac_nao_ajustavel"] == pytest.approx(1 / len(sin[sin.regra == "k3"]))
    # tau_medio verdadeiro de p0 na janela que começa 2 s depois do início do decaimento (0,5 s): amplitudes A_i*exp(-2/tau_i)
    a1, a2 = 60.0 * np.exp(-2.0 / 2.0), 40.0 * np.exp(-2.0 / 12.0)
    verdadeiro = (a1 * 2.0 + a2 * 12.0) / (a1 + a2)
    p0 = sin[(sin.regra == "k3") & (sin.proporcao == "p0") & sin.ajustavel]
    assert p0["c0_tau_medio"].median() == pytest.approx(verdadeiro, rel=0.05)
    # regra de base comparada a si mesma: mudança relativa zero e relatório calculado dos dados
    assert res.loc["k3", "c0_dtau_rel_vs_base"] == 0
    texto = (out / "relatorio_janela_r.txt").read_text(encoding="utf-8")
    assert "k3" in texto and "k6" in texto and "k3_d20s" in texto
    # k=6 termina antes ou igual a k=3 em todo sinal ajustável nos dois
    a = sin[sin.regra == "k3"].set_index("video")["fim_s"]
    b = sin[sin.regra == "k6"].set_index("video")["fim_s"]
    assert ((b.dropna() <= a.reindex(b.dropna().index) + 1e-9)).all()


def test_deriva_injetada_no_tau_e_detectada_pela_regra_de_recomendacao():
    base = {"regra": "k3", "frac_nao_ajustavel": 0.0, "cl_tau_cv_mediano": 0.10, "c0_tau_cv_mediano": 0.10, "disc_cl_c0_mediana": 0.3}
    melhor = {**base, "regra": "k5", "cl_tau_cv_mediano": 0.05, "c0_tau_cv_mediano": 0.06, "disc_cl_c0_mediana": 0.2}
    igual = {**base, "regra": "k4"}
    pior_aj = {**melhor, "regra": "k6", "frac_nao_ajustavel": 0.30}
    crit = {"max_aumento_nao_ajustavel": 0.02, "melhora_cv_minima": 0.10, "max_piora_discrepancia": 0.0}
    rec, expl = J.recomendar(pd.DataFrame([base, igual, melhor, pior_aj]), "k3", crit)
    assert rec == "k5" and len(expl) == 3
    assert "ajustar sinais demais" in [e for e in expl if "k6" in e][0]
    rec2, _ = J.recomendar(pd.DataFrame([base, igual]), "k3", crit)
    assert rec2 is None                                   # sem melhora, nada é recomendado


def test_conferencia_da_base_acusa_diferenca_injetada():
    sinais = pd.DataFrame({"video": ["a", "b"], "regra": ["k3", "k3"], "c0_tau_medio": [10.0, 10.0], "cl_tau_medio": [10.0, 10.0]})
    aj = pd.DataFrame({"video": ["a", "b"] * 2, "canal": "R", "caminho": ["principal"] * 2 + ["c_livre"] * 2,
                       "modelo": "exp_dupla", "tau_medio": [10.0, 11.0, 10.0, 10.0]})
    linhas = J.conferir_base(sinais, aj, "k3", "R")
    assert "9.09e-02" in linhas[0] and "0.00e+00" in linhas[1]    # deriva de 10% em b detectada no c0 (1/11 - ... = 9,1%)
