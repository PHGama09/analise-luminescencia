"""Testes da correção da luz de fundo (luz.py) com dados sintéticos de resposta conhecida.

O gerador põe ln(grandeza) = ln(base da preparação) + B_VERDADE * (canto - 2) + ruído pequeno nas amostras A e B, e uma relação
absurda na C (que não pode influenciar a calibração).
"""
import numpy as np
import pandas as pd
import pytest

from luminescencia import entradas as en
from luminescencia import luz

B_VERDADE = 0.05                    # +5% por nível de canto
P = {**luz.PADROES, "grandezas_canal": ["c0_tau_medio", "md_t50"], "razoes": [], "min_videos": 10, "min_preparacoes": 3}


def _dados(seed=1, b_c=0.50):
    rng = np.random.default_rng(seed)
    ds, de, vid = [], [], []
    for k, prop in enumerate(["P1", "P2", "P3", "P4"]):
        for am in "ABC":
            base = 5.0 + k
            for n in range(5):
                nome = f"{prop}_{am}{n}"
                canto = rng.uniform(1.0, 4.0)
                b = b_c if am == "C" else B_VERDADE
                f = np.exp(b * (canto - 2.0) + rng.normal(0, 0.002))
                ds.append({"video": nome, "amostra": am, "proporcao": prop, "excluido_saturacao": False,
                           "sinal_presente_R": True, "sinal_presente_G": False, "sinal_presente_B": False,
                           "R_c0_tau_medio": base * 2 * f, "R_md_I0": 100.0 + n})
                de.append({"video": nome, "R_md_t50": base * f, "R_md_I0": 100.0 + n})
                vid.append({"video": nome, "piso_canto_R": canto, "piso_canto_G": canto, "piso_canto_B": canto})
    ds = pd.DataFrame(ds)
    ds.loc[0, ["excluido_saturacao", "R_c0_tau_medio"]] = [True, 9999.0]          # saturado com valor absurdo: fora da calibração
    return ds, pd.DataFrame(de), pd.DataFrame(vid)


def _tabela(**kw):
    ds, de, vid = _dados(**kw)
    return luz.montar_tabela(ds, de, vid, P)


def test_inclinacao_recupera_a_verdade_so_com_treino():
    t, cols = _tabela()
    assert set(cols) == {"R_c0_tau_medio", "R_md_t50"}
    for col in cols:
        est = luz.estimar_inclinacao(t, col, ["A", "B"], P)
        assert est["status"] == "estimada" and est["b"] == pytest.approx(B_VERDADE, abs=0.003)


def test_amostra_c_nao_muda_a_calibracao():
    t1, _ = _tabela(b_c=0.50)
    t2, _ = _tabela(b_c=-3.0)
    b1 = luz.estimar_inclinacao(t1, "R_md_t50", ["A", "B"], P)["b"]
    b2 = luz.estimar_inclinacao(t2, "R_md_t50", ["A", "B"], P)["b"]
    assert b1 == pytest.approx(b2, abs=1e-12)


def test_correcao_remove_a_dependencia_do_canto():
    t, _ = _tabela()
    b = luz.estimar_inclinacao(t, "R_md_t50", ["A", "B"], P)["b"]
    s = t[t["amostra"].isin(["A", "B"])]
    corr = luz.aplicar(s, "R_md_t50", b, 2.0)
    dp = lambda v: (np.log(v) - np.log(v).groupby(s["prep"]).transform("mean")).std()
    assert dp(corr) < 0.2 * dp(s["R_md_t50"])
    esperado = s["proporcao"].map({"P1": 5.0, "P2": 6.0, "P3": 7.0, "P4": 8.0})
    assert np.allclose(corr, esperado, rtol=0.01)                     # com canto_ref = 2, volta à base da preparação


def test_valor_nao_positivo_ou_sem_canto_e_copiado():
    t = pd.DataFrame({"x": [0.0, -1.0, np.nan, 2.0, 3.0], "canto_luz": [3.0, 3.0, 3.0, np.nan, 3.0]})
    v = luz.aplicar(t, "x", 0.1, 2.0)
    assert v.iloc[:2].tolist() == [0.0, -1.0] and np.isnan(v.iloc[2]) and v.iloc[3] == 2.0
    assert v.iloc[4] == pytest.approx(3.0 * np.exp(-0.1))


def test_poucos_dados_nao_estima():
    t, _ = _tabela()
    est = luz.estimar_inclinacao(t, "R_md_t50", ["A", "B"], {**P, "min_videos": 1000})
    assert est["b"] == 0.0 and est["status"].startswith("não estimada")


def test_sem_sinal_medido_nao_entra_na_calibracao():
    t, _ = _tabela()
    t["sinal_presente_R"] = False
    assert luz.estimar_inclinacao(t, "R_md_t50", ["A", "B"], P)["n_videos"] == 0


def test_validacao_cruzada_a_b_mostra_queda():
    t, cols = _tabela()
    cv = luz.validacao_cruzada(t, cols, ["A", "B"], P)
    assert len(cv) == 4 and set(cv["testada_em"]) == {"A", "B"}
    assert (cv["dp_com"] < cv["dp_sem"]).all() and (cv["rho_com"].abs() < cv["rho_sem"].abs()).all()


def test_rodar_grava_colunas_paralelas_sem_mexer_no_dataset(tmp_path):
    ds, de, vid = _dados()
    tab, cons = tmp_path / "tabela", tmp_path / "consolidacao"
    tab.mkdir()
    cons.mkdir()
    ds.to_csv(tab / "dataset.csv", index=False, encoding="utf-8-sig")
    de.to_csv(tab / "descritores.csv", index=False, encoding="utf-8-sig")
    vid.to_csv(cons / "videos.csv", index=False, encoding="utf-8-sig")
    antes = (tab / "dataset.csv").read_bytes()
    cfg = {"pastas": {"saida": str(tmp_path)}, "divisao": {"treino": ["A", "B"]},
           "luz": {"grandezas_canal": ["c0_tau_medio", "md_t50"], "razoes": [], "min_videos": 10, "min_preparacoes": 3}}
    assert luz.rodar(cfg) == 0
    assert (tab / "dataset.csv").read_bytes() == antes
    lz = pd.read_csv(tab / "luz.csv", encoding="utf-8-sig")
    assert list(lz.columns) == ["video", "canto_luz", "R_c0_tau_medio", "R_md_t50"] and len(lz) == len(ds)
    assert "R_md_I0" not in lz.columns                                 # I0 não é corrigido
    inc = pd.read_csv(tmp_path / "luz" / "inclinacoes.csv", encoding="utf-8-sig")
    assert inc.set_index("coluna").loc["R_md_t50", "b"] == pytest.approx(B_VERDADE, abs=0.003)
    texto = (tmp_path / "luz" / "relatorio_luz.txt").read_text(encoding="utf-8")
    assert "C não entra" in texto and "Validação cruzada" in texto


def test_entradas_troca_pelas_colunas_corrigidas_so_com_a_chave():
    base = pd.DataFrame({"video": ["a", "b"], "R_md_t50": [5.0, 6.0], "R_md_I0": [1.0, 2.0]})
    lz = pd.DataFrame({"video": ["b", "a"], "canto_luz": [2.0, 3.0], "R_md_t50": [6.5, 4.5]})
    novo = en.trocar_por_luz(base, lz)
    assert novo["R_md_t50"].tolist() == [4.5, 6.5] and novo["R_md_I0"].tolist() == [1.0, 2.0]
    assert "canto_luz" not in novo.columns                             # o canto nunca vira entrada
    with pytest.raises(ValueError, match="sem linha em luz.csv"):
        en.trocar_por_luz(base, lz.iloc[:1])
    assert en.parametros({})["corrigir_luz"] is False
    with pytest.raises(ValueError, match="corrigir_luz"):
        en.parametros({"entradas": {"corrigir_luz": "sim"}})


def test_montar_entradas_exige_luz_quando_ligada():
    df = pd.DataFrame({"video": ["a"], "R_c0_tau_medio": [5.0]})
    dic = pd.DataFrame({"coluna": ["video", "R_c0_tau_medio"], "papel": ["meta", "feature"]})
    cfg = {"entradas": {"caminhos": ["c0"], "corrigir_luz": True}}
    with pytest.raises(ValueError, match="luz.csv"):
        en.montar_entradas(df, dic, cfg)
    lz = pd.DataFrame({"video": ["a"], "canto_luz": [3.0], "R_c0_tau_medio": [4.0]})
    assert en.montar_entradas(df, dic, cfg, luz=lz)["R_c0_tau_medio"].tolist() == [4.0]
    cfg["entradas"]["corrigir_luz"] = False
    assert en.montar_entradas(df, dic, cfg, luz=lz)["R_c0_tau_medio"].tolist() == [5.0]
