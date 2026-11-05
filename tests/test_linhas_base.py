"""Testes de linhas_base.py com dados SINTÉTICOS de resposta conhecida (I0 proporcional à fração => regra fixa erra 0)."""
import numpy as np
import pandas as pd
import pytest

from luminescencia import linhas_base as lb

PROPS = {"R1.5": (1.5, 0, 0), "G1.5": (0, 1.5, 0), "R0.75_G0.75": (0.75, 0.75, 0), "R0.5_G0.5_B0.5": (0.5, 0.5, 0.5)}
CFG = {"avaliacao": {"bootstrap_n": 200}, "linhas_base": {"proporcoes_sensibilidade": []}}


def _entradas(i0, sinal, auc=None):
    d = {}
    for j, c in enumerate("RGB"):
        d[f"{c}_md_I0"] = [row[j] for row in i0]
        d[f"{c}_md_auc_10s"] = [row[j] for row in (auc if auc is not None else i0)]
        d[f"sinal_presente_{c}"] = [row[j] for row in sinal]
    return pd.DataFrame(d)


def test_media_e_a_media_das_fracoes_do_treino():
    f = pd.DataFrame({"f_R": [1.0, 0.0, 0.5], "f_G": [0.0, 1.0, 0.5], "f_B": [0.0, 0.0, 0.0]})
    p = lb.prever_media(f, 4)
    assert p.shape == (4, 3) and np.allclose(p, [[0.5, 0.5, 0.0]] * 4)


def test_regra_fixa_normaliza_para_somar_um_valor_conhecido():
    e = _entradas([(100.0, 50.0, 50.0), (30.0, 10.0, 60.0)], [(True, True, True)] * 2)
    p, motivo = lb.prever_regra_fixa(e, "I0")
    assert np.allclose(p.to_numpy(), [[0.5, 0.25, 0.25], [0.3, 0.1, 0.6]]) and (motivo == "").all()
    assert np.allclose(p.sum(axis=1), 1.0)


def test_variante_auc_usa_a_coluna_auc_nao_a_i0():
    e = _entradas([(100.0, 50.0, 50.0)], [(True, True, True)], auc=[(10.0, 10.0, 20.0)])
    assert np.allclose(lb.prever_regra_fixa(e, "auc_10s")[0].to_numpy(), [[0.25, 0.25, 0.5]])
    assert np.allclose(lb.prever_regra_fixa(e, "I0")[0].to_numpy(), [[0.5, 0.25, 0.25]])


def test_canal_sem_sinal_medido_vale_zero_mesmo_com_nan():
    e = _entradas([(80.0, np.nan, np.nan)], [(True, False, False)])               # como R1.5: não há G nem B
    p, motivo = lb.prever_regra_fixa(e, "I0")
    assert np.allclose(p.to_numpy(), [[1.0, 0.0, 0.0]]) and motivo.iloc[0] == ""


def test_sinal_presente_com_grandeza_ausente_negativa_ou_soma_zero_fica_sem_previsao():
    e = _entradas([(100.0, np.nan, 10.0), (100.0, -5.0, 10.0), (0.0, 0.0, 0.0), (10.0, 10.0, 10.0)],
                  [(True, True, True), (True, True, True), (True, True, True), (True, True, True)])
    p, motivo = lb.prever_regra_fixa(e, "I0")
    assert p.iloc[:3].isna().all(axis=None) and not p.iloc[3].isna().any()          # nada de imputar
    assert "ausente" in motivo.iloc[0] and "negativo" in motivo.iloc[1] and "zero" in motivo.iloc[2] and motivo.iloc[3] == ""


def test_sinal_presente_desconhecido_nao_vira_true():
    e = _entradas([(1.0, 1.0, 1.0)], [(True, True, True)])
    e["sinal_presente_G"] = [np.nan]
    p, motivo = lb.prever_regra_fixa(e, "I0")
    assert p.isna().all(axis=None) and "sinal_presente_G ausente" in motivo.iloc[0]


def test_regra_fixa_ignora_colunas_prop_e_falha_se_faltar_coluna():
    e = _entradas([(100.0, 50.0, 50.0)], [(True, True, True)])
    antes = lb.prever_regra_fixa(e, "I0")[0]
    e["prop_R"], e["prop_G"], e["prop_B"] = 1.5, 0.0, 0.0                              # rótulo cheio de informação: não pode influenciar
    assert lb.prever_regra_fixa(e, "I0")[0].equals(antes)
    with pytest.raises(ValueError, match="colunas ausentes"):
        lb.prever_regra_fixa(e.drop(columns="R_md_I0"), "I0")


def test_config_recusa_canais_sem_po_false():
    with pytest.raises(ValueError, match="canais_sem_po"):
        lb.parametros({"linhas_base": {"entradas": {"canais_sem_po": False}}})
    assert lb.parametros({})["entradas"]["caminhos"] == ["c0"] and lb.parametros({})["regra_fixa"]["grandezas"][0] == "I0"


# --------------------------------------------------------------------------------------- avaliação ponta a ponta (sintética)

def _tabelas(i0_c_extra=None):
    """dataset + descritores sintéticos: A, B e C; I0 = 200 x fração (regra fixa perfeita); 3 regravações por preparação."""
    d_lin, s_lin = [], []
    for pr, (r, g, b) in PROPS.items():
        f = np.array([r, g, b]) / 1.5
        for am in "ABC":
            for k in range(3):
                nome = f"{pr}_{am}_{k}"
                d_lin.append({"video": nome, "amostra": am, "grupo": am, "num_video": k, "proporcao": pr, "prop_R": r, "prop_G": g, "prop_B": b,
                              "excluido_saturacao": False, "sat_ate_s": 0.0, "canal_ativo_R": r > 0, "canal_ativo_G": g > 0, "canal_ativo_B": b > 0,
                              **{f"sinal_presente_{c}": bool(f[j] > 0) for j, c in enumerate("RGB")},
                              "R_c0_tau_medio": 10.0 + k, "R_cl_tau_medio": 11.0, "R_c0_ok": True})
                s = {"video": nome, "amostra": am, "grupo": am, "num_video": k, "proporcao": pr}
                for j, c in enumerate("RGB"):
                    s[f"{c}_md_I0"] = 200.0 * f[j] if f[j] > 0 else np.nan
                    s[f"{c}_md_auc_10s"] = 50.0 * f[j] if f[j] > 0 else np.nan
                if am == "C" and i0_c_extra is not None:
                    s["R_md_I0"] = i0_c_extra
                s_lin.append(s)
    dic = pd.DataFrame({"coluna": list(d_lin[0]), "papel": [
        "meta", "meta", "meta", "meta", "meta", "alvo", "alvo", "alvo", "flag", "flag", "flag", "flag", "flag", "flag", "flag", "flag",
        "feature", "feature", "flag"]})
    dd = pd.DataFrame({"coluna": list(s_lin[0]), "papel": ["meta"] * 5 + ["feature"] * 6})
    return pd.DataFrame(d_lin), pd.DataFrame(s_lin), dic, dd


def _gravar(tmp_path, i0_c_extra=None, prop_c_extra=None):
    df, desc, dic, dd = _tabelas(i0_c_extra)
    if prop_c_extra is not None:                                             # estraga os alvos de C (jamais devem ser lidos nas métricas)
        df.loc[df.amostra == "C", ["prop_R", "prop_G", "prop_B"]] = prop_c_extra
    pasta = tmp_path / "output" / "tabela"
    pasta.mkdir(parents=True, exist_ok=True)
    df.to_csv(pasta / "dataset.csv", index=False, encoding="utf-8-sig")
    desc.to_csv(pasta / "descritores.csv", index=False, encoding="utf-8-sig")
    dic.to_csv(pasta / "dicionario.csv", index=False, encoding="utf-8-sig")
    dd.to_csv(pasta / "dicionario_descritores.csv", index=False, encoding="utf-8-sig")
    cfg = {"pastas": {"saida": str(tmp_path / "output")}, "avaliacao": {"bootstrap_n": 200},
           "linhas_base": {"proporcoes_sensibilidade": ["R1.5"]}}
    conf = tmp_path / "config.yaml"
    conf.write_text("avaliacao:\n  bootstrap_n: 200\n", encoding="utf-8")
    return cfg, conf


def test_ponta_a_ponta_regra_perfeita_erra_zero_e_media_erra_o_esperado(tmp_path):
    cfg, conf = _gravar(tmp_path)
    assert lb.rodar(cfg, config_path=conf) == 0
    pasta = tmp_path / "output" / "avaliacao"
    t = pd.read_csv(pasta / "avaliacao_por_esquema.csv")
    f = np.array(list(PROPS.values())) / 1.5
    media_a = f.mean(axis=0)                                                 # A e B têm as mesmas 4 proporções: média do treino conhecida
    esperado = np.mean([np.abs(fi - media_a).mean() for fi in f])
    a_b = t[(t.esquema == "a_b") & (t.subconjunto == "todas")].set_index("linha_base")
    assert a_b.loc["media", "mae_medio"] == pytest.approx(esperado)
    assert a_b.loc["regra_fixa_I0", "mae_medio"] == pytest.approx(0.0, abs=1e-12)
    assert a_b.loc["regra_fixa_auc_10s", "mae_medio"] == pytest.approx(0.0, abs=1e-12)
    assert a_b.loc["regra_fixa_I0", "razao_a_media"] == pytest.approx(0.0, abs=1e-12)
    assert a_b.loc["media", "n_preparacoes"] == 8                           # 4 proporções x A, B; nada de C
    sem = t[(t.esquema == "a_b") & (t.subconjunto == "sem R1.5")]
    assert (sem.n_preparacoes == 6).all()
    texto = (pasta / "relatorio_avaliacao.txt").read_text(encoding="utf-8")
    assert f"a média erra {esperado:.4f}" in texto and "a regra fixa (I0) erra 0.0000" in texto and "piso do critério D3(c)" in texto
    assert "luminescencia" in texto and "versao_codigo" in texto and b"\r" not in (pasta / "relatorio_avaliacao.txt").read_bytes()
    g = pd.read_csv(pasta / "ganho_pareado.csv")
    ganho = g[(g.esquema == "a_b") & (g.subconjunto == "todas") & (g.comparada == "regra_fixa_I0")].ganho_medio.to_numpy()
    assert len(ganho) == 1 and np.allclose(ganho, esperado)


def test_c_nunca_entra_nem_muda_o_resultado(tmp_path):
    cfg, conf = _gravar(tmp_path)
    lb.rodar(cfg, config_path=conf)
    pasta = tmp_path / "output" / "avaliacao"
    base = {n: (pasta / n).read_bytes() for n in ("avaliacao_por_esquema.csv", "erros_por_preparacao.csv", "mae_por_proporcao.csv",
                                                    "ganho_pareado.csv", "dobras.csv")}
    assert "|C" not in base["erros_por_preparacao.csv"].decode("utf-8")
    outro = tmp_path / "outro"
    cfg2, conf2 = _gravar(outro, i0_c_extra=99999.0, prop_c_extra=(0.0, 0.0, 1.5))      # C totalmente diferente
    lb.rodar(cfg2, config_path=conf2)
    for n, b in base.items():
        assert (outro / "output" / "avaliacao" / n).read_bytes() == b, n


def test_saturados_ficam_fora_e_video_sem_previsao_e_listado(tmp_path):
    cfg, conf = _gravar(tmp_path)
    pasta_t = tmp_path / "output" / "tabela"
    df = pd.read_csv(pasta_t / "dataset.csv", encoding="utf-8-sig")
    df.loc[df.video == "G1.5_A_0", "excluido_saturacao"] = True
    df.to_csv(pasta_t / "dataset.csv", index=False, encoding="utf-8-sig")
    desc = pd.read_csv(pasta_t / "descritores.csv", encoding="utf-8-sig")
    desc.loc[desc.video == "R0.75_G0.75_B_1", "R_md_I0"] = np.nan                # sinal presente mas I0 ausente: não imputar
    desc.to_csv(pasta_t / "descritores.csv", index=False, encoding="utf-8-sig")
    lb.rodar(cfg, config_path=conf)
    pasta = tmp_path / "output" / "avaliacao"
    sem = pd.read_csv(pasta / "sem_previsao.csv")
    assert list(sem.video) == ["R0.75_G0.75_B_1"] and "não imputado" in sem.motivo.iloc[0]
    erros = pd.read_csv(pasta / "erros_por_preparacao.csv")
    nv = erros[(erros.esquema == "a_b") & (erros.linha_base == "media") & (erros.preparacao == "G1.5|A")].n_videos.iloc[0]
    assert nv == 2                                                              # o saturado saiu
    nv = erros[(erros.esquema == "a_b") & (erros.linha_base == "media") & (erros.preparacao == "R0.75_G0.75|B")].n_videos.iloc[0]
    assert nv == 2                                                              # o sem previsão saiu de TODAS as linhas (universo comum)


def test_entradas_com_coluna_de_vazamento_sao_recusadas():
    df, desc, dic, dd = _tabelas()
    d = df[df.amostra != "C"].reset_index(drop=True)
    e = _entradas([(1.0, 1.0, 1.0)] * len(d), [(True, True, True)] * len(d))
    e["prop_R"] = d["prop_R"]
    with pytest.raises(ValueError, match="proibidas"):
        lb.avaliar(d, e, CFG)
