"""Testes de avaliacao.py com dados SINTÉTICOS de resposta conhecida: alvo, dobras sem vazamento, métricas, bootstrap e trava do C."""
import numpy as np
import pandas as pd
import pytest

from luminescencia import avaliacao as av
from luminescencia.avaliacao import Dobra, VazamentoDeDobras

PROPS = {"R1.5": (1.5, 0, 0), "G1.5": (0, 1.5, 0), "B1.5": (0, 0, 1.5), "R0.75_G0.75": (0.75, 0.75, 0), "X_so_B": (0.5, 0.5, 0.5)}


def _df(com_c=True, sat=()):
    """5 proporções x amostras A, B (e C) x 3 regravações. 'X_so_B' só existe em B (como G0.5_B1, saturada em A)."""
    linhas = []
    for pr, (r, g, b) in PROPS.items():
        for am in ("A", "B") + (("C",) if com_c else ()):
            if pr == "X_so_B" and am == "A":
                continue
            for k in range(3):
                linhas.append({"video": f"{pr}_{am}_{k}", "amostra": am, "proporcao": pr, "prop_R": r, "prop_G": g, "prop_B": b,
                               "excluido_saturacao": False})
    df = pd.DataFrame(linhas)
    for v in sat:
        df.loc[df.video == v, "excluido_saturacao"] = True
    return df


CFG = {}


# ------------------------------------------------------------------------------------------------------------- alvo

def test_fracoes_alvo_somam_um_e_valem_prop_sobre_1_5():
    f = av.fracoes_alvo(_df())
    assert list(f.columns) == ["f_R", "f_G", "f_B"]
    assert np.allclose(f.sum(axis=1), 1.0)
    linha = _df().assign(i=lambda d: range(len(d)))
    i = linha.index[linha.video == "R0.75_G0.75_A_0"][0]
    assert np.allclose(f.loc[i].to_numpy(), [0.5, 0.5, 0.0])


def test_fracoes_alvo_recusa_soma_errada_e_coluna_ausente():
    df = _df()
    df.loc[0, "prop_R"] = 1.0
    with pytest.raises(ValueError, match="não somam 1"):
        av.fracoes_alvo(df)
    with pytest.raises(ValueError, match="ausentes"):
        av.fracoes_alvo(_df().drop(columns="prop_G"))


def test_alvo_nunca_e_entrada():
    with pytest.raises(ValueError, match="proibidas"):
        av.validar_colunas_de_entrada(["R_md_I0", "f_R"])
    for ruim in ("prop_G", "canal_ativo_R", "usar_no_ajuste_B", "sat_ate_s", "excluido_saturacao", "proporcao", "R_c0_r2"):
        with pytest.raises(ValueError, match="proibidas"):
            av.validar_colunas_de_entrada(["R_md_I0", ruim])
    av.validar_colunas_de_entrada(["R_md_I0", "razao_I0_R_G", "sinal_presente_R", "R_c0_tau_medio"])


# ------------------------------------------------------------------------------------------------------------ dobras

def test_universo_exclui_c_e_saturados():
    df = _df(sat=("B1.5_A_0",))
    m = av.mascara_ab(df, CFG)
    assert not m[df.amostra == "C"].any()
    assert not m[df.video == "B1.5_A_0"].any()
    assert m.sum() == ((df.amostra != "C") & (df.video != "B1.5_A_0")).sum()


@pytest.mark.parametrize("esquema", av.ESQUEMAS)
def test_nenhum_esquema_vaza_preparacao_nem_regravacao(esquema):
    df = _df(sat=("G1.5_B_2",))
    dobras = av.gerar_dobras(df, esquema, CFG)               # já chama a verificação
    assert dobras
    prep = av.chave_preparacao(df).to_numpy()
    for d in dobras:
        assert not set(d.treino.tolist()) & set(d.validacao.tolist())
        assert not set(prep[d.treino]) & set(prep[d.validacao])
        assert not (df.amostra.to_numpy()[np.r_[d.treino, d.validacao]] == "C").any()          # nada de C
        assert not df.excluido_saturacao.to_numpy()[np.r_[d.treino, d.validacao]].any()        # nada de saturado
        for pr in set(prep[d.validacao]):                      # todas as regravações da preparação do mesmo lado
            assert set(np.flatnonzero(prep == pr)) - set(np.flatnonzero(df.excluido_saturacao.to_numpy())) <= set(d.validacao.tolist())


def test_a_b_treina_numa_e_valida_na_outra_e_lista_proporcao_sem_treino():
    df = _df()
    d1, d2 = av.gerar_dobras(df, "a_b", CFG)
    assert (df.amostra.to_numpy()[d1.treino] == "A").all() and (df.amostra.to_numpy()[d1.validacao] == "B").all()
    assert (df.amostra.to_numpy()[d2.treino] == "B").all() and (df.amostra.to_numpy()[d2.validacao] == "A").all()
    desc = av.descrever_dobras(df, [d1, d2]).set_index("dobra")
    assert desc.loc["A→B", "proporcoes_sem_treino"] == "X_so_B"          # nunca está em A
    assert desc.loc["B→A", "proporcoes_sem_treino"] == "-"
    assert desc.loc["A→B", "n_videos_treino"] == 12 and desc.loc["A→B", "n_preparacoes_treino"] == 4
    assert desc.loc["A→B", "n_videos_validacao"] == 15 and desc.loc["A→B", "n_preparacoes_validacao"] == 5


def test_leave_one_proporcao_out_tira_a_proporcao_toda_e_diz_qual():
    df = _df()
    dobras = av.gerar_dobras(df, "leave_one_proporcao_out", CFG)
    assert len(dobras) == len(PROPS)
    desc = av.descrever_dobras(df, dobras).set_index("dobra")
    for pr in PROPS:
        assert desc.loc[pr, "proporcoes_sem_treino"] == pr
    d = next(d for d in dobras if d.nome == "R1.5")
    assert set(df.proporcao.to_numpy()[d.validacao]) == {"R1.5"} and "R1.5" not in set(df.proporcao.to_numpy()[d.treino])


def test_por_preparacao_tem_uma_dobra_por_preparacao():
    df = _df()
    dobras = av.gerar_dobras(df, "por_preparacao", CFG)
    assert len(dobras) == df[df.amostra != "C"].groupby(["proporcao", "amostra"]).ngroups
    assert all(len(set(av.chave_preparacao(df).to_numpy()[d.validacao])) == 1 for d in dobras)


def test_verificador_pega_vazamento_de_proposito():
    df = _df()
    ok = av.gerar_dobras(df, "a_b", CFG)[0]
    prep = av.chave_preparacao(df).to_numpy()
    # (1) uma regravação atravessa o lado: move 1 vídeo de uma preparação de validação para o treino
    v = ok.validacao[0]
    ruim = Dobra("a_b", "x", np.append(ok.treino, v), ok.validacao[1:])
    with pytest.raises(VazamentoDeDobras, match="preparação"):
        av.verificar_sem_vazamento(df, [ruim], CFG, cobertura_unica=False)
    # (2) o mesmo vídeo nos dois lados
    with pytest.raises(VazamentoDeDobras, match="mesmo vídeo"):
        av.verificar_sem_vazamento(df, [Dobra("a_b", "x", ok.treino, np.append(ok.validacao, ok.treino[0]))], CFG, cobertura_unica=False)
    # (3) C no treino
    c = np.flatnonzero(df.amostra.to_numpy() == "C")
    with pytest.raises(VazamentoDeDobras, match="fora de A\\+B"):
        av.verificar_sem_vazamento(df, [Dobra("a_b", "x", np.append(ok.treino, c[0]), ok.validacao)], CFG, cobertura_unica=False)
    # (4) saturado na validação
    df2 = _df(sat=("B1.5_B_0",))
    i = int(np.flatnonzero((df2.video == "B1.5_B_0").to_numpy())[0])
    with pytest.raises(VazamentoDeDobras, match="fora de A\\+B"):
        av.verificar_sem_vazamento(df2, [Dobra("a_b", "x", np.flatnonzero((df2.amostra == "A").to_numpy()), np.array([i]))], CFG, cobertura_unica=False)
    # (5) vídeo de uma preparação some da dobra (regravação perdida do lado) também é vazamento de unidade
    prep_v = prep[ok.validacao[0]]
    parcial = np.array([i for i in ok.validacao if not (prep[i] == prep_v and i == ok.validacao[0])])
    with pytest.raises(VazamentoDeDobras, match="cruza o lado"):
        av.verificar_sem_vazamento(df, [Dobra("a_b", "x", ok.treino, parcial)], CFG, cobertura_unica=False)


# ----------------------------------------------------------------------------------------------------------- métricas

def _tab(erros_por_prop):
    """Tabela por preparação com erro conhecido: {proporcao: [erro_prep1, erro_prep2,...]}; mesmo erro nas 3 frações."""
    linhas = []
    for pr, es in erros_por_prop.items():
        for k, e in enumerate(es):
            linhas.append({"preparacao": f"{pr}|{k}", "proporcao": pr, "amostra": str(k), "n_videos": 5,
                           "erro_R": e, "erro_G": e, "erro_B": e, "erro_medio": e})
    return pd.DataFrame(linhas).set_index("preparacao")


def test_erros_por_preparacao_valor_conhecido():
    df = _df(com_c=False)
    f = av.fracoes_alvo(df)
    pred = f.to_numpy().copy()
    pred[:, 0] += 0.1
    pred[:, 2] -= 0.1                                         # erros por vídeo: R 0,1; G 0; B 0,1
    tab = av.erros_por_preparacao(df, f, pred)
    m = av.mae_por_fracao(tab)
    assert m["R"] == pytest.approx(0.1) and m["G"] == pytest.approx(0.0) and m["B"] == pytest.approx(0.1)
    assert m["medio"] == pytest.approx(0.2 / 3)
    assert (tab["n_videos"] > 0).all() and len(tab) == df.groupby(["proporcao", "amostra"]).ngroups


def test_previsao_perfeita_tem_erro_zero_e_nan_e_recusado():
    df = _df(com_c=False)
    f = av.fracoes_alvo(df)
    assert av.mae_por_fracao(av.erros_por_preparacao(df, f, f.to_numpy()))["medio"] == 0.0
    ruim = f.to_numpy().copy()
    ruim[0, 0] = np.nan
    with pytest.raises(ValueError, match="ausentes"):
        av.erros_por_preparacao(df, f, ruim)


def test_preparacao_conta_uma_vez_nao_o_numero_de_videos():
    df = pd.DataFrame({"video": list("abcd"), "amostra": ["A"] * 4, "proporcao": ["P1", "P1", "P1", "P2"]})
    f = pd.DataFrame({"f_R": [1.0] * 4, "f_G": [0.0] * 4, "f_B": [0.0] * 4})
    pred = np.array([[0.9, 0.1, 0.0]] * 3 + [[0.5, 0.5, 0.0]])        # erro por vídeo: P1 = 0,2/3 (R 0,1 G 0,1); P2 = R 0,5 G 0,5
    tab = av.erros_por_preparacao(df, f, pred)
    assert tab.loc["P1|A", "erro_medio"] == pytest.approx(0.2 / 3) and tab.loc["P2|A", "erro_medio"] == pytest.approx(1.0 / 3)
    assert av.mae_por_fracao(tab)["medio"] == pytest.approx((0.2 / 3 + 1.0 / 3) / 2)      # média de 2 preparações, não de 4 vídeos


def test_mae_por_proporcao_pior_proporcao_e_razao():
    modelo = _tab({"A": [0.1, 0.3], "B": [0.2], "C": [0.6]})
    media = _tab({"A": [0.2, 0.2], "B": [0.4], "C": [0.5]})
    por = av.mae_por_proporcao(modelo)
    assert por.loc["A", "erro_medio"] == pytest.approx(0.2) and por.loc["A", "n_preparacoes"] == 2
    assert av.pior_proporcao(modelo) == ("C", pytest.approx(0.6))
    assert av.razao_ao_erro_da_media(modelo, media) == pytest.approx(((0.1 + 0.3 + 0.2 + 0.6) / 4) / ((0.2 + 0.2 + 0.4 + 0.5) / 4))


def test_ganho_pareado_constante_ic_degenerado_e_exclui_zero():
    ref = _tab({"A": [0.30, 0.30], "B": [0.30, 0.30], "C": [0.30, 0.30]})
    novo = ref.assign(erro_medio=0.25)
    g = av.ganho_pareado(ref, novo, CFG)
    assert g["ganho_medio"] == pytest.approx(0.05) and g["ic_inf"] == pytest.approx(0.05) and g["ic_sup"] == pytest.approx(0.05)
    assert g["exclui_zero"] and g["n_unidades"] == 6
    gp = av.ganho_pareado(ref, novo, CFG, por="proporcao")
    assert gp["n_unidades"] == 3 and gp["ganho_medio"] == pytest.approx(0.05)


def test_ganho_pareado_simetrico_inclui_zero_e_ic_largo():
    ref = _tab({"A": [0.2, 0.2, 0.2, 0.2], "B": [0.2, 0.2, 0.2, 0.2]})
    novo = ref.copy()
    novo["erro_medio"] = [0.1, 0.3, 0.1, 0.3, 0.1, 0.3, 0.1, 0.3]       # ganho +0,1 / -0,1 alternado: média 0
    g = av.ganho_pareado(ref, novo, CFG)
    assert g["ganho_medio"] == pytest.approx(0.0, abs=1e-12) and not g["exclui_zero"] and g["ic_inf"] < 0 < g["ic_sup"]


def test_bootstrap_semente_fixa_reproduz_e_semente_diferente_muda():
    ref = _tab({"A": [0.2, 0.25, 0.3], "B": [0.1, 0.4, 0.2]})
    novo = ref.assign(erro_medio=[0.18, 0.2, 0.31, 0.12, 0.3, 0.19])
    a = av.ganho_pareado(ref, novo, CFG)
    b = av.ganho_pareado(ref, novo, CFG)
    assert a == b
    c = av.ganho_pareado(ref, novo, {"avaliacao": {"semente": 7}})
    assert (c["ic_inf"], c["ic_sup"]) != (a["ic_inf"], a["ic_sup"]) and c["ganho_medio"] == a["ganho_medio"]


def test_bootstrap_reamostra_preparacoes_nao_videos():
    # duas preparações com número de vídeos muito diferente: o IC do MAE só pode depender dos 2 valores por preparação
    t = _tab({"A": [0.1], "B": [0.5]})
    t.loc["A|0", "n_videos"] = 1
    t.loc["B|0", "n_videos"] = 100
    lo, hi = av.ic_bootstrap_mae(t, {"avaliacao": {"bootstrap_n": 2000}})
    assert lo == pytest.approx(0.1) and hi == pytest.approx(0.5)             # médias possíveis: 0,1 / 0,3 / 0,5
    r = av.ic_bootstrap_razao(t, t.assign(erro_medio=[0.2, 1.0]), CFG)
    assert r == (pytest.approx(0.5), pytest.approx(0.5))                     # razão constante = 0,5


def test_ganho_recusa_unidades_diferentes():
    with pytest.raises(ValueError, match="mesmas unidades"):
        av.ganho_pareado(_tab({"A": [0.1]}), _tab({"B": [0.1]}), CFG)


# ------------------------------------------------------------------------------------------------------- trava do C

def _tabela_em(tmp_path, df):
    pasta = tmp_path / "output" / "tabela"
    pasta.mkdir(parents=True)
    df.to_csv(pasta / "dataset.csv", index=False, encoding="utf-8-sig")
    for nome in ("descritores.csv", "dicionario.csv", "dicionario_descritores.csv"):
        (pasta / nome).write_text("coluna,papel\nx,feature\n", encoding="utf-8")
    return {"pastas": {"saida": str(tmp_path / "output")}, "preregistro": {"arquivo_json": str(tmp_path / "docs" / "preregistro.json"),
            "arquivo_md": str(tmp_path / "docs" / "preregistro.md"), "secoes_config": ["avaliacao"]}, "avaliacao": {}}


def test_abrir_c_recusa_sem_preregistro(tmp_path):
    cfg = _tabela_em(tmp_path, _df())
    with pytest.raises(RuntimeError, match="travada"):
        av.abrir_c(cfg)


def test_carregar_ab_nunca_devolve_c(tmp_path):
    cfg = _tabela_em(tmp_path, _df(sat=("B1.5_A_0",)))
    df, pos = av.carregar_ab(cfg)
    assert set(df.amostra) == {"A", "B"} and "B1.5_A_0" not in set(df.video) and len(df) == len(pos)
