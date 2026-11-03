"""Testes da consolidação: quem entra, correção do preto, nomes aprovados. Rode: pytest"""
import pandas as pd

from luminescencia.consolidacao import avaliar_correcao, corrigir_preto, selecionar

P = {"piso": "video", "incluir_pendentes_ok": True}


def _resumo():
    linhas = [
        dict(chave="c0", caminho="Amostra_referencia.avi", controle=True, incluir="pendente", status_auditoria="ok"),
        dict(chave="k1", caminho="Amostras A/1/R_15_v1a.avi", controle=False, incluir="pendente", status_auditoria="ok"),
        dict(chave="k2", caminho="Amostras A/1/G_15_v1a.avi", controle=False, incluir="pendente", status_auditoria="ok"),
        dict(chave="k3", caminho="Amostras C/1/RGB_05_05_05_v1a.avi", controle=False, incluir="pendente", status_auditoria="revisar"),
        dict(chave="k4", caminho="Amostras B/3/RG_075_075_v3b.avi", controle=False, incluir="pendente", status_auditoria="fora_do_padrao"),
        dict(chave="k5", caminho="Amostras C/3/RGB_05_05_05v3c.avi", controle=False, incluir="pendente", status_auditoria="revisar"),
    ]
    return pd.DataFrame(linhas)


def test_selecao_respeita_a_curadoria():
    cur = {"k2": {"incluir": "nao", "motivo": "qualidade_ruim"},
           "k4": {"incluir": "nao", "motivo": "erro_de_gravacao"},
           "k5": {"incluir": "sim", "nome_aprovado": "RGB_05_05_05_v3c.avi", "status_auditoria": "revisar"}}
    aceitos, fora = selecionar(_resumo(), cur, P)
    assert sorted(aceitos["chave"]) == ["k1", "k5"]
    assert aceitos.set_index("chave").loc["k5", "nome_aprovado"] == "RGB_05_05_05_v3c.avi"
    motivos = {f["caminho"]: f["motivo"] for f in fora}
    assert "controle" in motivos["Amostra_referencia.avi"]
    assert "decidir" in motivos["Amostras C/1/RGB_05_05_05_v1a.avi"]      # pendente com alerta não entra
    assert "erro_de_gravacao" in motivos["Amostras B/3/RG_075_075_v3b.avi"]


def test_pendente_ok_pode_ser_desligado():
    aceitos, _ = selecionar(_resumo(), {}, {**P, "incluir_pendentes_ok": False})
    assert aceitos.empty


def _df(ini, fin, canto, ref, prop=(0, 1, 0)):
    d = {}
    for i, c in enumerate("RGB"):
        d.update({f"inicio_{c}": 100.0, f"final_{c}": fin, f"piso_canto_{c}": canto, f"piso_ref_{c}": ref,
                  f"prop_{c}": float(prop[i])})
    return pd.DataFrame([d])


def test_correcao_do_preto_por_video():
    controle = pd.Series({**{f"piso_canto_{c}": 2.5 for c in "RGB"}, **{f"piso_ref_{c}": 1.6 for c in "RGB"}})
    df = corrigir_preto(_df(100, 10, canto=4.0, ref=1.6), controle, "video")
    # offset canto-ROI do controle = 0,9; piso do vídeo = 4,0 - 0,9 = 3,1
    assert df.loc[0, "piso_video_G"] == 3.1
    assert df.loc[0, "final_corr_G"] == 6.9
    assert df.loc[0, "final_G"] == 10                   # o bruto não muda
    assert df.loc[0, "dif_piso_G"] == 1.5


def test_modo_referencia_usa_o_piso_fixo():
    controle = pd.Series({**{f"piso_canto_{c}": 2.5 for c in "RGB"}, **{f"piso_ref_{c}": 1.6 for c in "RGB"}})
    df = corrigir_preto(_df(100, 10, canto=4.0, ref=1.6), controle, "referencia")
    assert df.loc[0, "piso_video_G"] == 1.6 and df.loc[0, "dif_piso_G"] == 0


def test_avaliacao_mostra_queda_da_dispersao_quando_a_correcao_ajuda():
    # 3 regravações com o mesmo sinal real (8) e preto diferente por sessão; o fixo não vê a deriva
    linhas = []
    for canto, final in [(2.0, 9.0), (3.0, 10.0), (4.0, 11.0)]:
        linhas.append(_df(100, final, canto, 1.6).iloc[0].to_dict() | {"proporcao": "G1.5", "amostra": "A"})
    controle = pd.Series({**{f"piso_canto_{c}": 2.5 for c in "RGB"}, **{f"piso_ref_{c}": 1.6 for c in "RGB"}})
    df = corrigir_preto(pd.DataFrame(linhas), controle, "video")
    a = avaliar_correcao(df)["G"]
    assert a["dp_piso_video"] < a["dp_piso_fixo"] and a["dp_piso_video"] < 1e-9
