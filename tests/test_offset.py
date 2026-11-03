"""Testes do teste do offset cantos-ROI (P5, v0.11.2): regra medida de canal sem sinal, offset, reta e relatório. Rode: pytest"""
import numpy as np
import pandas as pd

from luminescencia.offset import dispersao, medir, offset_do_controle, parametros, relatorio, reta

P = parametros({})
CONTROLE = pd.Series({**{f"piso_canto_{c}": 2.5 for c in "RGB"}, **{f"piso_ref_{c}": 1.5 for c in "RGB"}})      # offset 1,0


def _videos(n=8, b=1.0, seed=0):
    rng = np.random.default_rng(seed)
    linhas = []
    for i in range(n):
        canto = 1.5 + 0.5 * i
        v = {"video": f"v{i}", "amostra": "ABC"[i % 3], "num_video": i + 1, "proporcao": "R1.5", "prop_R": 1.5, "prop_G": 0.0, "prop_B": 0.0}
        for c in "RGB":
            v.update({f"piso_canto_{c}": canto, f"piso_ref_{c}": 1.5, f"inicio_{c}": 80.0, f"final_{c}": 5.0, f"perto_do_ruido_{c}": False})
        v.update(perto_do_ruido_B=True, inicio_B=canto + 0.5, final_B=canto - 1.0 * b + rng.normal(0, 0.02),    # B sem sinal
                 perto_do_ruido_G=True, inicio_G=canto + 6.0, final_G=canto + 0.5)                              # G com vazamento (não serve)
        linhas.append(v)
    return pd.DataFrame(linhas)


def test_so_canal_sem_sinal_medido_entra_e_o_offset_bate_com_o_controle():
    m = medir(_videos(), CONTROLE, P)
    assert set(m["canal"]) == {"B"}                       # R tem sinal; G tem vazamento (inicio 6 níveis acima do canto)
    assert abs(m["offset_medido"].mean() - 1.0) < 0.05 and abs(m["erro"].mean()) < 0.05
    assert offset_do_controle(CONTROLE) == {"R": 1.0, "G": 1.0, "B": 1.0}


def test_a_regra_nao_olha_a_proporcao():
    a, b = _videos(), _videos()
    b[["prop_R", "prop_G", "prop_B"]] = [0.0, 1.5, 1.5]
    assert medir(a, CONTROLE, P).equals(medir(b, CONTROLE, P))


def test_reta_recupera_a_inclinacao_e_ignora_canto_no_piso_minimo():
    v = _videos(n=10, b=0.5)                              # final = canto - 0,5 -> b = 1; troco por final = 0,5 x canto
    for i in range(len(v)):
        v.loc[i, "final_B"] = 0.5 * v.loc[i, "piso_canto_B"]
    v.loc[0, "piso_canto_B"] = 1.0                        # canto no piso mínimo (1,0): fica fora da reta
    m = medir(v, CONTROLE, P)
    assert m["canto_no_piso_minimo"].sum() == 1
    r = reta(m)
    assert abs(r["b"] - 0.5) < 1e-6 and r["n"] == len(m) - 1


def test_dispersao_cai_com_o_piso_por_video_quando_o_preto_muda_entre_regravacoes():
    linhas = []
    for canto in (2.0, 3.0, 4.0):                          # mesma proporção e amostra; o preto muda e o sinal real é igual (8)
        l = {"proporcao": "G1.5", "amostra": "A", "prop_R": 0.0, "prop_G": 1.5, "prop_B": 0.0}
        for c in "RGB":
            l.update({f"piso_canto_{c}": canto, f"piso_ref_{c}": 1.5, f"final_{c}": 8.0 + (canto - 1.5)})
        linhas.append(l)
    d = dispersao(pd.DataFrame(linhas), CONTROLE)
    assert d["atual (canto - offset do controle)"]["G"] < d["piso fixo"]["G"] and d["atual (canto - offset do controle)"]["G"] < 1e-9


def test_relatorio_com_e_sem_canal_testavel():
    v = _videos()
    texto = relatorio(medir(v, CONTROLE, P), v, CONTROLE, P)
    assert "TESTE DO OFFSET" in texto and "Reta final_ROI" in texto and "B: 8 vídeos" in texto
    v2 = v.copy(); v2["perto_do_ruido_B"] = False
    assert "Nenhum canal sem sinal" in relatorio(medir(v2, CONTROLE, P), v2, CONTROLE, P)
