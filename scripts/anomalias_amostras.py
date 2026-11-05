"""Busca de comportamentos anômalos entre as amostras A, B e C (verificação de 04/11/2026). SÓ ANÁLISE.

Uso (na raiz, venv ativo):  python scripts/anomalias_amostras.py
Lê output/consolidacao/videos.csv, output/auditoria/relatorio_auditoria.csv, output/roi/roi.csv,
output/tabela/dataset.csv e output/tabela/descritores.csv. Não grava nada e não altera nenhum arquivo.

Cuidados: lê features e metadados de C (como o `deriva`), nunca desempenho em C. `prop_*` e a proporção só
agrupam vídeos (mesma preparação), nunca viram entrada. Nada daqui deve ser usado para escolher entradas (B13).
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest, kruskal, spearmanr

ENC = "utf-8-sig"
CANAIS = "RGB"


def ler():
    v = pd.read_csv("output/consolidacao/videos.csv", encoding=ENC, low_memory=False)
    ro = pd.read_csv("output/roi/roi.csv", encoding=ENC).drop_duplicates("caminho")
    v = v.merge(ro[["caminho", "circularidade", "contraste"]], on="caminho", how="left")
    ds = pd.read_csv("output/tabela/dataset.csv", encoding=ENC, low_memory=False)
    de = pd.read_csv("output/tabela/descritores.csv", encoding=ENC)
    cols_v = ["video"] + [f"piso_canto_{c}" for c in CANAIS] + [f"final_corr_{c}" for c in CANAIS] + ["contraste"]
    x = (ds[["video", "amostra", "proporcao", "num_video", "excluido_saturacao"] + [f"prop_{c}" for c in CANAIS]
            + [f"{c}_c0_tau_medio" for c in CANAIS]]
         .merge(v[cols_v], on="video")
         .merge(de[["video"] + [c for c in de.columns if c.endswith(("_md_I0", "_md_t50", "_md_r_5s", "_md_auc_10s"))]], on="video"))
    x = x[~x["excluido_saturacao"].astype(bool)].copy()
    x["prep"] = x["proporcao"] + "|" + x["amostra"]
    x["canto"] = x[[f"piso_canto_{c}" for c in CANAIS]].mean(axis=1)
    return v, de, x


def dentro(s, col, log=True):
    val = np.log(s[col]) if log else s[col]
    return val - val.groupby(s["prep"]).transform("mean")


def aquisicao(v):
    print("== 1. Aquisição por amostra (Kruskal-Wallis; eps² = tamanho de efeito)")
    linhas = []
    for col in ["roi_cx", "roi_cy", "contraste", "piso_canto_R", "piso_canto_G", "piso_canto_B", "repetidos_pct",
                "primeiro_decaimento_s", "duracao_s", "roi_r", "circularidade"]:
        x = pd.to_numeric(v[col], errors="coerce")
        g = [x[v["amostra"] == a].dropna() for a in "ABC"]
        h, p = kruskal(*g)
        n = sum(len(i) for i in g)
        linhas.append((col, *[round(i.median(), 2) for i in g], round(max((h - 2) / (n - 3), 0), 3), f"{p:.1e}"))
    print(pd.DataFrame(linhas, columns=["variável", "A", "B", "C", "eps2", "p"]).to_string(index=False))


def canto_x_decaimento(x):
    print("\n== 2. Dentro da preparação: nível de luz do canto x decaimento (resíduos da média da preparação)")
    for c in CANAIS:
        s = x[x[f"prop_{c}"] > 0].dropna(subset=[f"{c}_c0_tau_medio", f"{c}_md_t50", f"{c}_md_I0"])
        rc = dentro(s, f"piso_canto_{c}", log=False)
        r = {g: spearmanr(dentro(s, col), rc)[0] for g, col in
             [("tau_c0", f"{c}_c0_tau_medio"), ("t50", f"{c}_md_t50"), ("r_5s", f"{c}_md_r_5s"), ("I0", f"{c}_md_I0")]}
        outros = [spearmanr(rc, dentro(s, f"piso_canto_{o}", log=False))[0] for o in CANAIS if o != c]
        print(f"  {c} (n={len(s)}): rho canto x tau {r['tau_c0']:+.2f}, x t50 {r['t50']:+.2f}, x r_5s {r['r_5s']:+.2f}, "
              f"x I0 {r['I0']:+.2f} | canto deste canal x canto dos outros {min(outros):+.2f} a {max(outros):+.2f}")
    print("  cauda (final_corr) x canto, dentro da preparação:",
          {c: round(spearmanr(dentro(x[x[f'prop_{c}'] > 0], f'final_corr_{c}', False),
                              dentro(x[x[f'prop_{c}'] > 0], 'canto', False))[0], 2) for c in CANAIS})


def sessoes_descontado_canto(x):
    print("\n== 3. Diferença entre sessões antes e depois de descontar o canto (inclinação estimada só DENTRO das preparações)")
    print("  canto médio (RGB) por amostra:", x.groupby("amostra")["canto"].median().round(2).to_dict())
    for c in CANAIS:
        for g, col in [("tau_c0", f"{c}_c0_tau_medio"), ("t50", f"{c}_md_t50")]:
            s = x[x[f"prop_{c}"] > 0].dropna(subset=[col])
            b = np.polyfit(dentro(s, "canto", False), dentro(s, col), 1)[0]
            v = np.log(s[col])
            rp = v - v.groupby(s["proporcao"]).transform("mean")
            rc = s["canto"] - s["canto"].groupby(s["proporcao"]).transform("mean")
            f = lambda r: {a: round(float(np.expm1(r[s["amostra"] == a].median()) * 100), 1) for a in "ABC"}
            print(f"  {c} {g:6s} {b * 100:+.1f}%/nível | antes {f(rp)} -> descontado {f(rp - b * rc)}")


def ordem(x, de):
    print("\n== 4. Ordem das regravações (num_video) dentro da preparação")
    r = pd.Series([spearmanr(t["num_video"], t["canto"])[0] for _, t in x.groupby("prep") if len(t) >= 4]).dropna()
    print(f"  canto sobe com a ordem: rho mediano {r.median():+.2f}, positivo em {(r > 0).sum()}/{len(r)} (binomial p={binomtest(int((r > 0).sum()), len(r)).pvalue:.1e})")
    for c in CANAIS:
        col = f"{c}_md_I0"
        r = pd.Series([spearmanr(t["num_video"], t[col])[0] for _, t in x[x[f"prop_{c}"] > 0].groupby("prep")
                       if len(t.dropna(subset=[col])) >= 4]).dropna()
        print(f"  {col}: rho mediano {r.median():+.2f}, positivo em {(r > 0).sum()}/{len(r)} (p={binomtest(int((r > 0).sum()), len(r)).pvalue:.1e})")


def piso_minimo(v, x):
    print("\n== 5. Canto preso no piso mínimo da extração (<= 1,01; extracao.piso_minimo = 1,0)")
    for c in CANAIS:
        print(f"  {c}:", {a: int((v[v["amostra"] == a][f"piso_canto_{c}"] <= 1.01).sum()) for a in "ABC"})
    s = x[x["prop_R"] > 0].dropna(subset=["R_md_t50"]).copy()
    s["preso"] = (s[[f"piso_canto_{c}" for c in CANAIS]] <= 1.01).any(axis=1)
    d = np.log(s["R_md_t50"]) - np.log(s["R_md_t50"]).groupby(s["prep"]).transform("median")
    sem = s[~s["preso"]]
    sem = sem[sem.groupby("prep")["video"].transform("size") >= 3]
    print(f"  t50 do R nos vídeos presos: mediana {np.expm1(d[s['preso']].median()) * 100:+.1f}% em relação aos irmãos (n={int(s['preso'].sum())}); "
          f"sem os presos, rho canto x t50 continua {spearmanr(dentro(sem, 'R_md_t50'), dentro(sem, 'piso_canto_R', False))[0]:+.2f}")


def vazamento(v):
    print("\n== 7. Vazamento de cor da câmera (pó único): canal sem pó / canal do pó, mediana por amostra")
    for prop, c0 in [("G1.5", "G"), ("B1.5", "B")]:
        s = v[v["proporcao"] == prop]
        for c in CANAIS:
            if c != c0:
                r = s[f"inicio_corr_{c}"] / s[f"inicio_corr_{c0}"]
                print(f"  {prop} {c}/{c0}:", {a: round(float(r[s["amostra"] == a].median()), 4) for a in "ABC"})


def destoantes(de):
    print("\n== 6. Assinatura de cor: vídeos cujas frações de I0 batem melhor com OUTRA composição")
    i = de.set_index("video")[[f"{c}_md_I0" for c in CANAIS]].fillna(0)
    i = i.div(i.sum(axis=1), axis=0)
    lab = de.set_index("video")["proporcao"]
    med = i.join(lab).groupby("proporcao").median()
    achou = False
    for vid, linha in i.iterrows():
        d = ((med - linha) ** 2).sum(axis=1)
        prox = d.idxmin()
        if prox != lab[vid] and d[lab[vid]] > 4 * d[prox]:
            achou = True
            print(f"  {vid}: rotulado {lab[vid]}, mais próximo de {prox} "
                  f"(distância {d[prox]:.4f} contra {d[lab[vid]]:.4f} da própria); frações {linha.round(3).to_dict()}")
    if not achou:
        print("  nenhum")


def main():
    v, de, x = ler()
    aquisicao(v)
    canto_x_decaimento(x)
    sessoes_descontado_canto(x)
    ordem(x, de)
    piso_minimo(v, x)
    destoantes(de)
    vazamento(v)


if __name__ == "__main__":
    main()
