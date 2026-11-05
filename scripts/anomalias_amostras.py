"""Busca de comportamentos anômalos entre as amostras A, B e C (verificação de 04/11/2026). SÓ ANÁLISE.

Uso (na raiz, venv ativo):  python scripts/anomalias_amostras.py
Lê output/consolidacao/videos.csv, output/auditoria/relatorio_auditoria.csv, output/roi/roi.csv,
output/tabela/dataset.csv e output/tabela/descritores.csv. Não grava nada e não altera nenhum arquivo.

Seções 8 a 17 (verificações de 04 e 05/11/2026, v0.19.0 e v0.20.0): o que sobra depois da correção da luz (output/tabela/luz.csv), intensidade
inicial por sessão, ausentes por sessão, escolha de modelo por sessão, duplicatas, assinatura de decaimento e ordem das regravações
depois da correção.

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
    print("\n== 5. Canto abaixo do piso mínimo da extração (extracao.piso_minimo = 1,0)")
    for c in CANAIS:
        if f"canto_fonte_{c}" in v:          # v0.17.0: consolidacao.canto_sem_piso usa o canto real
            print(f"  {c}: fontes do canto", v.groupby("amostra")[f"canto_fonte_{c}"].value_counts().unstack(fill_value=0).to_dict("index"))
        else:
            print(f"  {c} (preso em 1,0):", {a: int((v[v["amostra"] == a][f"piso_canto_{c}"] <= 1.01).sum()) for a in "ABC"})
    s = x[x["prop_R"] > 0].dropna(subset=["R_md_t50"]).copy()
    s["preso"] = (s[[f"piso_canto_{c}" for c in CANAIS]] <= 1.01).any(axis=1)
    d = np.log(s["R_md_t50"]) - np.log(s["R_md_t50"]).groupby(s["prep"]).transform("median")
    sem = s[~s["preso"]]
    sem = sem[sem.groupby("prep")["video"].transform("size") >= 3]
    print("  (os 'presos' abaixo incluem cantos reais perto de 1,0 no R/G: o desvio deles é o efeito da luz da seção 2, não do piso)")
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


def _luz_e_sinal(ds, de):
    lz = pd.read_csv("output/tabela/luz.csv", encoding=ENC)
    x = (de[["video", "amostra", "proporcao"] + [c for c in de.columns if c.endswith(("_md_t50", "_md_auc_10s", "_md_I0"))]]
         .merge(lz.rename(columns=lambda c: c if c in ("video", "canto_luz") else c + "_luz"), on="video")
         .merge(ds[["video", "num_video", "excluido_saturacao"] + [f"sinal_presente_{c}" for c in CANAIS]], on="video"))
    x = x[~x["excluido_saturacao"].astype(bool)].copy()
    x["prep"] = x["proporcao"] + "|" + x["amostra"]
    return x


def pos_correcao(x):
    print("\n== 8. Depois da correção da luz: canto x t50 dentro da preparação, por amostra (rho sem -> com)")
    print("  (A e B calibram a correção; no C ela é só aplicada. Isto mede a medição, NUNCA serve para ajustar nada.)")
    for c in CANAIS:
        col = f"{c}_md_t50"
        s = x[x[f"sinal_presente_{c}"].astype(bool)].dropna(subset=[col])
        s = s[s.groupby("prep")["video"].transform("size") >= 3]
        w = lambda v: v - v.groupby(s["prep"]).transform("mean")
        r = {a: f"{spearmanr(w(s['canto_luz'])[s['amostra'] == a], w(np.log(s[col]))[s['amostra'] == a])[0]:+.2f} -> "
                f"{spearmanr(w(s['canto_luz'])[s['amostra'] == a], w(np.log(s[col + '_luz']))[s['amostra'] == a])[0]:+.2f}" for a in "ABC"}
        print(f"  {col}: {r}")
    print("\n== 9. Deriva entre sessões que sobra (desvio mediano em relação à média da composição, %; sem -> com correção)")
    for c in CANAIS:
        for g in ("t50", "auc_10s"):
            col = f"{c}_md_{g}"
            s = x[x[f"sinal_presente_{c}"].astype(bool)].dropna(subset=[col])
            f = lambda v: {a: round(float(np.expm1((v - v.groupby(s["proporcao"]).transform("mean"))[s["amostra"] == a].median()) * 100), 1) for a in "ABC"}
            print(f"  {col:14s} {f(np.log(s[col]))} -> {f(np.log(s[col + '_luz']))}")


def intensidade(ds, de):
    print("\n== 10. Intensidade inicial (I0) por sessão, relativa à composição (%; só canais com sinal medido)")
    s = de.merge(ds[["video", "excluido_saturacao"] + [f"sinal_presente_{c}" for c in CANAIS]], on="video")
    s = s[~s["excluido_saturacao"].astype(bool)]
    for c in CANAIS:
        t = s[s[f"sinal_presente_{c}"].astype(bool)].dropna(subset=[f"{c}_md_I0"])
        r = np.log(t[f"{c}_md_I0"]) - np.log(t[f"{c}_md_I0"]).groupby(t["proporcao"]).transform("mean")
        print(f"  {c}_md_I0: {dict((a, round(float(np.expm1(r[t['amostra'] == a].median()) * 100), 1)) for a in 'ABC')} "
              f"(Kruskal p={kruskal(*[r[t['amostra'] == a] for a in 'ABC']).pvalue:.1e})")
    for col in ("razao_I0_R_G", "razao_I0_R_B", "razao_I0_G_B"):
        t = s.dropna(subset=[col])
        t = t[t[col] > 0]
        r = np.log(t[col]) - np.log(t[col]).groupby(t["proporcao"]).transform("mean")
        print(f"  {col}: {dict((a, round(float(np.expm1(r[t['amostra'] == a].median()) * 100), 1)) for a in 'ABC')}")
    print("  saturados por amostra:", ds[ds["excluido_saturacao"].astype(bool)].groupby("amostra").size().to_dict())


def ausentes_e_modelos(ds):
    print("\n== 11. Ausentes por sessão nas colunas de entrada (descritores ligados) e escolha de modelo por sessão")
    import yaml

    from luminescencia import entradas as en
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    cfg["entradas"]["descritores"] = True
    ent = en.carregar_entradas(cfg)
    ent["amostra"] = ds["amostra"].to_numpy()
    ent = ent[~ds["excluido_saturacao"].astype(bool).to_numpy()]
    na = ent.drop(columns=["amostra"]).isna().groupby(ent["amostra"]).mean()
    dif = (na.max() - na.min()).sort_values(ascending=False)
    dif = dif[dif > 0.05]
    print(f"  {len(dif)} colunas com fração de ausentes diferente em mais de 5 p.p. entre amostras:",
          {k: na[k].round(2).to_dict() for k in dif.index})
    aj = pd.read_csv("output/modelagem/ajustes.csv", encoding=ENC, low_memory=False)
    for cam in ("principal", "c_livre"):
        print(f"  modelos ({cam}):", pd.crosstab(aj[aj["caminho"] == cam]["amostra"], aj[aj["caminho"] == cam]["modelo"]).to_dict("index"))


def duplicatas(de):
    print("\n== 12. Duplicatas: vídeos com I0 e t50 idênticos nos 3 canais (ignorando os sem descritores)")
    k = de.set_index("video")[[f"{c}_md_{g}" for c in CANAIS for g in ("I0", "t50")]].round(3)
    k = k[k.notna().any(axis=1)].fillna(-1)
    d = k[k.duplicated(keep=False)]
    print("  ", sorted(d.index) if len(d) else "nenhuma")


def assinatura_decaimento(ds):
    print("\n== 13. Assinatura de DECAIMENTO (t50 corrigido nos canais com sinal) que bate melhor com outra composição")
    lz = pd.read_csv("output/tabela/luz.csv", encoding=ENC).set_index("video")
    lab = ds.set_index("video")["proporcao"]
    sp = ds.set_index("video")[[f"sinal_presente_{c}" for c in CANAIS]].astype(bool)
    lg = np.log(lz[[f"{c}_md_t50" for c in CANAIS]].where(sp.loc[lz.index].to_numpy()))
    med = lg.join(lab).groupby("proporcao").median()
    achados = []
    for vid, row in lg.iterrows():
        if row.isna().all() or vid not in lab.index:
            continue
        d = ((med - row) ** 2).mean(axis=1, skipna=True).dropna()
        if lab[vid] in d and d.idxmin() != lab[vid] and d[lab[vid]] > 4 * d.min() and d[lab[vid]] > 0.01:
            achados.append(vid)
            print(f"  {vid}: rotulado {lab[vid]}, mais perto de {d.idxmin()} ({d.min():.4f} contra {d[lab[vid]]:.4f})")
    print(f"  total {len(achados)} (sinal fraco: composições com t50 parecidos se confundem; vale como alerta quando coincide com a seção 6)")


def ordem_pos_correcao(x):
    print("\n== 14. Ordem das regravações DEPOIS da correção: t50 corrigido e I0, tendência dentro da preparação")
    for col in [f"{c}_md_t50_luz" for c in CANAIS] + [f"{c}_md_I0" for c in CANAIS]:
        c = col[0]
        r = pd.Series([spearmanr(g["num_video"], g[col])[0] for _, g in x[x[f"sinal_presente_{c}"].astype(bool)].groupby("prep")
                       if g[col].notna().sum() >= 4]).dropna()
        print(f"  {col:14s} rho mediano {r.median():+.2f}, positivo em {(r > 0).sum()}/{len(r)} (p={binomtest(int((r > 0).sum()), len(r)).pvalue:.1e})")


def ordem_razoes(ds, de):
    print("\n== 15. Ordem das regravações nas RAZÕES de I0 (a queda do brilho se cancela entre canais?)")
    x = de[[c for c in de.columns if c not in ("num_video", "grupo")]].merge(ds[["video", "num_video", "excluido_saturacao"]], on="video")
    x = x[~x["excluido_saturacao"].astype(bool)].copy()
    x["prep"] = x["proporcao"] + "|" + x["amostra"]
    for col in ("razao_I0_R_G", "razao_I0_R_B", "razao_I0_G_B", "R_md_I0", "G_md_I0", "B_md_I0"):
        s = x.dropna(subset=[col])
        s = s[s[col] > 0]
        r = pd.Series([spearmanr(g["num_video"], g[col])[0] for _, g in s.groupby("prep") if len(g) >= 4]).dropna()
        w = np.log(s[col]) - np.log(s[col]).groupby(s["prep"]).transform("mean")
        b = np.polyfit(s["num_video"] - s["num_video"].groupby(s["prep"]).transform("mean"), w, 1)[0]
        print(f"  {col:13s} positivo em {(r > 0).sum()}/{len(r)} (p={binomtest(int((r > 0).sum()), len(r)).pvalue:.2f}); "
              f"{b * 100:+.2f}% por regravação (da 1ª à 5ª: {np.expm1(4 * b) * 100:+.1f}%)")


def sessao_nas_entradas(ds):
    print("\n== 16. Quanto as ENTRADAS finais ainda diferenciam as sessões (A x B, dentro da composição; só treino)")
    print("  fração da variação dentro da composição que é diferença entre A e B (0 = nenhuma; 1 = só sessão).")
    print("  Sessão e preparação estão confundidas (cada amostra é uma preparação gravada numa sessão): isto mede as duas juntas.")
    import yaml

    from luminescencia import entradas as en
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    cfg["entradas"]["descritores"] = True
    ent = en.carregar_entradas(cfg)
    ent["amostra"] = ds["amostra"].to_numpy()
    ent["proporcao"] = ds["proporcao"].to_numpy()
    ent = ent[(~ds["excluido_saturacao"].astype(bool)).to_numpy() & ds["amostra"].isin(["A", "B"]).to_numpy()]
    ent = ent[ent.groupby("proporcao")["amostra"].transform("nunique") == 2]
    linhas = []
    for col in [c for c in ent.columns if c not in ("amostra", "proporcao") and ent[c].dtype.kind == "f"]:
        t = ent[["amostra", "proporcao", col]].dropna()
        if t[col].nunique() < 5:
            continue
        m_comp = t[col].groupby(t["proporcao"]).transform("mean")
        m_prep = t[col].groupby(t["proporcao"] + "|" + t["amostra"]).transform("mean")
        tot = float(((t[col] - m_comp) ** 2).sum())
        if tot > 0:
            linhas.append((col, float(((m_prep - m_comp) ** 2).sum()) / tot))
    r = pd.DataFrame(linhas, columns=["coluna", "fracao_sessao"]).sort_values("fracao_sessao", ascending=False)
    print(f"  {len(r)} colunas; mediana {r['fracao_sessao'].median():.2f}; acima de 0,5: {int((r['fracao_sessao'] > 0.5).sum())}")
    print("  as 8 maiores:", [(c, round(v, 2)) for c, v in r.head(8).values])
    d4 = r[r["coluna"].str.contains("md_t50|md_r_5s|md_auc_10s|razao_")]
    print("  colunas da D4 (t50, r_5s, auc_10s, razões):", [(c, round(v, 2)) for c, v in d4.values])
    sem = r[~r["coluna"].map(en.eh_intensidade)]
    print(f"  sem as colunas de intensidade (entradas.sem_intensidade): mediana {sem['fracao_sessao'].median():.2f}; "
          f"acima de 0,5: {int((sem['fracao_sessao'] > 0.5).sum())}")


def cobertura(ds):
    print("\n== 17. Cobertura: vídeos utilizáveis (sem saturados) por composição e amostra")
    u = ds[~ds["excluido_saturacao"].astype(bool)]
    tab = pd.crosstab(u["proporcao"], u["amostra"]).reindex(columns=list("ABC"), fill_value=0)
    print("  composições sem alguma amostra:", {p: [a for a in "ABC" if tab.loc[p, a] == 0] for p in tab.index if (tab.loc[p] == 0).any()})
    print("  composições com menos de 5 vídeos em alguma amostra:", {p: tab.loc[p].to_dict() for p in tab.index if (tab.loc[p] < 5).any()})
    print("  saturados por composição e amostra:", ds[ds["excluido_saturacao"].astype(bool)].groupby(["proporcao", "amostra"]).size().to_dict())


def main():
    v, de, x = ler()
    aquisicao(v)
    canto_x_decaimento(x)
    sessoes_descontado_canto(x)
    ordem(x, de)
    piso_minimo(v, x)
    destoantes(de)
    vazamento(v)
    ds = pd.read_csv("output/tabela/dataset.csv", encoding=ENC, low_memory=False)
    xl = _luz_e_sinal(ds, de)
    pos_correcao(xl)
    intensidade(ds, de)
    ausentes_e_modelos(ds)
    duplicatas(de)
    assinatura_decaimento(ds)
    ordem_pos_correcao(xl)
    ordem_razoes(ds, de)
    sessao_nas_entradas(ds)
    cobertura(ds)


if __name__ == "__main__":
    main()
