"""P5 (aprovada em 03/11/2026) - Teste da suposição do preto por vídeo: a diferença "cantos - dentro da ROI" é a mesma em todas as sessões?

Só análise (nenhum número do pipeline muda). Entrada: output/extracao/extracao_resumo.csv (o controle) e output/consolidacao/videos.csv.
Saída (output/offset/): offset_por_video.csv e relatorio_offset.txt.

A consolidação corrige o preto por vídeo: piso_video = piso_canto_do_vídeo - offset_do_controle, com offset_do_controle = canto_do_controle -
piso_ref_do_controle (dentro da ROI). Isso supõe que o offset vale igual em todas as gravações. Como testar sem vídeo novo: num vídeo em que um
canal NÃO tem sinal nenhum, o nível final dentro da ROI é só o preto da câmera naquele ponto, então
      offset_medido = piso_canto - final_ROI
pode ser comparado com o do controle. "Canal sem sinal" é uma regra MEDIDA (nunca a proporção do nome): perto_do_ruido_* e nível inicial a menos de
offset.limiar_vazamento níveis acima do canto. Canal com vazamento de cor (ex.: o G em R_15, que começa ~5 níveis acima do preto) não serve: o
final dele ainda inclui um resto do sinal.

Limites: poucos vídeos (hoje só o B das 15 gravações de R_15), cantos no piso mínimo da extração (extracao.piso_minimo) escondem o valor real, e o
nível final dentro da ROI não pode ficar abaixo de 0 (efeito de piso nas sessões mais escuras).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CANAIS = ["R", "G", "B"]


def parametros(cfg: dict) -> dict:
    p = {"limiar_vazamento": 3.0, "tolerancia_offset": 1.0}
    p.update(cfg.get("offset", {}) or {})
    p["piso_minimo"] = float((cfg.get("extracao", {}) or {}).get("piso_minimo", 1.0))
    return p


def offset_do_controle(controle: pd.Series) -> dict:
    return {c: float(controle[f"piso_canto_{c}"]) - float(controle[f"piso_ref_{c}"]) for c in CANAIS}


def medir(videos: pd.DataFrame, controle: pd.Series, p: dict) -> pd.DataFrame:
    """Uma linha por vídeo x canal sem sinal (regra medida): offset_medido, erro contra o offset do controle e se o canto está no piso mínimo."""
    off = offset_do_controle(controle)
    linhas = []
    for _, v in videos.iterrows():
        for c in CANAIS:
            vazamento = float(v[f"inicio_{c}"]) - float(v[f"piso_canto_{c}"])
            if not (bool(v[f"perto_do_ruido_{c}"]) and vazamento < p["limiar_vazamento"]):
                continue
            medido = float(v[f"piso_canto_{c}"]) - float(v[f"final_{c}"])
            linhas.append({"video": v["video"], "amostra": v["amostra"], "num_video": v["num_video"], "canal": c,
                           "piso_canto": float(v[f"piso_canto_{c}"]), "final_roi": float(v[f"final_{c}"]),
                           "offset_medido": round(medido, 3), "offset_controle": round(off[c], 3), "erro": round(medido - off[c], 3),
                           "canto_no_piso_minimo": bool(float(v[f"piso_canto_{c}"]) <= p["piso_minimo"] + 1e-6)})
    return pd.DataFrame(linhas)


def reta(m: pd.DataFrame) -> dict:
    """final_ROI = a + b x canto, só com cantos acima do piso mínimo (b = 1 se o offset fosse constante)."""
    u = m[~m["canto_no_piso_minimo"]]
    if len(u) < 4 or u["piso_canto"].nunique() < 3:
        return {"n": len(u)}
    b, a = np.polyfit(u["piso_canto"], u["final_roi"], 1)
    return {"n": len(u), "a": float(a), "b": float(b), "r": float(np.corrcoef(u["piso_canto"], u["final_roi"])[0, 1])}


def dispersao(videos: pd.DataFrame, controle: pd.Series, a: float | None = None, b: float | None = None) -> dict:
    """dp do nível final entre regravações (canais com o pó): piso fixo, piso por vídeo atual e (se a, b) piso = a + b x canto."""
    off = offset_do_controle(controle)
    metodos = {"piso fixo": lambda g, c: g[f"piso_ref_{c}"], "atual (canto - offset do controle)": lambda g, c: g[f"piso_canto_{c}"] - off[c]}
    if a is not None:
        metodos["alternativa a + b x canto"] = lambda g, c: a + b * g[f"piso_canto_{c}"]
    saida = {}
    for nome, f in metodos.items():
        saida[nome] = {}
        for c in CANAIS:
            var = [(g[f"final_{c}"] - f(g, c)).var(ddof=1) for _, g in videos[videos[f"prop_{c}"] > 0].groupby(["proporcao", "amostra"]) if len(g) > 1]
            saida[nome][c] = float(np.sqrt(np.mean(var))) if var else float("nan")
    return saida


def relatorio(m: pd.DataFrame, videos: pd.DataFrame, controle: pd.Series, p: dict) -> str:
    off = offset_do_controle(controle)
    L = ["TESTE DO OFFSET CANTOS - ROI (P5; só análise, nada do pipeline muda)", "=" * 70,
         "Offset do controle (canto - dentro da ROI): " + "  ".join(f"{c} {off[c]:.2f}" for c in CANAIS), ""]
    if m.empty:
        return "\n".join(L + ["Nenhum canal sem sinal encontrado (regra: perto do ruído e início a menos de "
                              f"{p['limiar_vazamento']:g} níveis do canto): não há como testar a suposição."]) + "\n"
    L.append(f"Canais sem sinal (regra medida: perto do ruído e início < {p['limiar_vazamento']:g} níveis acima do canto):")
    for c, g in m.groupby("canal"):
        e = g["erro"]
        L.append(f"  {c}: {len(g)} vídeos ({', '.join(f'{a}: {n}' for a, n in g.groupby('amostra').size().items())})")
        if len(g) < 5:
            L.append("     poucos vídeos para estatística (menos de 5): ignorado nas conclusões")
            continue
        L.append(f"     offset medido: média {g['offset_medido'].mean():.2f}, mediana {g['offset_medido'].median():.2f}, dp {g['offset_medido'].std():.2f}   "
                 f"(controle {off[c]:.2f})")
        L.append(f"     erro contra o controle: média {e.mean():+.2f}, dp {e.std():.2f}; dentro de ±{p['tolerancia_offset']:g} nível: "
                 f"{int((e.abs() <= p['tolerancia_offset']).sum())}/{len(g)}")
        pior = g.loc[g["erro"].abs().idxmax()]
        L.append(f"     sem o pior vídeo ({pior['video']}, erro {pior['erro']:+.2f}): dp {g[g['video'] != pior['video']]['erro'].std():.2f}")
        por = g.groupby("amostra")["offset_medido"].agg(["mean", "std", "count"]).round(2)
        L.append("     por amostra (média, dp, n): " + "   ".join(f"{a}: {r['mean']:.2f}, {r['std']:.2f}, {int(r['count'])}" for a, r in por.iterrows()))
        if g["amostra"].nunique() > 1 and len(g) >= 6:
            from scipy import stats
            pv = stats.kruskal(*[x["offset_medido"].to_numpy() for _, x in g.groupby("amostra")]).pvalue
            L.append(f"     diferença entre amostras (Kruskal-Wallis): p = {pv:.2f}   (poucos vídeos por amostra: pouco poder para achar diferença pequena)")
        n_piso = int(g["canto_no_piso_minimo"].sum())
        if n_piso:
            L.append(f"     {n_piso} vídeos com o canto no piso mínimo da extração ({p['piso_minimo']:g}): o canto real pode ser menor e o offset medido fica distorcido")
    r = {c: reta(g) for c, g in m.groupby("canal") if len(g) >= 5}
    L += ["", "O offset é constante? Reta final_ROI = a + b x canto (só cantos acima do piso mínimo; b = 1 se o offset fosse constante):"]
    for c, x in r.items():
        L.append(f"  {c}: n = {x['n']}" + (f"   b = {x['b']:.2f}, a = {x['a']:.2f}, r = {x['r']:.2f}" if "b" in x else "   (poucos pontos)"))
    ref = next((x for x in r.values() if "b" in x), None)
    L += ["", "Efeito na dispersão do nível final entre as regravações (dp, níveis; canais com o pó; menor = melhor):"]
    disp = dispersao(videos, controle, ref["a"] if ref else None, ref["b"] if ref else None)
    for nome, d in disp.items():
        L.append(f"  {nome:<36}" + "  ".join(f"{c} {d[c]:.2f}" for c in CANAIS))
    L += ["",
          "Como ler: o offset do controle vale, em média, para o canal testado se o erro médio estiver perto de 0. Se b ficar bem abaixo de 1, o offset NÃO é",
          "constante (o ROI sobe menos que os cantos), e a correção por vídeo erra mais nos vídeos com cantos muito acima do controle. A alternativa linear só",
          "vale se reduzir a dispersão acima; calibrada com poucos pontos de um só canal, ela não deve substituir a correção atual sem mais dados."]
    return "\n".join(L) + "\n"


def rodar(cfg: dict, pasta_extracao=None, pasta_consolidacao=None, saida=None) -> int:
    p = parametros(cfg)
    raiz = Path(cfg.get("pastas", {}).get("saida", "output"))
    ext, cons = Path(pasta_extracao or raiz / "extracao"), Path(pasta_consolidacao or raiz / "consolidacao")
    for f in (ext / "extracao_resumo.csv", cons / "videos.csv"):
        if not f.exists():
            raise FileNotFoundError(f"{f} não existe (rode 'luminescencia extrair' e 'luminescencia consolidar')")
    resumo = pd.read_csv(ext / "extracao_resumo.csv", encoding="utf-8-sig")
    ctrl = resumo[resumo["controle"].astype(bool)]
    if ctrl.empty:
        raise ValueError("sem vídeo de controle em extracao_resumo.csv: não há offset de referência")
    videos = pd.read_csv(cons / "videos.csv", encoding="utf-8-sig")
    m = medir(videos, ctrl.iloc[0], p)
    out = Path(saida or raiz / "offset")
    out.mkdir(parents=True, exist_ok=True)
    m.to_csv(out / "offset_por_video.csv", index=False, encoding="utf-8-sig")
    texto = relatorio(m, videos, ctrl.iloc[0], p)
    (out / "relatorio_offset.txt").write_text(texto, encoding="utf-8")
    print(texto)
    print(f"Arquivos em {out}: offset_por_video.csv, relatorio_offset.txt")
    return 0
