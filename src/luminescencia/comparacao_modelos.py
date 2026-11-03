"""Comparação de modelos de decaimento (D3, decidido em 02/11/2026).

Compara, SÓ PARA ANÁLISE, a exponencial simples, a dupla, a exponencial esticada e a lei de potência, na MESMA janela do
ajuste principal (começa em posprocessamento.inicio_ajuste_s e termina onde o sinal some no ruído, regra de modelar.py).
Nada daqui entra no caminho principal nem na tabela final: o principal continua sendo a exponencial dupla (a simples,
preferida pela professora, quase nunca serve).

Saída (output/modelagem/):
  comparacao_modelos.csv            uma linha por vídeo x canal x conjunto x modelo
  relatorio_comparacao_modelos.txt  R², erro, BIC, vitórias e repetibilidade por modelo

Conjuntos
  c0   C = 0 sobre corr_*:  exp_simples_c0 (2 parâmetros), potencia_c0 (3), exp_esticada_c0 (3), exp_dupla_c0 (4)
  cl   C livre sobre corr_*: exp_simples (3), potencia (4), exp_esticada (4), exp_dupla (5)

Como comparar modelos com formas diferentes
  - O tau de uma esticada ou o t0 de uma potência NÃO são o tau_medio da dupla. Por isso a repetibilidade é medida em um
    descritor que não depende do modelo: t_meio, o tempo (desde o início da janela) em que a curva ajustada cai à metade
    entre o valor inicial e o patamar (0 no conjunto c0, C no conjunto cl). Se o t_meio cair depois do fim da janela
    ajustada, é extrapolação e fica marcado (t_meio_fora_da_janela).
  - BIC já penaliza o número de parâmetros. Um modelo "vence" quando tem o menor BIC entre os modelos válidos do conjunto
    naquele sinal; "vitória clara" = vantagem de BIC maior que modelagem de delta_bic (padrão 10) sobre o segundo.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .modelar import fim_util, parametros as parametros_modelar
from .modelos import MODELOS, PADRAO, ajustar_modelo
from .posprocessamento import CANAIS, parametros as parametros_pos

CONJUNTOS = {
    "c0": ("exp_simples_c0", "potencia_c0", "exp_esticada_c0", "exp_dupla_c0"),
    "cl": ("exp_simples", "potencia", "exp_esticada", "exp_dupla"),
}


def t_meio(modelo: str, params: dict, duracao_s: float, com_c: bool) -> float:
    """Tempo (desde o início da janela) em que a curva ajustada cai à metade entre o valor inicial e o patamar."""
    m = MODELOS[modelo]
    grade = np.linspace(0, 5 * duracao_s, 20001)
    y = m.func(grade, *(params[k] for k in m.params))
    c = params.get("C", 0.0) if com_c else 0.0
    alvo = c + (y[0] - c) / 2
    abaixo = np.where(y <= alvo)[0]
    return float(grade[abaixo[0]]) if len(abaixo) else float("nan")


def _tarefa(args) -> list[dict]:
    ident, t, y, inicio, k, janela, dur_min = args[:7]
    c_min = args[7] if len(args) > 7 else None                 # P2: mesmo limite do C livre do modelar.py (só no conjunto cl)
    t, y = np.asarray(t, float), np.asarray(y, float)
    sel = t >= inicio - 1e-9
    t, y = t[sel], y[sel]
    if len(t) < 8:
        return []
    fim, ruido = fim_util(t, y, k, janela)
    if not fim == fim or fim - t[0] < dur_min:
        return []
    sel = t <= fim + 1e-9
    tt, yy = t[sel] - t[0], y[sel]
    duracao = float(tt[-1])
    linhas = []
    for conjunto, nomes in CONJUNTOS.items():
        for nome in nomes:
            cfg_aj = {**PADRAO, "c_minimo": c_min} if (conjunto == "cl" and c_min is not None) else PADRAO
            a = ajustar_modelo(MODELOS[nome], tt, yy, cfg_aj)
            L = {**ident, "conjunto": conjunto, "modelo": nome, "ruido": round(ruido, 3), "n_pontos": len(tt),
                 "fim_s": round(fim, 3), "ok": bool(a.ok), "motivo": a.motivo}
            if a.params:
                L.update({f"p_{k2}": v for k2, v in a.params.items()})
                L.update(r2=a.r2, rmse=a.rmse, bic=a.bic)
                L["t_meio"] = t_meio(nome, a.params, duracao, conjunto == "cl")
                L["t_meio_fora_da_janela"] = bool(L["t_meio"] != L["t_meio"] or L["t_meio"] > duracao)
            linhas.append(L)
    return linhas


def montar_tarefas(series, vpos, cfg, p, pp) -> list[tuple]:
    por_video = {v: g for v, g in series.groupby("video")}
    tarefas = []
    for _, v in vpos.iterrows():
        g = por_video.get(v["video"])
        if g is None:
            continue
        for c in CANAIS:
            if not v[f"usar_no_ajuste_{c}"]:
                continue
            ident = {"video": v["video"], "amostra": v["amostra"], "grupo": v["grupo"], "num_video": v["num_video"],
                     "proporcao": v["proporcao"], "canal": c, "canal_ativo": bool(v[f"canal_ativo_{c}"])}
            tarefas.append((ident, g["tempo_s"].to_numpy(), g[f"corr_{c}"].to_numpy(), pp["inicio_ajuste_s"],
                            p["ruido_k"], p["janela_mediana_pontos"], p["duracao_minima_s"], p.get("c_livre_c_minimo")))
    return tarefas


def _cv(x: pd.Series) -> float:
    return float(x.std(ddof=1) / x.mean()) if len(x) > 1 and x.mean() else np.nan


def resumir(df: pd.DataFrame, delta_bic: float = 10.0) -> str:
    L = ["COMPARAÇÃO DE MODELOS DE DECAIMENTO (só análise; o caminho principal não muda)", "=" * 70,
         f"Sinais comparados (vídeo x canal): {df.groupby(['video', 'canal']).ngroups}   linhas: {len(df)}", ""]
    for conjunto, nomes in CONJUNTOS.items():
        d = df[df["conjunto"] == conjunto]
        L += [f"CONJUNTO {conjunto.upper()}  ({'C = 0' if conjunto == 'c0' else 'C livre'}; modelos: {', '.join(nomes)})", "-" * 70]
        valido = d[d["ok"] & d["bic"].notna()]
        # vencedor por sinal
        venc, claro = {}, {}
        for chave, g in valido.groupby(["video", "canal"]):
            g = g.sort_values("bic")
            venc[chave] = g.iloc[0]["modelo"]
            claro[chave] = bool(len(g) == 1 or g.iloc[1]["bic"] - g.iloc[0]["bic"] > delta_bic)
        for c in CANAIS:
            dc = d[d["canal"] == c]
            if dc.empty:
                continue
            ruido = dc["ruido"].median()
            L.append(f"  Canal {c}  ({dc.groupby('video').ngroups} sinais; ruído mediano {ruido:.2f} níveis)")
            L.append(f"    {'modelo':<16}{'parâm.':>7}{'válido':>8}{'R² med.':>9}{'rmse med.':>10}{'rmse/ruído':>11}{'vence BIC':>11}{'(clara)':>9}"
                     f"{'CV t_meio':>11}{'t_meio fora':>12}")
            for nome in nomes:
                x = dc[dc["modelo"] == nome]
                xv = x[x["ok"] & x["r2"].notna()]
                chaves = list(zip(dc["video"].unique(), [c] * dc["video"].nunique()))
                v_, c_ = (sum(venc.get(k) == nome for k in chaves), sum(venc.get(k) == nome and claro.get(k, False) for k in chaves))
                cv = xv.groupby(["proporcao", "amostra"])["t_meio"].agg(_cv).median() if len(xv) else np.nan
                fora = float(xv["t_meio_fora_da_janela"].mean()) if len(xv) else np.nan
                npar = len(MODELOS[nome].params)
                L.append(f"    {nome:<16}{npar:>7}{len(xv) / max(len(x), 1):>8.0%}{xv['r2'].median():>9.5f}{xv['rmse'].median():>10.3f}"
                         f"{xv['rmse'].median() / ruido:>11.1f}{v_:>11}{c_:>9}{cv:>11.1%}{fora:>12.0%}")
            # parâmetros de forma
            for nome, par, rot in (("exp_esticada_c0" if conjunto == "c0" else "exp_esticada", "p_beta", "beta (esticada; 1 = simples)"),
                                   ("potencia_c0" if conjunto == "c0" else "potencia", "p_alpha", "alpha (potência)")):
                if nome in nomes and par in dc.columns:
                    xs = dc[(dc["modelo"] == nome) & dc["ok"]][par].dropna()
                    if len(xs):
                        enc = float((xs >= 0.999).mean()) if par == "p_beta" else float((xs >= 19.9).mean())
                        extra = f"; {enc:.0%} no limite superior ({'beta = 1' if par == 'p_beta' else 'alpha = 20'})" if enc > 0.02 else ""
                        L.append(f"      {rot}: mediana {xs.median():.2f} (q10 {xs.quantile(.1):.2f}, q90 {xs.quantile(.9):.2f}){extra}")
        L.append("")
    L += ["Como ler:",
          "  - rmse/ruído perto de 1 = o modelo explica a série até o ruído; bem acima = sobra estrutura no resíduo.",
          "  - 'vence BIC' penaliza parâmetros; '(clara)' = vantagem maior que delta_bic sobre o segundo colocado.",
          "  - CV t_meio = repetibilidade entre as 5 regravações (mediana por proporção e amostra); comparável entre modelos porque",
          "    t_meio não depende da forma do modelo. 't_meio fora' = fração em que o tempo de meia-queda foi extrapolado além da janela.",
          "  - Resultado de comparação: não muda o modelo do caminho principal sem decisão sua."]
    return "\n".join(L) + "\n"


def rodar(cfg: dict, pasta_pos=None, saida=None) -> int:
    p, pp = parametros_modelar(cfg), parametros_pos(cfg)
    raiz = Path(cfg.get("pastas", {}).get("saida", "output"))
    pos = Path(pasta_pos or raiz / "posprocessamento")
    for f in (pos / "series_pos.csv", pos / "videos_pos.csv"):
        if not f.exists():
            raise FileNotFoundError(f"{f} não existe (rode 'luminescencia consolidar' e 'luminescencia pos')")
    out = Path(saida or raiz / "modelagem")
    out.mkdir(parents=True, exist_ok=True)
    series = pd.read_csv(pos / "series_pos.csv", encoding="utf-8-sig")
    vpos = pd.read_csv(pos / "videos_pos.csv", encoding="utf-8-sig")
    tarefas = montar_tarefas(series, vpos, cfg, p, pp)
    print(f"{len(tarefas)} sinais a comparar x {sum(len(v) for v in CONJUNTOS.values())} modelos (workers: {p['workers']})...")
    if p["workers"] > 1:
        with ProcessPoolExecutor(p["workers"]) as ex:
            linhas = [l for parte in ex.map(_tarefa, tarefas, chunksize=10) for l in parte]
    else:
        linhas = [l for t in tarefas for l in _tarefa(t)]
    df = pd.DataFrame(linhas)
    df["motivo"] = df["motivo"].fillna("")
    df.to_csv(out / "comparacao_modelos.csv", index=False, encoding="utf-8-sig")
    texto = resumir(df, float(cfg.get("ajuste", {}).get("delta_bic", PADRAO["delta_bic"])))
    (out / "relatorio_comparacao_modelos.txt").write_text(texto, encoding="utf-8")
    print(texto)
    print(f"Arquivos em {out}: comparacao_modelos.csv, relatorio_comparacao_modelos.txt")
    return 0
