"""Gráficos (D4, decididos em 02/11/2026), na ordem combinada:

  1_decaimento_semilog.png   decaimento em escala semilog com os dois ajustes (cl e c0) em vídeos típicos
  2_tau_cl_vs_c0.png         tau_medio de cl contra c0, por canal e por amostra (o resultado para o orientador)
  3_residuos.png             resíduo médio (dado menos ajuste) ao longo do tempo, para cl e c0, por canal
  4_normalizada.png          I/I0 por proporção, por canal

Lê output/modelagem/ajustes.csv, output/posprocessamento/series_pos.csv e videos_pos.csv. Não altera nada, só desenha.
Também grava graficos_resumo.txt com os números por trás dos desenhos (para não depender só do olho).

Cuidados
  - Escala logarítmica ignora pontos <= 0 (cauda no ruído), como previsto no README.
  - Os dois caminhos usam a mesma janela (começa em posprocessamento.inicio_ajuste_s, termina onde o sinal some no ruído).
  - canal_ativo_* (vem do nome do arquivo) é usado SÓ para legenda e conferência dos desenhos, nunca como entrada de IA.
  - Vídeos que saturam depois de 2,5 s não têm ajuste principal e não aparecem.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .modelos import MODELOS
from .posprocessamento import CANAIS

COR_CANAL = {"R": "#c0392b", "G": "#27ae60", "B": "#2c5fb3"}
COR_AMOSTRA = {"A": "#e08e0b", "B": "#7d3c98", "C": "#117a65"}
CAMINHOS = {"c0": "principal", "cl": "c_livre"}
TIPICAS = ["B1.5", "G1.5", "R1.5", "G1_B0.5", "R1_B0.5", "R0.5_G0.5_B0.5"]


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as erro:  # pragma: no cover
        raise ImportError("os gráficos precisam do matplotlib: pip install matplotlib") from erro
    return plt


def parametros(cfg: dict) -> dict:
    p = {"proporcoes_tipicas": TIPICAS, "dpi": 130, "bin_residuo_s": 2.0, "bin_normalizada_s": 1.0}
    p.update(cfg.get("graficos", {}) or {})
    return p


def curva(linha, t_rel: np.ndarray) -> np.ndarray:
    """Curva ajustada de uma linha de ajustes.csv, em função do tempo desde o início da janela."""
    m = MODELOS[linha["modelo"]]
    return m.func(np.asarray(t_rel, float), *(float(linha[k]) for k in m.params))


def janela(g: pd.DataFrame, linha) -> pd.DataFrame:
    """Pontos da série usados naquele ajuste (mesmo recorte de modelar.py)."""
    d = g[(g["tempo_s"] >= linha["inicio_s"] - 1e-9) & (g["tempo_s"] <= linha["fim_s"] + 1e-9)]
    return d


def _ajustes_validos(aj: pd.DataFrame) -> pd.DataFrame:
    aj = aj.copy()
    aj["modelo"] = aj["modelo"].fillna("")
    return aj[(aj["modelo"] != "") & aj["caminho"].isin(CAMINHOS.values())]


def _tipico(aj: pd.DataFrame, proporcao: str) -> str | None:
    """Vídeo típico da proporção: tem os dois ajustes em todos os canais que têm algum e o tau mais perto da mediana."""
    d = aj[(aj["proporcao"] == proporcao) & (aj["caminho"] == "principal")]
    if d.empty:
        return None
    por_video = d.groupby("video")["tau_medio"].apply(lambda x: float(np.mean(np.log(x))))
    n_canais = d.groupby("video")["canal"].nunique()
    por_video = por_video[n_canais == n_canais.max()]
    if por_video.empty:
        return None
    return (por_video - por_video.median()).abs().idxmin()


# ---------------------------------------------------------------- 1. semilog
def grafico_semilog(aj, series, p, out) -> list[str]:
    plt = _plt()
    escolhidos = [(pr, _tipico(aj, pr)) for pr in p["proporcoes_tipicas"]]
    escolhidos = [(pr, v) for pr, v in escolhidos if v]
    if not escolhidos:
        return []
    fig, eixos = plt.subplots(len(escolhidos), 3, figsize=(15, 3.1 * len(escolhidos)), squeeze=False)
    notas = []
    for i, (pr, vid) in enumerate(escolhidos):
        g = series[series["video"] == vid]
        for j, c in enumerate(CANAIS):
            ax = eixos[i][j]
            linhas = {k: aj[(aj["video"] == vid) & (aj["canal"] == c) & (aj["caminho"] == cam)] for k, cam in CAMINHOS.items()}
            if any(len(v) == 0 for v in linhas.values()):
                ax.text(0.5, 0.5, "sem ajuste\n(sem sinal ou saturado)", ha="center", va="center", transform=ax.transAxes, color="gray")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{pr} · {c}", fontsize=9, loc="left")
                continue
            l0, lc = linhas["c0"].iloc[0], linhas["cl"].iloc[0]
            d = janela(g, l0)
            y = d[f"corr_{c}"].to_numpy()
            ok = y > 0
            ax.scatter(d["tempo_s"][ok], y[ok], s=4, color="#999999", alpha=0.7, label="dado (corr > 0)", zorder=1)
            tr = np.linspace(0, l0["fim_s"] - l0["inicio_s"], 400)
            ax.plot(l0["inicio_s"] + tr, np.clip(curva(l0, tr), 1e-6, None), color="#d35400", lw=1.6, label=f"c0: tau {l0['tau_medio']:.1f} s")
            ax.plot(lc["inicio_s"] + tr, np.clip(curva(lc, tr), 1e-6, None), color="#1f77b4", lw=1.6, ls="--",
                    label=f"cl: tau {lc['tau_medio']:.1f} s, C {lc['C']:.1f}")
            ax.set_yscale("log")
            ymin = max(float(np.nanmin(y[ok])) if ok.any() else 0.1, 0.05)
            ax.set_ylim(bottom=ymin * 0.7)
            ax.set_title(f"{pr} · {c}   ({vid})", fontsize=9, loc="left", color=COR_CANAL[c])
            ax.grid(alpha=0.25, which="both")
            ax.legend(fontsize=7, loc="upper right")
            if i == len(escolhidos) - 1:
                ax.set_xlabel("tempo no vídeo (s)")
            if j == 0:
                ax.set_ylabel("sinal sem o preto (níveis)")
        notas.append(f"  {pr}: {vid}")
    fig.suptitle("Decaimento (semilog) com os dois ajustes: c0 (C = 0) e cl (C livre), na mesma janela", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "1_decaimento_semilog.png", dpi=p["dpi"], bbox_inches="tight")
    plt.close(fig)
    return ["Vídeos típicos usados no gráfico 1 (tau mais perto da mediana da proporção):"] + notas


# ---------------------------------------------------------------- 2. tau cl x c0
def tabela_tau(aj, vpos) -> pd.DataFrame:
    a = _ajustes_validos(aj)
    pv = a.pivot_table(index=["video", "canal", "amostra", "proporcao"], columns="caminho", values="tau_medio").reset_index()
    pv = pv.dropna(subset=["principal", "c_livre"]).rename(columns={"principal": "tau_c0", "c_livre": "tau_cl"})
    ativo = {(r["video"], c): bool(r[f"canal_ativo_{c}"]) for _, r in vpos.iterrows() for c in CANAIS}
    pv["com_po"] = [ativo.get((v, c), False) for v, c in zip(pv["video"], pv["canal"])]
    return pv


def grafico_tau(aj, vpos, p, out) -> list[str]:
    plt = _plt()
    t = tabela_tau(aj, vpos)
    if t.empty:
        return []
    fig, eixos = plt.subplots(2, 3, figsize=(15, 9))
    L = ["tau_medio: c0 (C = 0) contra cl (C livre)   [mediana; razão cl/c0 mediana por canal]"]
    for j, c in enumerate(CANAIS):
        d = t[t["canal"] == c]
        ax = eixos[0][j]
        if d.empty:
            for a_ in (ax, eixos[1][j]):
                a_.text(0.5, 0.5, f"canal {c}: sem ajustes", ha="center", va="center", transform=a_.transAxes, color="gray")
                a_.set_xticks([]); a_.set_yticks([])
            continue
        lim = [max(d[["tau_c0", "tau_cl"]].min().min() * 0.8, 0.3), d[["tau_c0", "tau_cl"]].max().max() * 1.2]
        ax.plot(lim, lim, color="gray", lw=1, ls=":", label="cl = c0")
        for a_, cor in COR_AMOSTRA.items():
            for po, mk, fc in ((True, "o", cor), (False, "o", "none")):
                x = d[(d["amostra"] == a_) & (d["com_po"] == po)]
                ax.scatter(x["tau_c0"], x["tau_cl"], s=16, marker=mk, facecolors=fc, edgecolors=cor, linewidths=1,
                           label=f"{a_}{'' if po else ' (sem o pó)'}", alpha=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("tau_medio c0 (s)"); ax.set_ylabel("tau_medio cl (s)")
        razao = (d["tau_cl"] / d["tau_c0"]).median()
        ax.set_title(f"canal {c}: cl/c0 mediana {razao:.2f}", color=COR_CANAL[c], fontsize=10)
        ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=7, ncol=2)
        L.append(f"  {c}: n={len(d)}  razão cl/c0 mediana {razao:.2f}  (q10 {(d['tau_cl'] / d['tau_c0']).quantile(.1):.2f}, "
                 f"q90 {(d['tau_cl'] / d['tau_c0']).quantile(.9):.2f})")
        # por amostra
        ax = eixos[1][j]
        posicoes, dados, cores, rot = [], [], [], []
        k = 0
        for a_ in sorted(d["amostra"].unique()):
            for rotulo, col, tom in (("c0", "tau_c0", 1.0), ("cl", "tau_cl", 0.45)):
                x = d[(d["amostra"] == a_) & d["com_po"]][col].dropna()
                if len(x):
                    posicoes.append(k); dados.append(x.to_numpy()); cores.append((COR_AMOSTRA[a_], tom)); rot.append(f"{a_}\n{rotulo}")
                k += 1
            k += 0.6
        if dados:
            bp = ax.boxplot(dados, positions=posicoes, widths=0.7, patch_artist=True, showfliers=False)
            for patch, (cor, tom) in zip(bp["boxes"], cores):
                patch.set_facecolor(cor); patch.set_alpha(tom)
            for m in bp["medians"]:
                m.set_color("black")
            ax.set_xticks(posicoes); ax.set_xticklabels(rot, fontsize=8)
        ax.set_ylabel("tau_medio (s), só vídeos com o pó no canal"); ax.grid(alpha=0.25, axis="y")
        ax.set_title(f"canal {c}: por amostra (A, B treino; C validação)", fontsize=10, color=COR_CANAL[c])
        med = {a_: (d[(d['amostra'] == a_) & d['com_po']]['tau_c0'].median(), d[(d['amostra'] == a_) & d['com_po']]['tau_cl'].median())
               for a_ in sorted(d["amostra"].unique())}
        L.append("     mediana por amostra (com o pó), c0 / cl: " + "   ".join(f"{a_}: {x[0]:.1f}/{x[1]:.1f}" for a_, x in med.items()))
    fig.suptitle("tau_medio: C livre contra C = 0  (círculo cheio = canal com o pó; vazado = só vazamento de cor)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "2_tau_cl_vs_c0.png", dpi=p["dpi"], bbox_inches="tight")
    plt.close(fig)
    return L


# ---------------------------------------------------------------- 3. resíduos
def residuos(aj, series) -> pd.DataFrame:
    """Resíduo (dado - ajuste) por ponto, para todos os ajustes de cl e c0."""
    a = _ajustes_validos(aj)
    por_video = {v: g for v, g in series.groupby("video")}
    partes = []
    for _, l in a.iterrows():
        g = por_video.get(l["video"])
        if g is None:
            continue
        d = janela(g, l)
        if d.empty:
            continue
        t_rel = d["tempo_s"].to_numpy() - d["tempo_s"].iloc[0]
        r = d[f"corr_{l['canal']}"].to_numpy() - curva(l, t_rel)
        partes.append(pd.DataFrame({"video": l["video"], "canal": l["canal"], "caminho": l["caminho"], "t_rel": t_rel,
                                    "residuo": r, "ruido": l["ruido"]}))
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()


def grafico_residuos(aj, series, p, out) -> list[str]:
    plt = _plt()
    r = residuos(aj, series)
    if r.empty:
        return []
    passo = p["bin_residuo_s"]
    r["bin"] = (r["t_rel"] // passo) * passo + passo / 2
    fig, eixos = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    L = ["Resíduo (dado - ajuste): mediana por faixa de tempo, em níveis. Ruído do canal = desvio dos últimos 10 s."]
    for i, c in enumerate(CANAIS):
        ruido = float(r[r["canal"] == c]["ruido"].median())
        for j, (rot, cam) in enumerate((("c0 (C = 0)", "principal"), ("cl (C livre)", "c_livre"))):
            ax = eixos[i][j]
            d = r[(r["canal"] == c) & (r["caminho"] == cam)]
            g = d.groupby("bin")["residuo"]
            med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
            ax.fill_between(med.index, q1, q3, color=COR_CANAL[c], alpha=0.25, label="quartis entre vídeos")
            ax.plot(med.index, med, color=COR_CANAL[c], lw=1.6, label="mediana")
            ax.axhline(0, color="black", lw=0.8)
            ax.axhspan(-ruido, ruido, color="gray", alpha=0.15, label=f"±ruído ({ruido:.2f})")
            ax.set_title(f"canal {c} · {rot}", fontsize=10, loc="left", color=COR_CANAL[c]); ax.grid(alpha=0.25)
            if j == 0:
                ax.set_ylabel("resíduo (níveis)")
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
            if i == 2:
                ax.set_xlabel("tempo desde o início da janela (s)")
            mx = float(med.abs().max())
            L.append(f"  {c} {rot}: |mediana do resíduo| máx. {mx:.2f}  ({mx / ruido:.0f}x o ruído); média dos |resíduos| {d['residuo'].abs().mean():.2f}")
    fig.suptitle("Resíduos dos ajustes: estrutura sistemática ao longo do tempo indica forma de modelo inadequada", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "3_residuos.png", dpi=p["dpi"], bbox_inches="tight")
    plt.close(fig)
    return L


# ---------------------------------------------------------------- 4. normalizada
def normalizadas(series, vpos, aj) -> pd.DataFrame:
    """I/I0 por vídeo e canal (I0 = média dos 3 primeiros pontos da janela), só canais com ajuste principal."""
    usados = set(zip(*[_ajustes_validos(aj).query("caminho == 'principal'")[c] for c in ("video", "canal")]))
    ini = aj[aj["caminho"] == "principal"].groupby("video")["inicio_s"].first()
    partes = []
    for v, g in series.groupby("video"):
        if v not in ini.index:
            continue
        d = g[g["tempo_s"] >= ini[v] - 1e-9]
        for c in CANAIS:
            if (v, c) not in usados:
                continue
            y = d[f"corr_{c}"].to_numpy()
            i0 = float(np.mean(y[:3]))
            if i0 <= 0:
                continue
            partes.append(pd.DataFrame({"video": v, "canal": c, "t_rel": d["tempo_s"].to_numpy() - d["tempo_s"].iloc[0], "i_norm": y / i0}))
    n = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()
    if len(n):
        m = vpos.set_index("video")
        n["proporcao"] = n["video"].map(m["proporcao"])
        n["com_po"] = [bool(m.loc[v, f"canal_ativo_{c}"]) for v, c in zip(n["video"], n["canal"])]
    return n


def grafico_normalizada(series, vpos, aj, p, out) -> list[str]:
    plt = _plt()
    n = normalizadas(series, vpos, aj)
    if n.empty:
        return []
    passo = p["bin_normalizada_s"]
    n["bin"] = (n["t_rel"] // passo) * passo + passo / 2
    props = sorted(n["proporcao"].unique())
    cmap = plt.get_cmap("tab20")
    cores = {pr: cmap(i % 20) for i, pr in enumerate(props)}
    fig, eixos = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
    L = ["I/I0 (I0 = média dos 3 primeiros pontos da janela): tempo para cair a 50% e a 10%, mediana por canal (só canais com o pó)"]
    for ax, c in zip(eixos, CANAIS):
        d = n[(n["canal"] == c) & n["com_po"]]
        t50, t10 = [], []
        if d.empty:
            ax.text(0.5, 0.5, f"canal {c}: sem ajustes", ha="center", va="center", transform=ax.transAxes, color="gray")
            continue
        for pr in props:
            x = d[d["proporcao"] == pr]
            if x.empty:
                continue
            curva_med = x.groupby("bin")["i_norm"].median()
            ax.plot(curva_med.index, curva_med, color=cores[pr], lw=1.3, label=pr)
            for lista, nivel in ((t50, 0.5), (t10, 0.1)):
                ab = curva_med[curva_med <= nivel]
                if len(ab):
                    lista.append(float(ab.index[0]))
        ax.set_yscale("log"); ax.set_ylim(0.01, 1.3); ax.grid(alpha=0.25, which="both")
        ax.set_title(f"canal {c} (só proporções com o pó em {c})", fontsize=10, color=COR_CANAL[c])
        ax.set_xlabel("tempo desde o início da janela (s)")
        if c == "R":
            ax.set_ylabel("I / I0")
        ax.legend(fontsize=6, ncol=2, loc="lower left")
        if t50:
            L.append(f"  {c}: t(I/I0 = 0,5) mediana {np.median(t50):.1f} s (min {min(t50):.1f}, máx {max(t50):.1f}); "
                     + (f"t(0,1) mediana {np.median(t10):.1f} s" if t10 else "não chega a 0,1 na janela"))
    fig.suptitle("Decaimento normalizado por proporção (mediana entre os vídeos da proporção)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "4_normalizada.png", dpi=p["dpi"], bbox_inches="tight")
    plt.close(fig)
    return L


def rodar(cfg: dict, pasta_modelagem=None, pasta_pos=None, saida=None) -> int:
    p = parametros(cfg)
    raiz = Path(cfg.get("pastas", {}).get("saida", "output"))
    mod, pos = Path(pasta_modelagem or raiz / "modelagem"), Path(pasta_pos or raiz / "posprocessamento")
    for f in (mod / "ajustes.csv", pos / "series_pos.csv", pos / "videos_pos.csv"):
        if not f.exists():
            raise FileNotFoundError(f"{f} não existe (rode 'luminescencia pos' e 'luminescencia modelar')")
    out = Path(saida or raiz / "graficos")
    out.mkdir(parents=True, exist_ok=True)
    aj = pd.read_csv(mod / "ajustes.csv", encoding="utf-8-sig")
    series = pd.read_csv(pos / "series_pos.csv", encoding="utf-8-sig")
    vpos = pd.read_csv(pos / "videos_pos.csv", encoding="utf-8-sig")
    aj["modelo"] = aj["modelo"].fillna("")
    texto = ["GRÁFICOS", "=" * 60, ""]
    for nome, fn in (("1 semilog", lambda: grafico_semilog(aj, series, p, out)), ("2 tau", lambda: grafico_tau(aj, vpos, p, out)),
                     ("3 resíduos", lambda: grafico_residuos(aj, series, p, out)),
                     ("4 normalizada", lambda: grafico_normalizada(series, vpos, aj, p, out))):
        print(f"gráfico {nome}...")
        texto += fn() + [""]
    (out / "graficos_resumo.txt").write_text("\n".join(texto), encoding="utf-8")
    print("\n".join(texto))
    print(f"Arquivos em {out}")
    return 0
