"""Programa 6a - Ajuste de modelos em lote (C = 0) com sensibilidades.

Entrada: output/posprocessamento/ (series_pos.csv, videos_pos.csv) e output/consolidacao/videos.csv (para o piso fixo).
Saída (output/modelagem/):
  ajustes.csv               uma linha por vídeo x canal x caminho: janela usada, modelo escolhido, parâmetros, R², tau_medio
  relatorio_modelagem.txt   comparação entre caminhos e a repetibilidade entre regravações

Caminhos (decididos em 02/11/2026)
  principal        C = 0 sobre corr_* (preto por vídeo), janela fixa a partir de posprocessamento.inicio_ajuste_s
  inicio_2s/3s     igual ao principal, mas começando nos outros instantes de posprocessamento.janelas_sensibilidade
  piso_fixo        C = 0 sobre media_* menos o piso fixo do controle (consolidacao.piso: referencia). Se o tau quase não
                   mudar em relação ao principal, o resultado não depende da correção do preto
  c_livre          C livre (exp_simples e exp_dupla) sobre corr_*, em todos os vídeos (modelagem.c_livre_num_video: todos;
                   um número limita a esse num_video)
  gama             sensibilidade (D5, 02/11/2026; NÃO é o padrão): C = 0 sobre o sinal linearizado pela gama
                   (modelagem.gama, padrão 2,2; null desliga): 255*((media/255)^g - (piso_video/255)^g). A média da ROI já
                   foi feita em valores de vídeo, então o resultado não é perfeitamente linear (processamento próprio da câmera)
  a_parte          vídeos que saturam depois do início da janela: C = 0 sobre corr_*, começando no 1º quadro depois da
                   última saturação. Janela diferente por vídeo: NÃO misturar com o principal ao treinar a IA

Fim da janela por sinal (todos os canais, mesma regra): o ajuste termina onde a mediana móvel do sinal deixa de passar
de ajuste.ruido_k x ruído do canal (ruído = MAD das diferenças dos últimos 10 s). Serve sobretudo ao R, cuja cauda cai no
ruído. Sinal que dura menos de ajuste.duracao_minima_s depois do início não é ajustado.
Só entram canais com usar_no_ajuste_* = True (principal e sensibilidades) ou com sinal presente (a_parte). A decisão vem do sinal
medido (perto do ruído, duração do sinal), nunca da proporção do nome do arquivo.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .modelos import ajustar_serie
from .posprocessamento import CANAIS, _sufixo, parametros as parametros_pos

MODELOS_C0 = ("exp_simples_c0", "exp_dupla_c0")
MODELOS_CLIVRE = ("exp_simples", "exp_dupla")
PARAMS_COLS = ["A", "tau", "A1", "tau1", "A2", "tau2", "C", "fracao_rapida", "tau_medio", "r2", "rmse", "bic"]


def parametros(cfg: dict) -> dict:
    p = {"ruido_k": 3.0, "janela_mediana_pontos": 31, "duracao_minima_s": 5.0, "c_livre_num_video": "todos", "workers": 4,
         "gama": 2.2}
    p.update({k: v for k, v in (cfg.get("modelagem", {}) or {}).items()})
    return p


def linearizar_gama(media, piso, gama: float) -> np.ndarray:
    """Sinal em escala linear de luz (0-255): 255*((media/255)^g - (piso/255)^g). Sem o preto, como o corr_*."""
    m = np.clip(np.asarray(media, float), 0, None) / 255.0
    return 255.0 * (m ** gama - (max(float(piso), 0.0) / 255.0) ** gama)


def ruido_do_canal(y: np.ndarray, t: np.ndarray, ultimos_s: float = 10.0) -> float:
    """Desvio padrão do ruído pelos últimos segundos (MAD das diferenças / raiz de 2)."""
    cauda = y[t >= t[-1] - ultimos_s]
    if len(cauda) < 5:
        cauda = y
    d = np.diff(cauda)
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))


def fim_util(t: np.ndarray, y: np.ndarray, k: float, janela: int) -> tuple[float, float]:
    """Último instante em que a mediana móvel do sinal ainda passa de k x ruído. Retorna (fim_s, ruido)."""
    ruido = ruido_do_canal(y, t)
    med = pd.Series(y).rolling(janela, center=True, min_periods=max(3, janela // 2)).median().to_numpy()
    acima = np.where(med > k * max(ruido, 1e-9))[0]
    return (float(t[acima.max()]) if len(acima) else float("nan")), ruido


def _um_ajuste(args) -> dict:
    """Ajusta uma série (função de nível de módulo para poder rodar em outro processo)."""
    ident, t, y, inicio, modelos, k, janela, dur_min = args
    t, y = np.asarray(t, float), np.asarray(y, float)
    sel = t >= inicio - 1e-9
    t, y = t[sel], y[sel]
    L = {**ident, "inicio_s": round(inicio, 3)}
    if len(t) < 8:
        return {**L, "modelo": "", "observacao": "poucos pontos depois do início"}
    fim, ruido = fim_util(t, y, k, janela)
    L.update(ruido=round(ruido, 3))
    if not fim == fim or fim - t[0] < dur_min:
        return {**L, "fim_s": fim, "modelo": "", "observacao": f"sinal some no ruído antes de {dur_min:g} s de janela"}
    sel = t <= fim + 1e-9
    r = ajustar_serie(t[sel] - t[0], y[sel], modelos)
    r.pop("n_pontos", None)
    return {**L, "fim_s": round(fim, 3), "n_pontos": int(sel.sum()), **r}


def montar_tarefas(series, vpos, vcons, cfg: dict, p: dict, pp: dict) -> list[tuple]:
    k, jan, dur = p["ruido_k"], p["janela_mediana_pontos"], p["duracao_minima_s"]
    inicio = pp["inicio_ajuste_s"]
    outros = [x for x in pp["janelas_sensibilidade"] if abs(x - inicio) > 1e-9]
    piso_ref = vcons.set_index("video")
    por_video = {v: g for v, g in series.groupby("video")}
    tarefas = []
    for _, v in vpos.iterrows():
        g = por_video.get(v["video"])
        if g is None:
            continue
        t = g["tempo_s"].to_numpy()
        for c in CANAIS:
            base = {"video": v["video"], "amostra": v["amostra"], "grupo": v["grupo"], "num_video": v["num_video"],
                    "proporcao": v["proporcao"], "canal": c}
            corr = g[f"corr_{c}"].to_numpy()
            if v[f"usar_no_ajuste_{c}"]:
                tarefas.append(({**base, "caminho": "principal"}, t, corr, inicio, MODELOS_C0, k, jan, dur))
                for x in outros:
                    tarefas.append(({**base, "caminho": f"inicio_{_sufixo(x)}"}, t, corr, x, MODELOS_C0, k, jan, dur))
                ref = g[f"media_{c}"].to_numpy() - float(piso_ref.loc[v["video"], f"piso_ref_{c}"])
                tarefas.append(({**base, "caminho": "piso_fixo"}, t, ref, inicio, MODELOS_C0, k, jan, dur))
                if p.get("gama"):
                    lin = linearizar_gama(g[f"media_{c}"].to_numpy(), v[f"piso_video_{c}"], float(p["gama"]))
                    tarefas.append(({**base, "caminho": "gama"}, t, lin, inicio, MODELOS_C0, k, jan, dur))
                if str(p["c_livre_num_video"]).lower() in ("todos", "none", "") or int(v["num_video"]) == int(p["c_livre_num_video"]):
                    tarefas.append(({**base, "caminho": "c_livre"}, t, corr, inicio, MODELOS_CLIVRE, k, jan, dur))
            elif v[f"sinal_presente_{c}"] and v["sat_ate_s"] >= inicio:
                apos = t[t > v["sat_ate_s"] + 1e-9]
                if len(apos):
                    tarefas.append(({**base, "caminho": "a_parte"}, t, corr, float(apos[0]), MODELOS_C0, k, jan, dur))
    return tarefas


def _tau_por_canal(aj: pd.DataFrame, caminho: str) -> pd.Series:
    d = aj[(aj["caminho"] == caminho) & (aj["modelo"] != "")]
    return d.set_index(["video", "canal"])["tau_medio"]


def _relatorio(aj: pd.DataFrame, p: dict, pp: dict) -> str:
    L = ["MODELAGEM (C = 0)", "=" * 60, f"Ajustes: {len(aj)}   com modelo: {int((aj['modelo'] != '').sum())}", ""]
    L.append("Caminhos e modelo escolhido (contagem):")
    for cam, g in aj.groupby("caminho"):
        m = g["modelo"].replace("", "sem ajuste").value_counts().to_dict()
        L.append(f"  {cam}: {m}")
    L += ["", "Sem ajuste (motivos):"]
    sem = aj[aj["modelo"] == ""]
    L += [f"  {cam}/{c}: {n} ({o})" for (cam, c, o), n in sem.groupby(["caminho", "canal", "observacao"]).size().items()] or ["  nenhum"]
    pr = _tau_por_canal(aj, "principal")
    L += ["", "Repetibilidade do tau_medio entre as 5 regravações (CV mediano = desvio/média, por proporção e amostra; menor = mais estável):"]
    for cam in ("principal", "c_livre", "gama"):
        pa = aj[(aj["caminho"] == cam) & (aj["modelo"] != "")]
        partes = []
        for c in CANAIS:
            cv = pa[pa["canal"] == c].groupby(["proporcao", "amostra"])["tau_medio"].agg(lambda x: x.std(ddof=1) / x.mean() if len(x) > 1 else np.nan)
            if cv.notna().any():
                partes.append(f"{c} {cv.median():.1%}")
        if partes:
            L.append(f"  {cam}: " + "   ".join(partes))
    L += ["", "tau_medio mediano por amostra (diferença entre amostras = possível viés de sessão que a IA poderia aprender):"]
    for cam in ("principal", "c_livre", "gama"):
        pa = aj[(aj["caminho"] == cam) & (aj["modelo"] != "")]
        if len(pa):
            tab = pa.groupby(["canal", "amostra"])["tau_medio"].median().round(1).unstack()
            L.append(f"  {cam}: " + "   ".join(f"{c}: " + "/".join(f"{a}={tab.loc[c, a]}" for a in tab.columns) for c in CANAIS if c in tab.index))
    cl = aj[(aj["caminho"] == "c_livre") & (aj["modelo"] != "") & aj["C"].notna()]
    if len(cl):
        L += ["", "Patamar C do ajuste livre (níveis, mediana) contra o erro do ajuste (rmse mediano) com C = 0 e com C livre:"]
        m = cl.merge(aj[aj["caminho"] == "principal"][["video", "canal", "rmse"]], on=["video", "canal"], suffixes=("_livre", "_c0"))
        for c in CANAIS:
            x = m[m["canal"] == c]
            if len(x):
                L.append(f"  {c}: C = {x['C'].median():.2f}   rmse C=0: {x['rmse_c0'].median():.2f}   rmse C livre: {x['rmse_livre'].median():.2f}   "
                         f"ruído do canal: {aj[(aj['caminho'] == 'principal') & (aj['canal'] == c)]['ruido'].median():.2f}")
    L += ["", "Sensibilidade: diferença relativa mediana do tau_medio contra o principal (mesmo vídeo e canal):"]
    for cam in sorted(set(aj["caminho"]) - {"principal", "a_parte"}):
        o = _tau_por_canal(aj, cam)
        comum = pr.index.intersection(o.index)
        if not len(comum):
            continue
        d = ((o[comum] - pr[comum]) / pr[comum])
        por = {c: f"{d.xs(c, level='canal').abs().median():.1%} (viés {d.xs(c, level='canal').median():+.1%}, n={int((d.index.get_level_values('canal') == c).sum())})"
               for c in CANAIS if (d.index.get_level_values('canal') == c).any()}
        L.append(f"  {cam}: {por}")
    L.append("  Regra: se a diferença for MAIOR que o CV entre regravações, o caminho muda o resultado além do ruído.")
    if p.get("gama"):
        L.append(f"  gama: sobre o sinal linearizado (gama {p['gama']:g}). Se o decaimento em valores de vídeo fosse uma exponencial simples, o tau linear seria "
                 f"tau/{p['gama']:g} (diferença de {1 - 1 / p['gama']:.0%} para menos): essa parte é só mudança de escala. O que importa é o CV entre "
                 "regravações e a diferença entre amostras acima, comparados com o principal.")
    ap = aj[aj["caminho"] == "a_parte"]
    if len(ap):
        L += ["", "Ajustados à parte (janela própria; não misturar com o principal):"]
        for r in ap.itertuples():
            L.append(f"  {r.video} {r.canal}: início {r.inicio_s} s, {r.modelo or 'sem ajuste'}, tau_medio {getattr(r, 'tau_medio', float('nan')):.1f}")
    return "\n".join(L) + "\n"


def rodar(cfg: dict, pasta_pos=None, pasta_consolidacao=None, saida=None) -> int:
    p, pp = parametros(cfg), parametros_pos(cfg)
    raiz = Path(cfg.get("pastas", {}).get("saida", "output"))
    pos = Path(pasta_pos or raiz / "posprocessamento")
    cons = Path(pasta_consolidacao or raiz / "consolidacao")
    for f in (pos / "series_pos.csv", pos / "videos_pos.csv", cons / "videos.csv"):
        if not f.exists():
            raise FileNotFoundError(f"{f} não existe (rode 'luminescencia consolidar' e 'luminescencia pos')")
    out = Path(saida or raiz / "modelagem")
    out.mkdir(parents=True, exist_ok=True)
    series = pd.read_csv(pos / "series_pos.csv", encoding="utf-8-sig")
    vpos = pd.read_csv(pos / "videos_pos.csv", encoding="utf-8-sig")
    vcons = pd.read_csv(cons / "videos.csv", encoding="utf-8-sig")
    tarefas = montar_tarefas(series, vpos, vcons, cfg, p, pp)
    print(f"{len(tarefas)} ajustes a fazer (workers: {p['workers']})...")
    if p["workers"] > 1:
        with ProcessPoolExecutor(p["workers"]) as ex:
            linhas = list(ex.map(_um_ajuste, tarefas, chunksize=20))
    else:
        linhas = [_um_ajuste(t) for t in tarefas]
    aj = pd.DataFrame(linhas)
    for c in ("modelo", "observacao"):
        aj[c] = aj[c].fillna("")
    ordem = ["video", "amostra", "grupo", "num_video", "proporcao", "canal", "caminho", "inicio_s", "fim_s", "n_pontos", "ruido",
             "modelo"] + PARAMS_COLS + ["r2_exp_simples_c0", "r2_exp_dupla_c0", "observacao"]
    aj = aj[[c for c in ordem if c in aj.columns] + [c for c in aj.columns if c not in ordem]]
    aj.to_csv(out / "ajustes.csv", index=False, encoding="utf-8-sig")
    texto = _relatorio(aj, p, pp)
    (out / "relatorio_modelagem.txt").write_text(texto, encoding="utf-8")
    print(texto)
    print(f"Arquivos em {out}: ajustes.csv, relatorio_modelagem.txt")
    return 0
