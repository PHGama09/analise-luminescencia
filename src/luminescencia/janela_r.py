"""Q4/B12 - Fim de janela específico do R (só análise; não muda dataset.csv nem descritores.csv).

Entrada: output/posprocessamento/ (series_pos.csv, videos_pos.csv). Opcional: output/modelagem/ajustes.csv (confere que a regra
atual é reproduzida) e output/tabela/dataset.csv (lista as colunas que mudariam).
Saída (output/janela_r/): resumo_regras.csv, sinais.csv e relatorio_janela_r.txt.

Cada regra de fim de janela é aplicada só ao canal indicado (padrão R) e o sinal é REAJUSTADO com os mesmos modelos e a mesma
regra de escolha do `modelar`: C = 0 (exp_simples_c0/exp_dupla_c0) e C livre com c_minimo (exp_simples/exp_dupla). O início da
janela é o mesmo do pipeline (posprocessamento.inicio_ajuste_s). A regra de base é a do pipeline (k = modelagem.ruido_k, sem
duração máxima); as outras são comparadas a ela. Nenhum número do relatório vem do código: tudo é calculado dos CSVs.

Regra = k x ruído (fim onde a mediana móvel deixa de passar de k x ruído, como em modelar.fim_util) e, opcionalmente, duração máxima
(o fim é o menor entre o da regra de ruído e início + duração). Assim a duração máxima nunca estica o fim além do sinal.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .modelar import MODELOS_C0, MODELOS_CLIVRE, TOL_LIMITE_C, fim_util, parametros as parametros_modelar
from .modelos import ajustar_serie
from .posprocessamento import parametros as parametros_pos

AJUSTES = {"cl": MODELOS_CLIVRE, "c0": MODELOS_C0}


def parametros(cfg: dict) -> dict:
    p = {"canal": "R",
         "regras": [{"k": 3.0}, {"k": 4.0}, {"k": 5.0}, {"k": 6.0}, {"k": 3.0, "duracao_max_s": 20.0}, {"k": 3.0, "duracao_max_s": 30.0}],
         "min_n_por_grupo": 3, "dt_fim_min_s": 0.05, "t50_pontos_i0": 5, "t50_suavizacao_pontos": 5, "workers": None,
         "criterios": {"max_aumento_nao_ajustavel": 0.02, "melhora_cv_minima": 0.10, "max_piora_discrepancia": 0.0}}
    p.update({k: v for k, v in (cfg.get("janela_r", {}) or {}).items() if k != "criterios"})
    p["criterios"] = {**p["criterios"], **((cfg.get("janela_r", {}) or {}).get("criterios", {}) or {})}
    return p


def nome_regra(r: dict) -> str:
    d = r.get("duracao_max_s")
    return f"k{float(r['k']):g}" + (f"_d{float(d):g}s" if d is not None else "")


def normalizar_regras(regras: list[dict], k_base: float) -> list[dict]:
    """Regra de base primeiro (k = ruido_k do pipeline, sem duração máxima; incluída se faltar), sem repetidas."""
    base = {"k": float(k_base), "duracao_max_s": None}
    vistos, saida = {nome_regra(base)}, [{**base, "nome": nome_regra(base)}]
    for r in regras:
        r = {"k": float(r["k"]), "duracao_max_s": (float(r["duracao_max_s"]) if r.get("duracao_max_s") is not None else None)}
        r["nome"] = nome_regra(r)
        if r["nome"] not in vistos:
            vistos.add(r["nome"])
            saida.append(r)
    return saida


def fim_da_regra(t: np.ndarray, y: np.ndarray, regra: dict, janela: int) -> tuple[float, float]:
    """(fim_s, ruído) da regra para uma série que já começa no início da janela. fim = nan se o sinal nunca passa de k x ruído."""
    fim, ruido = fim_util(t, y, float(regra["k"]), janela)
    d = regra.get("duracao_max_s")
    if d is not None and fim == fim:
        fim = min(fim, float(t[0]) + float(d))
    return fim, ruido


def t50_livre(t, y, n0: int = 5, suav: int = 5) -> float:
    """Tempo (desde o início da janela) em que a mediana móvel cai a metade da mediana dos n0 primeiros pontos. nan se não cai."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    if len(y) < max(n0, suav) + 1:
        return float("nan")
    s = pd.Series(y).rolling(suav, center=True, min_periods=1).median().to_numpy()
    i0 = float(np.median(y[:n0]))
    if not i0 > 0:
        return float("nan")
    alvo = i0 / 2
    idx = np.where(s <= alvo)[0]
    if not len(idx) or idx[0] == 0:
        return float("nan")
    j = idx[0]
    y1, y2 = s[j - 1], s[j]
    t_alvo = t[j - 1] + (y1 - alvo) * (t[j] - t[j - 1]) / (y1 - y2) if y1 != y2 else t[j]
    return float(t_alvo - t[0])


def _ajustar_janela(args):
    chave, t, y, modelos, c_min = args
    r = ajustar_serie(t, y, modelos, c_minimo=c_min)
    r.pop("n_pontos", None)
    if c_min is not None and r.get("C") is not None and r["C"] == r["C"]:
        r["C_no_limite"] = bool(r["C"] <= c_min + TOL_LIMITE_C)
    return chave, r


def montar_janelas(series: pd.DataFrame, usar: set, regras: list[dict], inicio: float, jan: int, dur_min: float, canal: str, p: dict):
    """Uma linha por vídeo x regra (janela, ruído, cauda, t50) e as tarefas de ajuste sem repetição (mesmo vídeo e mesmo fim = um ajuste)."""
    linhas, tarefas = [], {}
    ident_cols = [c for c in ("video", "amostra", "grupo", "num_video", "proporcao") if c in series.columns]
    for video, g in series.groupby("video", sort=False):
        if video not in usar:
            continue
        t_all, y_all = g["tempo_s"].to_numpy(float), g[f"corr_{canal}"].to_numpy(float)
        sel = t_all >= inicio - 1e-9
        t, y = t_all[sel], y_all[sel]
        ident = {c: g[c].iloc[0] for c in ident_cols}
        for r in regras:
            L = {**ident, "regra": r["nome"], "inicio_s": round(inicio, 3)}
            if len(t) < 8:
                linhas.append({**L, "fim_s": np.nan, "ajustavel": False, "motivo": "poucos pontos depois do início"})
                continue
            fim, ruido = fim_da_regra(t, y, r, jan)
            L["ruido"] = ruido
            if not fim == fim:
                linhas.append({**L, "fim_s": np.nan, "ajustavel": False, "motivo": "sinal nunca passa do limiar"})
                continue
            w = t <= fim + 1e-9
            tw, yw = t[w], y[w]
            L.update(fim_s=round(fim, 3), duracao_s=round(float(tw[-1] - tw[0]), 3), n_pontos=int(w.sum()),
                     frac_nao_positivo=float((yw <= 0).mean()),
                     t50_s=t50_livre(tw, yw, p["t50_pontos_i0"], p["t50_suavizacao_pontos"]))
            L["ajustavel"] = bool(fim - t[0] >= dur_min)
            L["motivo"] = "" if L["ajustavel"] else f"sinal some no ruído antes de {dur_min:g} s de janela"
            if L["ajustavel"]:
                chave = (video, round(fim, 6))
                L["_chave"] = chave
                for s, modelos in AJUSTES.items():
                    tarefas.setdefault((chave, s), (chave, tw - tw[0], yw, modelos, p["_c_min"] if s == "cl" else None))
            linhas.append(L)
    return linhas, tarefas


def ajustar_tudo(tarefas: dict, workers: int) -> dict:
    lista = list(tarefas.items())
    args = [v for _, v in lista]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_ajustar_janela, args, chunksize=4))
    else:
        res = [_ajustar_janela(a) for a in args]
    return {(chave_s[0], chave_s[1]): r for (chave_s, _), (_, r) in zip(lista, res)}


def juntar(linhas: list[dict], ajustes: dict) -> pd.DataFrame:
    for L in linhas:
        chave = L.pop("_chave", None)
        ruido = L.get("ruido", np.nan)
        for s in AJUSTES:
            r = ajustes.get(((chave, s))) if chave else None
            if r is None:
                continue
            for c in ("modelo", "tau_medio", "r2", "rmse", "C", "C_no_limite", "observacao"):
                if c in r:
                    L[f"{s}_{c}"] = r[c]
            if r.get("rmse") is not None and ruido == ruido and ruido > 0:
                L[f"{s}_rmse_ruido"] = r["rmse"] / ruido
    df = pd.DataFrame(linhas)
    for s in AJUSTES:                      # ajuste que não rodou ou não convergiu = sem tau
        if f"{s}_modelo" in df.columns:
            df.loc[df[f"{s}_modelo"].fillna("") == "", f"{s}_tau_medio"] = np.nan
    return df


def cv_medio_por_grupo(df: pd.DataFrame, coluna: str, min_n: int, grupo=("proporcao", "amostra")) -> tuple[float, int]:
    """Mediana dos CV (desvio com ddof=1 / média) entre regravações, por grupo com pelo menos min_n valores. Retorna (mediana, n_grupos)."""
    cvs = []
    for _, g in df.groupby(list(grupo)):
        v = g[coluna].dropna().to_numpy(float)
        if len(v) >= min_n and v.mean() != 0:
            cvs.append(v.std(ddof=1) / abs(v.mean()))
    return (float(np.median(cvs)) if cvs else float("nan")), len(cvs)


def _med(x) -> float:
    x = pd.Series(x).dropna()
    return float(x.median()) if len(x) else float("nan")


def resumir(sinais: pd.DataFrame, nome_base: str, p: dict, frac_nao_positivo_max: float) -> pd.DataFrame:
    base = sinais[sinais["regra"] == nome_base].set_index("video")
    linhas = []
    for nome, g in sinais.groupby("regra", sort=False):
        aj = g[g["ajustavel"]]
        L = {"regra": nome, "n_sinais": len(g), "n_ajustaveis": len(aj), "frac_nao_ajustavel": 1 - len(aj) / len(g) if len(g) else np.nan,
             "frac_cauda_no_ruido": float((aj["frac_nao_positivo"] > frac_nao_positivo_max).mean()) if len(aj) else np.nan,
             "duracao_mediana_s": _med(aj.get("duracao_s")), "frac_t50_definido": float(aj["t50_s"].notna().mean()) if len(aj) else np.nan,
             "t50_mediana_s": _med(aj["t50_s"]), "t50_cv_mediano": cv_medio_por_grupo(aj, "t50_s", p["min_n_por_grupo"])[0]}
        gi = g.set_index("video")
        for s in AJUSTES:
            col = f"{s}_tau_medio"
            if col not in g.columns:
                continue
            cv, ng = cv_medio_por_grupo(g, col, p["min_n_por_grupo"])
            L.update({f"{s}_tau_mediana": _med(g[col]), f"{s}_tau_cv_mediano": cv, f"{s}_n_grupos_cv": ng,
                      f"{s}_rmse_ruido_mediano": _med(g.get(f"{s}_rmse_ruido")),
                      f"{s}_dtau_rel_vs_base": _med((gi[col] / base[col].reindex(gi.index) - 1).abs())})
            if s == "cl" and "cl_C_no_limite" in g.columns:
                ok = g[col].notna()
                L["cl_frac_C_no_limite"] = float(g.loc[ok, "cl_C_no_limite"].astype(bool).mean()) if ok.any() else np.nan
        if {"cl_tau_medio", "c0_tau_medio"} <= set(g.columns):
            L["disc_cl_c0_mediana"] = _med((g["cl_tau_medio"] / g["c0_tau_medio"] - 1).abs())
        linhas.append(L)
    return pd.DataFrame(linhas)


def recomendar(resumo: pd.DataFrame, nome_base: str, crit: dict) -> tuple[str | None, list[str]]:
    """Regra que atende TODOS os critérios do config (limiares de decisão) e tem o menor CV médio (cl, c0); None se nenhuma."""
    b = resumo[resumo["regra"] == nome_base].iloc[0]
    expl, cand = [], []
    for _, r in resumo.iterrows():
        if r["regra"] == nome_base:
            continue
        falhas = []
        if not r["frac_nao_ajustavel"] - b["frac_nao_ajustavel"] <= crit["max_aumento_nao_ajustavel"] + 1e-12:
            falhas.append("deixa de ajustar sinais demais")
        for s in AJUSTES:
            if not r[f"{s}_tau_cv_mediano"] <= b[f"{s}_tau_cv_mediano"] * (1 - crit["melhora_cv_minima"]):
                falhas.append(f"CV do tau {s} não melhora o suficiente")
        if not r["disc_cl_c0_mediana"] <= b["disc_cl_c0_mediana"] + crit["max_piora_discrepancia"]:
            falhas.append("a diferença cl x c0 aumenta")
        expl.append(f"  {r['regra']}: " + ("atende todos os critérios" if not falhas else "; ".join(falhas)))
        if not falhas:
            cand.append((0.5 * (r["cl_tau_cv_mediano"] + r["c0_tau_cv_mediano"]), r["regra"]))
    return (min(cand)[1] if cand else None), expl


def conferir_base(sinais: pd.DataFrame, aj: pd.DataFrame, nome_base: str, canal: str) -> list[str]:
    """Compara a regra de base com o ajustes.csv do modelar (principal = c0, c_livre = cl): tem de dar o mesmo."""
    b = sinais[sinais["regra"] == nome_base].set_index("video")
    L = []
    for caminho, s in (("principal", "c0"), ("c_livre", "cl")):
        r = aj[(aj["caminho"] == caminho) & (aj["canal"] == canal) & (aj["modelo"].fillna("") != "")].set_index("video")
        comum = r.index.intersection(b.index[b[f"{s}_tau_medio"].notna()])
        if not len(comum):
            L.append(f"  {caminho}: nenhum vídeo em comum")
            continue
        d = ((b.loc[comum, f"{s}_tau_medio"] / r.loc[comum, "tau_medio"]) - 1).abs()
        so_um = len(set(r.index) ^ set(b.index[b[f"{s}_tau_medio"].notna()]))
        L.append(f"  {caminho}: {len(comum)} sinais pareados; maior diferença relativa do tau_medio = {d.max():.2e}; vídeos só de um lado = {so_um}")
    return L


def _fmt(df: pd.DataFrame, colunas: dict, casas: int = 3) -> str:
    d = df[["regra"] + list(colunas)].rename(columns=colunas).round(casas)
    return d.to_string(index=False)


def _relatorio(resumo, sinais, nome_base, p, pp, rec, expl, conf, mudaria, aviso_serie=None) -> str:
    n_sin = int(resumo.iloc[0]["n_sinais"])
    L = [f"Q4/B12 - Fim de janela do canal {p['canal']} (só análise; {n_sin} sinais com usar_no_ajuste_{p['canal']}; início da janela {pp['inicio_ajuste_s']:g} s)",
         f"Regra de base (pipeline): {nome_base}. Regras testadas: {', '.join(resumo['regra'])}.",
         "tau = tau_medio; C livre com c_minimo do modelar; mesmo critério de escolha simples/dupla do pipeline.", ""]
    if aviso_serie:
        L += [f"Contexto: na série inteira (até o fim do vídeo), {aviso_serie[0]} de {aviso_serie[1]} sinais têm mais de {pp['frac_nao_positivo_max']:.0%} dos pontos corrigidos <= 0 "
              "(aviso do pos). A tabela 1 mostra o que sobra DENTRO da janela de ajuste de cada regra.", ""]
    if conf:
        L += ["Conferência da regra de base contra output/modelagem/ajustes.csv (tem de coincidir):"] + conf + [""]
    L += ["1) Sinais e cauda no ruído (frac_cauda = ajustáveis com mais de {:.0%} dos pontos corrigidos <= 0)".format(pp["frac_nao_positivo_max"]),
          _fmt(resumo, {"n_ajustaveis": "ajust.", "frac_nao_ajustavel": "nao_ajust", "frac_cauda_no_ruido": "cauda", "duracao_mediana_s": "dur_med_s",
                        "disc_cl_c0_mediana": "disc_cl_c0"}), "",
          "2) tau_medio (mediana; CV = mediana dos CV entre regravações por proporção x amostra; rmse/ruido; mudança relativa do tau frente à base)"]
    for s in AJUSTES:
        L += [f" C {'livre' if s == 'cl' else '= 0'}:",
              _fmt(resumo, {f"{s}_tau_mediana": "tau_med", f"{s}_tau_cv_mediano": "CV", f"{s}_n_grupos_cv": "n_grupos",
                            f"{s}_rmse_ruido_mediano": "rmse/ruido", f"{s}_dtau_rel_vs_base": "|dtau|_base"})]
    if "cl_frac_C_no_limite" in resumo.columns:
        L += ["", "   fração de ajustes C livre colados em c_minimo: " + ", ".join(f"{r.regra} {r.cl_frac_C_no_limite:.1%}" for r in resumo.itertuples())]
    L += ["", "3) t50 sem modelo (reimplementado aqui: mediana móvel cai à metade da mediana dos primeiros pontos; só varia com o fim da janela se a regra corta antes de 50%)",
          _fmt(resumo, {"frac_t50_definido": "def", "t50_mediana_s": "t50_med_s", "t50_cv_mediano": "CV"}), "",
          "4) Recomendação (critérios do config: " + ", ".join(f"{k}={v:g}" for k, v in p["criterios"].items()) + ")"] + expl
    if rec:
        L.append(f"  -> Regra que atende todos os critérios com menor CV médio: {rec}.")
    else:
        L.append("  -> Nenhuma regra atende todos os critérios: manter a regra atual para o R (o dataset não muda).")
    if mudaria:
        L += ["", f"5) O que mudaria no dataset com {mudaria['regra']}:"] + [f"  {x}" for x in mudaria["linhas"]]
    L += ["", "Só sensibilidade: o padrão (modelagem.ruido_k) não foi alterado; nenhum número de dataset.csv/descritores.csv mudou."]
    return "\n".join(L) + "\n"


def o_que_mudaria(sinais: pd.DataFrame, nome_base: str, regra: str, p: dict, pasta_tabela: Path | None, canal: str) -> dict:
    b = sinais[sinais["regra"] == nome_base].set_index("video")
    r = sinais[sinais["regra"] == regra].set_index("video").reindex(b.index)
    L = [f"sinais com fim de janela diferente (> {p['dt_fim_min_s']:g} s): {int(((r['fim_s'] - b['fim_s']).abs() > p['dt_fim_min_s']).sum())} de {len(b)}",
         f"deixam de ser ajustáveis: {int((b['ajustavel'] & ~r['ajustavel']).sum())}; passam a ser ajustáveis: {int((~b['ajustavel'] & r['ajustavel']).sum())}"]
    for s in AJUSTES:
        d = (r[f"{s}_tau_medio"] / b[f"{s}_tau_medio"] - 1).abs().dropna()
        if len(d):
            L.append(f"tau {s}: mediana |variação| {d.median():.1%}, máxima {d.max():.1%}, {int((d > 0.10).sum())} sinais com mais de 10%")
    if pasta_tabela and (pasta_tabela / "dataset.csv").exists():
        cols = pd.read_csv(pasta_tabela / "dataset.csv", nrows=0, encoding="utf-8-sig").columns
        afet = [c for c in cols if c.startswith(f"{canal}_cl_") or c.startswith(f"{canal}_c0_")]
        L.append(f"colunas do dataset.csv que podem mudar ({len(afet)}): {', '.join(afet[:6])}{', ...' if len(afet) > 6 else ''}")
    return {"regra": regra, "linhas": L}


def rodar(cfg: dict, pasta_pos=None, pasta_modelagem=None, pasta_tabela=None, saida=None) -> int:
    p, pm, pp = parametros(cfg), parametros_modelar(cfg), parametros_pos(cfg)
    raiz = Path(cfg.get("pastas", {}).get("saida", "output"))
    pos, mod = Path(pasta_pos or raiz / "posprocessamento"), Path(pasta_modelagem or raiz / "modelagem")
    tab = Path(pasta_tabela or raiz / "tabela")
    for f in (pos / "series_pos.csv", pos / "videos_pos.csv"):
        if not f.exists():
            raise FileNotFoundError(f"{f} não existe (rode 'luminescencia consolidar' e 'luminescencia pos')")
    out = Path(saida or raiz / "janela_r")
    out.mkdir(parents=True, exist_ok=True)
    canal = p["canal"]
    series = pd.read_csv(pos / "series_pos.csv", encoding="utf-8-sig")
    vpos = pd.read_csv(pos / "videos_pos.csv", encoding="utf-8-sig")
    usar = set(vpos.loc[vpos[f"usar_no_ajuste_{canal}"].astype(bool), "video"])
    regras = normalizar_regras(p["regras"], pm["ruido_k"])
    p["_c_min"] = pm.get("c_livre_c_minimo")
    workers = int(p["workers"] or pm["workers"])
    workers = max(1, min(workers, os.cpu_count() or 1))
    linhas, tarefas = montar_janelas(series, usar, regras, pp["inicio_ajuste_s"], pm["janela_mediana_pontos"], pm["duracao_minima_s"], canal, p)
    print(f"{len(usar)} sinais {canal} x {len(regras)} regras: {len(tarefas)} ajustes distintos (workers: {workers})...")
    sinais = juntar(linhas, ajustar_tudo(tarefas, workers))
    nome_base = regras[0]["nome"]
    resumo = resumir(sinais, nome_base, p, pp["frac_nao_positivo_max"])
    rec, expl = recomendar(resumo, nome_base, p["criterios"])
    alvo = rec
    if alvo is None and len(resumo) > 1:                     # sem recomendada: mostra o que mudaria com a de menor CV médio, só para informar
        outras = resumo[resumo["regra"] != nome_base]
        alvo = outras.loc[(outras["cl_tau_cv_mediano"] + outras["c0_tau_cv_mediano"]).idxmin(), "regra"]
    mudaria = o_que_mudaria(sinais, nome_base, alvo, p, tab, canal) if alvo else None
    if mudaria and rec is None:
        mudaria["linhas"].insert(0, "(regra NÃO recomendada; é a de menor CV médio, mostrada só para dimensionar a mudança)")
    conf = conferir_base(sinais, pd.read_csv(mod / "ajustes.csv", encoding="utf-8-sig"), nome_base, canal) if (mod / "ajustes.csv").exists() else []
    col = f"frac_nao_positivo_{canal}"
    aviso = (int((vpos.loc[vpos["video"].isin(usar), col] > pp["frac_nao_positivo_max"]).sum()), len(usar)) if col in vpos.columns else None
    sinais.to_csv(out / "sinais.csv", index=False, encoding="utf-8-sig")
    resumo.to_csv(out / "resumo_regras.csv", index=False, encoding="utf-8-sig")
    texto = _relatorio(resumo, sinais, nome_base, p, pp, rec, expl, conf, mudaria, aviso)
    (out / "relatorio_janela_r.txt").write_text(texto, encoding="utf-8")
    print(texto)
    print(f"Arquivos em {out}: sinais.csv, resumo_regras.csv, relatorio_janela_r.txt")
    return 0
