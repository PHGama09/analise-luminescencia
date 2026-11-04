"""Q4/B11 - Saturados (caminho a_parte) x restante. SÓ ANÁLISE: não muda dataset.csv nem descritores.csv.

Pergunta: os vídeos que saturam depois do início da janela (ajuste só a_parte, janela própria) podem ser usados?

Método (tudo calculado dos CSVs; nenhum número ou conclusão fixa no código)
  1. Saturados = vídeos com sat_ate_s >= posprocessamento.inicio_ajuste_s (mesma regra de modelar.py). A janela de cada
     canal é recalculada com as regras de modelar.py (1º quadro depois da última saturação; fim pelo ruído) e conferida
     com o ajustes.csv (caminho a_parte), se ele existir.
  2. Irmãos = mesma proporção e outros vídeos, reajustados NA MESMA JANELA [início, fim] do saturado (C = 0, mesmos modelos
     do pipeline). Só entra irmão sem quadro saturado dentro da janela. Dois conjuntos: mesma_amostra (preferido) e
     outras_amostras (confundido pela deriva de sessão A/B/C: no máximo "duvidoso").
  3. Por canal e grandeza (tau_medio, t50, I0): razão saturado/mediana dos irmãos e posição z = (razão - 1) / CV dos irmãos
     (CV com piso cv_piso). t50 e I0 são sem modelo, medidos a partir do início da janela.
  4. Classe por canal pelos limiares do config (z_comparavel, z_duvidoso; razao_maxima_comparavel) sobre as grandezas de metricas_veredito; classe
     do vídeo = a pior entre os canais de decisão (canal que a maioria dos vídeos da proporção sem saturação usa no ajuste).
  5. Efeito no grupo: para cada proporção x amostra com poucos vídeos de treino sem saturação (< min_n_por_amostra), os
     vídeos de todas as amostras e proporções são reajustados numa janela comum e a razão entre amostras da proporção alvo
     é posta contra a razão entre as mesmas amostras das outras proporções (a deriva A/B "normal" na mesma janela).

Saída (output/saturados/): saturados_janelas.csv, irmaos_ajustes.csv, comparacao.csv, veredito_canal.csv, veredito_video.csv,
treino_por_grupo.csv, efeito_grupo.csv, relatorio_saturados.txt
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .modelar import MODELOS_C0, fim_util, parametros as parametros_modelagem
from .modelos import ajustar_serie
from .posprocessamento import CANAIS, parametros as parametros_pos

PADRAO = {
    "min_irmaos": 3,                       # irmãos com ajuste válido exigidos para uma base de comparação
    "z_comparavel": 2.0,                   # |z| até aqui: comparável
    "z_duvidoso": 3.0,                     # |z| até aqui: duvidoso; acima: não usar
    "cv_piso": 0.02,                       # CV mínimo usado no z (CV de 3 a 5 regravações pode sair perto de zero por acaso)
    "metricas_veredito": ["tau_medio", "t50"],   # I0 só informativo por padrão
    "fracao_canal_decisao": 0.5,           # canal conta na decisão se esta fração dos vídeos da proporção (sem saturação) o usa
    "outras_amostras_limita_a_duvidoso": True,
    "razao_maxima_comparavel": 0.20,       # |razão - 1| acima disto nunca é "comparável", mesmo com irmãos muito dispersos
    "calibrar_pelos_irmaos": True,         # o limite de "comparável" sobe até o maior |z| que um irmão tem contra os demais (CV de 3-5 regravações subestima o espalhamento)
    "tolerancia_borda_quadros": 2,         # o vídeo cobre a janela se seus quadros ficam a até este nº de passos das bordas
    "min_n_por_amostra": 3,                # grupo proporção x amostra com menos vídeos de treino que isto entra no efeito_grupo
    "t50_pontos_i0": 5,
    "t50_suavizacao_pontos": 5,
    "workers": None,                       # None = modelagem.workers limitado ao número de CPUs
}
ROTULO = {0: "comparável", 1: "duvidoso", 2: "não usar"}
GRANDEZAS = ("tau_medio", "t50", "I0")
POOLS = ("mesma_amostra", "outras_amostras")


def parametros(cfg: dict) -> dict:
    p = dict(PADRAO)
    p.update({k: v for k, v in (cfg.get("saturados", {}) or {}).items()})
    pm, pp = parametros_modelagem(cfg), parametros_pos(cfg)
    p.update(ruido_k=pm["ruido_k"], janela_mediana_pontos=pm["janela_mediana_pontos"], duracao_minima_s=pm["duracao_minima_s"],
             inicio_ajuste_s=pp["inicio_ajuste_s"], saturacao_pct_max=pp["saturacao_pct_max"])
    if p["workers"] is None:
        p["workers"] = max(1, min(int(pm["workers"]), os.cpu_count() or 1))
    return p


# ----------------------------------------------------------------------------- medidas sem modelo e ajuste numa janela
def descritores_janela(t_rel: np.ndarray, y: np.ndarray, p: dict) -> dict:
    """I0 = mediana dos primeiros pontos; t50 = 1º instante (a partir do início da janela) em que a mediana móvel cai a I0/2."""
    t_rel, y = np.asarray(t_rel, float), np.asarray(y, float)
    out = {"I0": float("nan"), "t50": float("nan")}
    if len(y) == 0:
        return out
    I0 = float(np.median(y[:max(1, int(p["t50_pontos_i0"]))]))
    out["I0"] = I0
    if not I0 > 0:
        return out
    s = pd.Series(y).rolling(int(p["t50_suavizacao_pontos"]), center=True, min_periods=1).median().to_numpy()
    idx = np.where(s <= I0 / 2)[0]
    if len(idx):
        j = int(idx[0])
        if j == 0:
            out["t50"] = float(t_rel[0])
        else:
            f = (s[j - 1] - I0 / 2) / (s[j - 1] - s[j])
            out["t50"] = float(t_rel[j - 1] + f * (t_rel[j] - t_rel[j - 1]))
    return out


def ajustar_janela(t, y, p: dict) -> dict:
    """Ajuste do pipeline (C = 0) e descritores numa janela já recortada. t, y: só os pontos da janela."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    out = {"n_pontos": int(len(t)), "modelo": "", "tau_medio": float("nan"), "r2": float("nan"), "I0": float("nan"),
           "t50": float("nan"), "observacao": ""}
    if len(t) < 8:
        out["observacao"] = "poucos pontos na janela"
        return out
    tr = t - t[0]
    out.update(descritores_janela(tr, y, p))
    r = ajustar_serie(tr, y, MODELOS_C0)
    if r.get("modelo"):
        out.update(modelo=r["modelo"], tau_medio=float(r["tau_medio"]), r2=float(r["r2"]))
    out["observacao"] = str(r.get("observacao", "") or "")
    return out


def _tarefa(args):
    chave, t, y, p = args
    return chave, ajustar_janela(t, y, p)


def executar(tarefas: list[tuple], workers: int) -> dict:
    """Roda as tarefas (chave, t, y, p) sem repetir chaves. Devolve {chave: resultado}."""
    unicas: dict = {}
    for tr in tarefas:
        unicas.setdefault(tr[0], tr)
    lista = list(unicas.values())
    if workers > 1 and len(lista) > 20:
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_tarefa, lista, chunksize=10))
    else:
        res = [_tarefa(x) for x in lista]
    return dict(res)


# ----------------------------------------------------------------------------- janelas
def janela_saturado(g: pd.DataFrame, canal: str, sat_ate_s: float, p: dict) -> dict:
    """Janela a_parte do canal, com as regras de modelar.py. Devolve inicio_s, fim_s e motivo ('' se há janela)."""
    t = g["tempo_s"].to_numpy()
    y = g[f"corr_{canal}"].to_numpy()
    apos = t[t > sat_ate_s + 1e-9]
    if not len(apos):
        return {"inicio_s": float("nan"), "fim_s": float("nan"), "motivo": "sem quadro depois da saturação"}
    ini = float(apos[0])
    sel = t >= ini - 1e-9
    if sel.sum() < 8:
        return {"inicio_s": ini, "fim_s": float("nan"), "motivo": "poucos pontos depois do início"}
    fim, _ = fim_util(t[sel], y[sel], p["ruido_k"], p["janela_mediana_pontos"])
    if not fim == fim or fim - t[sel][0] < p["duracao_minima_s"]:
        return {"inicio_s": ini, "fim_s": fim, "motivo": f"sinal some no ruído antes de {p['duracao_minima_s']:g} s de janela"}
    return {"inicio_s": ini, "fim_s": float(fim), "motivo": ""}


def recorte(g: pd.DataFrame, canal: str, ini: float, fim: float, p: dict):
    """(t, y) da janela se o vídeo cobre a janela e não tem quadro saturado nela; senão None."""
    t = g["tempo_s"].to_numpy()
    sel = (t >= ini - 1e-9) & (t <= fim + 1e-9)
    if sel.sum() < 8:
        return None
    passo = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    tol = p.get("tolerancia_borda_quadros", PADRAO["tolerancia_borda_quadros"]) * passo
    if t[sel][0] > ini + tol + 1e-6 or t[sel][-1] < fim - tol - 1e-6:
        return None
    if (g["saturados_pct"].to_numpy()[sel] > p["saturacao_pct_max"]).any():
        return None
    return t[sel], g[f"corr_{canal}"].to_numpy()[sel]


def chave(video, canal, ini, fim):
    return (str(video), str(canal), round(float(ini), 4), round(float(fim), 4))


# ----------------------------------------------------------------------------- estatística
def classe_z(z: float, p: dict, z_irmaos: float = float("nan")) -> int | None:
    """0 comparável, 1 duvidoso, 2 não usar; None sem base. z_irmaos (opcional): maior |z| entre os próprios irmãos;
    com calibrar_pelos_irmaos os dois limites sobem até ele (o saturado não é cobrado mais que os irmãos entre si)."""
    if z is None or not z == z:
        return None
    lim_c, lim_d = p["z_comparavel"], p["z_duvidoso"]
    if p.get("calibrar_pelos_irmaos") and z_irmaos == z_irmaos:
        lim_c, lim_d = max(lim_c, z_irmaos), max(lim_d, z_irmaos)
    a = abs(z)
    return 0 if a <= lim_c else 1 if a <= lim_d else 2


def limitar_por_razao(k: int, razao: float, p: dict) -> int:
    """Diferença grande de nível não passa por 'comparável' só porque os irmãos são dispersos (ruído)."""
    if razao == razao and abs(razao - 1) > p["razao_maxima_comparavel"]:
        return max(k, 1)
    return k


def comparar_valor(x: float, ref, p: dict) -> dict:
    ref = np.asarray([v for v in ref if v == v], float)
    out = {"n": int(len(ref)), "mediana": float("nan"), "cv": float("nan"), "razao": float("nan"), "z": float("nan"),
           "z_irmaos_max": float("nan")}
    if len(ref) == 0:
        return out
    med = float(np.median(ref))
    out["mediana"] = med
    if len(ref) > 1 and med != 0:
        out["cv"] = float(np.std(ref, ddof=1) / abs(med))
    if x == x and med != 0 and out["cv"] == out["cv"]:
        out["razao"] = float(x / med)
        out["z"] = float((x / med - 1) / max(out["cv"], p["cv_piso"]))
    if len(ref) > p["min_irmaos"]:                       # calibração: o quanto cada irmão destoa dos demais
        zs = []
        for i in range(len(ref)):
            outros = np.delete(ref, i)
            m = float(np.median(outros))
            cv = float(np.std(outros, ddof=1) / abs(m)) if m != 0 else float("nan")
            if cv == cv:
                zs.append(abs((ref[i] / m - 1) / max(cv, p["cv_piso"])))
        if zs:
            out["z_irmaos_max"] = float(max(zs))
    return out


def canais_decisao(vpos: pd.DataFrame, proporcao: str, p: dict) -> list[str]:
    """Canais que a maioria dos vídeos da proporção sem saturação usa no ajuste (regra medida em videos_pos)."""
    v = vpos[(vpos["proporcao"] == proporcao) & (vpos["sat_ate_s"] < p["inicio_ajuste_s"] - 1e-9)]
    if v.empty:
        return []
    return [c for c in CANAIS if float(v[f"usar_no_ajuste_{c}"].astype(bool).mean()) >= p["fracao_canal_decisao"]]


# ----------------------------------------------------------------------------- análise principal
def analisar(series: pd.DataFrame, vpos: pd.DataFrame, ajustes: pd.DataFrame | None, cfg: dict) -> dict:
    p = parametros(cfg)
    por_video = {v: g.sort_values("tempo_s").reset_index(drop=True) for v, g in series.groupby("video")}
    sat = vpos[vpos["sat_ate_s"] >= p["inicio_ajuste_s"] - 1e-9].reset_index(drop=True)

    # 1. janelas dos saturados
    jan = []
    for _, v in sat.iterrows():
        for c in CANAIS:
            if not bool(v[f"sinal_presente_{c}"]) or v["video"] not in por_video:
                continue
            jan.append({"video": v["video"], "amostra": v["amostra"], "proporcao": v["proporcao"], "canal": c,
                        "sat_ate_s": float(v["sat_ate_s"]), **janela_saturado(por_video[v["video"]], c, float(v["sat_ate_s"]), p)})
    jan = pd.DataFrame(jan, columns=["video", "amostra", "proporcao", "canal", "sat_ate_s", "inicio_s", "fim_s", "motivo"])
    jan["tem_janela"] = jan["motivo"] == ""

    # 2. tarefas: o saturado e os irmãos (dois conjuntos), todos na janela do saturado
    tarefas, ligacoes = [], []
    for _, j in jan[jan["tem_janela"]].iterrows():
        g = por_video[j["video"]]
        r = recorte(g, j["canal"], j["inicio_s"], j["fim_s"], {**p, "saturacao_pct_max": np.inf})   # o próprio saturado: a janela já é pós-saturação
        if r is None:
            continue
        tarefas.append((chave(j["video"], j["canal"], j["inicio_s"], j["fim_s"]), r[0], r[1], p))
        cand = vpos[(vpos["proporcao"] == j["proporcao"]) & (vpos["video"] != j["video"]) & vpos[f"sinal_presente_{j['canal']}"].astype(bool)]
        for _, w in cand.iterrows():
            if w["video"] not in por_video:
                continue
            r2 = recorte(por_video[w["video"]], j["canal"], j["inicio_s"], j["fim_s"], p)
            if r2 is None:
                continue
            k = chave(w["video"], j["canal"], j["inicio_s"], j["fim_s"])
            tarefas.append((k, r2[0], r2[1], p))
            ligacoes.append((j["video"], j["canal"], "mesma_amostra" if w["amostra"] == j["amostra"] else "outras_amostras", w["video"], w["amostra"], k))
    res = executar(tarefas, p["workers"])

    # 3. tabelas de ajustes (saturado e irmãos) e conferência com ajustes.csv
    ap = None
    if ajustes is not None and len(ajustes):
        ap = ajustes[ajustes["caminho"] == "a_parte"].set_index(["video", "canal"])
    linhas_j = []
    for _, j in jan.iterrows():
        L = j.to_dict()
        r = res.get(chave(j["video"], j["canal"], j["inicio_s"], j["fim_s"])) if j["tem_janela"] else None
        L.update({f"sat_{k}": (r or {}).get(k, float("nan")) for k in ("modelo", "tau_medio", "r2", "I0", "t50")})
        if ap is not None and (j["video"], j["canal"]) in ap.index:
            a = ap.loc[(j["video"], j["canal"])]
            L["tau_ajustes_csv"] = float(a["tau_medio"]) if a["modelo"] else float("nan")
            L["fim_ajustes_csv"] = float(a["fim_s"])
            L["dif_rel_tau"] = abs(L["sat_tau_medio"] / L["tau_ajustes_csv"] - 1) if L["tau_ajustes_csv"] == L["tau_ajustes_csv"] and L["sat_tau_medio"] == L["sat_tau_medio"] else float("nan")
        linhas_j.append(L)
    jan_out = pd.DataFrame(linhas_j)

    irm = pd.DataFrame([{"video": v, "canal": c, "base": b, "irmao": w, "amostra_irmao": a, **{k: res[k_][k] for k in ("modelo", "tau_medio", "r2", "I0", "t50", "n_pontos")}}
                        for v, c, b, w, a, k_ in ligacoes if k_ in res],
                       columns=["video", "canal", "base", "irmao", "amostra_irmao", "modelo", "tau_medio", "r2", "I0", "t50", "n_pontos"])

    # 4. comparação por vídeo x canal x base x grandeza
    comp = []
    for _, j in jan[jan["tem_janela"]].iterrows():
        rs = res.get(chave(j["video"], j["canal"], j["inicio_s"], j["fim_s"]))
        if rs is None:
            continue
        for base in POOLS:
            d = irm[(irm["video"] == j["video"]) & (irm["canal"] == j["canal"]) & (irm["base"] == base)]
            for gr in GRANDEZAS:
                ref = d.loc[d["modelo"] != "", gr] if gr == "tau_medio" else d[gr]
                comp.append({"video": j["video"], "canal": j["canal"], "base": base, "grandeza": gr,
                             "valor_saturado": rs[gr] if (gr != "tau_medio" or rs["modelo"]) else float("nan"),
                             **comparar_valor(rs[gr] if (gr != "tau_medio" or rs["modelo"]) else float("nan"), ref, p)})
    comp = pd.DataFrame(comp, columns=["video", "canal", "base", "grandeza", "valor_saturado", "n", "mediana", "cv", "razao", "z", "z_irmaos_max"])

    # 5. veredito por canal e por vídeo
    vc = []
    for _, j in jan.iterrows():
        decide = j["canal"] in canais_decisao(vpos, j["proporcao"], p)
        L = {"video": j["video"], "amostra": j["amostra"], "proporcao": j["proporcao"], "canal": j["canal"], "decisao": decide,
             "inicio_s": j["inicio_s"], "fim_s": j["fim_s"], "classe": None, "base": "", "motivo": j["motivo"]}
        classes, bases = [], set()
        if j["tem_janela"]:
            for gr in GRANDEZAS:
                escolha = None
                for base in POOLS:
                    s = comp[(comp["video"] == j["video"]) & (comp["canal"] == j["canal"]) & (comp["base"] == base) & (comp["grandeza"] == gr)]
                    if len(s) and int(s["n"].iloc[0]) >= p["min_irmaos"] and classe_z(float(s["z"].iloc[0]), p) is not None:
                        k = classe_z(float(s["z"].iloc[0]), p, float(s["z_irmaos_max"].iloc[0]))
                        k = limitar_por_razao(k, float(s["razao"].iloc[0]), p)
                        if base == "outras_amostras" and p["outras_amostras_limita_a_duvidoso"]:
                            k = max(k, 1)
                        escolha = (k, base, s.iloc[0])
                        break
                if escolha is not None:
                    k, base, s = escolha
                    L[f"z_{gr}"], L[f"razao_{gr}"], L[f"classe_{gr}"], L[f"base_{gr}"] = float(s["z"]), float(s["razao"]), ROTULO[k], base
                    if gr in p["metricas_veredito"]:
                        classes.append((k, gr)); bases.add(base)
            if classes:
                pior = max(classes)
                L["classe"], L["base"] = pior[0], "+".join(sorted(bases))
                L["motivo"] = f"pior grandeza: {pior[1]}"
            else:
                L["motivo"] = "sem irmãos suficientes (mesma amostra ou outras) para as grandezas do veredito"
        vc.append(L)
    vc = pd.DataFrame(vc, columns=None if vc else ["video", "amostra", "proporcao", "canal", "decisao", "inicio_s", "fim_s", "classe", "base", "motivo"])

    vv = []
    for (video), d in (vc.groupby("video", sort=False) if len(vc) else []):
        dec = d[d["decisao"]]
        L = {"video": video, "amostra": d["amostra"].iloc[0], "proporcao": d["proporcao"].iloc[0],
             "inicio_a_parte_s": float(d["inicio_s"].min()), "canais_decisao": "".join(dec["canal"]), "veredito": "", "motivo": ""}
        if dec.empty:
            L.update(veredito=ROTULO[2], motivo="nenhum canal de decisão com sinal")
        elif dec["classe"].isna().any():
            c = dec[dec["classe"].isna()].iloc[0]
            L.update(veredito=ROTULO[2], motivo=f"canal {c['canal']}: {c['motivo']}")
        else:
            c = dec.loc[dec["classe"].astype(float).idxmax()]
            k = int(c["classe"])
            L.update(veredito=ROTULO[k], motivo=f"canal {c['canal']}, {c['motivo']}, base {c['base']}")
        vv.append(L)
    vv = pd.DataFrame(vv, columns=["video", "amostra", "proporcao", "inicio_a_parte_s", "canais_decisao", "veredito", "motivo"])
    vc["classe"] = vc["classe"].map(lambda k: ROTULO[int(k)] if k == k and k is not None else "")

    return {"p": p, "janelas": jan_out, "irmaos": irm, "comparacao": comp, "veredito_canal": vc, "veredito_video": vv,
            "por_video": por_video, "sat": sat}


# ----------------------------------------------------------------------------- efeito no grupo (viés de treino)
def treino_por_grupo(vpos: pd.DataFrame, vv: pd.DataFrame) -> pd.DataFrame:
    usa = vpos[[f"usar_no_ajuste_{c}" for c in CANAIS]].astype(bool).any(axis=1)
    base = vpos.assign(_usa=usa).groupby(["proporcao", "amostra"])["_usa"].sum().rename("n_treino_hoje").reset_index()
    n_sat = vpos[vpos["sat_ate_s"] >= 0].assign(_u=usa).query("not _u").groupby(["proporcao", "amostra"]).size().rename("n_fora_do_principal")
    out = base.merge(n_sat.reset_index(), how="left", on=["proporcao", "amostra"]).fillna({"n_fora_do_principal": 0})
    for rot, col in (("comparável", "n_a_parte_comparavel"), ("duvidoso", "n_a_parte_duvidoso"), ("não usar", "n_a_parte_nao_usar")):
        n = vv[vv["veredito"] == rot].groupby(["proporcao", "amostra"]).size().rename(col)
        out = out.merge(n.reset_index(), how="left", on=["proporcao", "amostra"])
        out[col] = out[col].fillna(0)
    for c in out.columns[2:]:
        out[c] = out[c].astype(int)
    out["n_treino_com_comparaveis"] = out["n_treino_hoje"] + out["n_a_parte_comparavel"]
    out["n_treino_com_comparaveis_e_duvidosos"] = out["n_treino_com_comparaveis"] + out["n_a_parte_duvidoso"]
    return out


def efeito_grupo(A: dict, vpos: pd.DataFrame, series: pd.DataFrame) -> pd.DataFrame:
    """Razão entre amostras da proporção alvo, numa janela comum, contra a mesma razão nas outras proporções."""
    p, jan, por_video = A["p"], A["janelas"], A["por_video"]
    usa = vpos[[f"usar_no_ajuste_{c}" for c in CANAIS]].astype(bool).any(axis=1)
    n_usa = vpos.assign(_u=usa).groupby(["proporcao", "amostra"])["_u"].sum()
    alvos = [(pr, am) for (pr, am), n in n_usa.items()
             if n < p["min_n_por_amostra"] and ((jan["proporcao"] == pr) & (jan["amostra"] == am)).any()]
    linhas = []
    for pr, am in alvos:
        dec = canais_decisao(vpos, pr, p)
        outras = sorted(set(vpos["amostra"]) - {am})
        for c in dec:
            j = jan[(jan["proporcao"] == pr) & (jan["amostra"] == am) & (jan["canal"] == c) & jan["tem_janela"]]
            if j.empty:
                linhas.append({"proporcao": pr, "amostra_alvo": am, "canal": c, "nota": "nenhum saturado do grupo tem janela neste canal"})
                continue
            ini, fim = float(j["inicio_s"].max()), float(j["fim_s"].min())
            if fim - ini < p["duracao_minima_s"]:
                linhas.append({"proporcao": pr, "amostra_alvo": am, "canal": c, "inicio_comum_s": ini, "fim_comum_s": fim,
                               "nota": "janela comum curta demais"})
                continue
            tarefas, meta = [], []
            for _, v in vpos[vpos[f"sinal_presente_{c}"].astype(bool)].iterrows():
                if v["video"] not in por_video:
                    continue
                r = recorte(por_video[v["video"]], c, ini, fim, p)
                if r is None:
                    continue
                k = chave(v["video"], c, ini, fim)
                tarefas.append((k, r[0], r[1], p))
                meta.append((v["proporcao"], v["amostra"], k))
            res = executar(tarefas, p["workers"])
            cel = pd.DataFrame([{"proporcao": a, "amostra": b, **{g: res[k][g] for g in GRANDEZAS}, "ok": bool(res[k]["modelo"])} for a, b, k in meta])
            if cel.empty:
                continue
            cel.loc[~cel["ok"], "tau_medio"] = np.nan
            agg = cel.groupby(["proporcao", "amostra"]).agg(n=("ok", "size"), **{g: (g, "median") for g in GRANDEZAS}).reset_index()
            for o in outras:
                a = agg[(agg["proporcao"] == pr) & (agg["amostra"] == am)]
                b = agg[(agg["proporcao"] == pr) & (agg["amostra"] == o)]
                if a.empty or b.empty or int(a["n"].iloc[0]) < p["min_irmaos"] or int(b["n"].iloc[0]) < p["min_irmaos"]:
                    continue
                for g in GRANDEZAS:
                    ra = float(a[g].iloc[0]) / float(b[g].iloc[0]) if float(b[g].iloc[0]) else float("nan")
                    ref = []
                    for pr2 in sorted(set(agg["proporcao"]) - {pr}):
                        x = agg[(agg["proporcao"] == pr2) & (agg["amostra"] == am)]
                        y = agg[(agg["proporcao"] == pr2) & (agg["amostra"] == o)]
                        if len(x) and len(y) and int(x["n"].iloc[0]) >= p["min_irmaos"] and int(y["n"].iloc[0]) >= p["min_irmaos"] \
                                and float(y[g].iloc[0]) > 0 and float(x[g].iloc[0]) > 0:
                            ref.append(float(x[g].iloc[0]) / float(y[g].iloc[0]))
                    L = {"proporcao": pr, "amostra_alvo": am, "amostra_ref": o, "canal": c, "grandeza": g, "inicio_comum_s": ini,
                         "fim_comum_s": fim, "n_alvo": int(a["n"].iloc[0]), "n_ref_amostra": int(b["n"].iloc[0]),
                         "valor_alvo": float(a[g].iloc[0]), "valor_ref_amostra": float(b[g].iloc[0]), "razao_alvo": ra,
                         "n_outras_proporcoes": len(ref)}
                    if len(ref) >= 3 and ra > 0:
                        lr = np.log(ref)
                        sd = float(np.std(lr, ddof=1))
                        L.update(razao_mediana_outras=float(np.exp(np.median(lr))), cv_log_outras=sd,
                                 z=float((np.log(ra) - np.median(lr)) / max(sd, p["cv_piso"])),
                                 percentil=float(np.mean(np.asarray(ref) <= ra) * 100))
                        L["classe"] = ROTULO[classe_z(L["z"], p)]
                    else:
                        L["nota"] = "poucas proporções de referência"
                    linhas.append(L)
    return pd.DataFrame(linhas)


# ----------------------------------------------------------------------------- relatório
def _tab(df: pd.DataFrame, cols: list[str], n: int = 15, fmt: dict | None = None) -> str:
    if df.empty:
        return "  (vazio)"
    d = df[cols].head(n).copy()
    for c, f in (fmt or {}).items():
        if c in d:
            d[c] = d[c].map(lambda x: format(x, f) if x == x and x is not None and not isinstance(x, str) else ("" if x != x else x))
    s = d.to_string(index=False)
    return s + (f"\n  ... (+{len(df) - n} linhas; veja o CSV)" if len(df) > n else "")


def _relatorio(A: dict, treino: pd.DataFrame, efeito: pd.DataFrame) -> str:
    p, vv, vc, jan = A["p"], A["veredito_video"], A["veredito_canal"], A["janelas"]
    L = ["Q4/B11 - Saturados (a_parte) x restante (só análise; não altera dataset.csv)", ""]
    L.append(f"Saturados: {len(vv)} vídeos (sat_ate_s >= {p['inicio_ajuste_s']:g} s). Irmãos reajustados na MESMA janela do saturado (C = 0).")
    L.append(f"Limiares (config): |z| <= {p['z_comparavel']:g} comparável; <= {p['z_duvidoso']:g} duvidoso; acima não usar"
             f"{' (limites sobem até o maior |z| entre os irmãos, quando há mais irmãos que o mínimo)' if p['calibrar_pelos_irmaos'] else ''}. "
             f"CV mínimo {p['cv_piso']:g}; irmãos mínimos {p['min_irmaos']}; grandezas do veredito: {', '.join(p['metricas_veredito'])}.")
    if "dif_rel_tau" in jan:
        d = jan["dif_rel_tau"].dropna()
        L.append(f"Conferência com ajustes.csv (a_parte): {len(d)} ajustes comparados; diferença relativa máxima do tau = {d.max():.2e}." if len(d) else
                 "Conferência com ajustes.csv: nada a comparar.")
    L += ["", "1) Veredito por vídeo", _tab(vv, ["video", "inicio_a_parte_s", "canais_decisao", "veredito", "motivo"], fmt={"inicio_a_parte_s": ".2f"}), ""]
    L += ["2) Detalhe por canal de decisão (razão saturado/mediana dos irmãos e z)",
          _tab(vc[vc["decisao"]].assign(video=lambda d: d["video"]),
               ["video", "canal", "classe", "base", "razao_tau_medio", "z_tau_medio", "razao_t50", "z_t50", "razao_I0", "z_I0"],
               n=27, fmt={c: ".2f" for c in ("razao_tau_medio", "z_tau_medio", "razao_t50", "z_t50", "razao_I0", "z_I0")} if "razao_I0" in vc else None), ""]
    comp = A["comparacao"]
    cal = comp[(comp["base"] == "mesma_amostra") & comp["z_irmaos_max"].notna()]
    if len(cal):
        L.append(f"Calibração: entre irmãos da mesma amostra, o maior |z| de um irmão contra os demais tem mediana {cal['z_irmaos_max'].median():.2f} "
                 f"e máximo {cal['z_irmaos_max'].max():.2f} (comparar com z_comparavel = {p['z_comparavel']:g}).")
        L.append("")
    L += ["3) Vídeos de treino por proporção x amostra (menores contagens)",
          _tab(treino.sort_values(["n_treino_hoje", "proporcao"]),
               ["proporcao", "amostra", "n_treino_hoje", "n_a_parte_comparavel", "n_a_parte_duvidoso", "n_a_parte_nao_usar",
                "n_treino_com_comparaveis", "n_treino_com_comparaveis_e_duvidosos"], n=8), ""]
    L.append("4) Efeito no grupo (razão entre amostras da proporção alvo contra a das outras proporções, mesma janela)")
    if efeito.empty or "razao_alvo" not in efeito:
        L.append("  Nenhum grupo com menos de min_n_por_amostra vídeos de treino, ou sem janela comum.")
    else:
        e = efeito.dropna(subset=["razao_alvo"])
        L.append(_tab(e, ["proporcao", "amostra_alvo", "amostra_ref", "canal", "grandeza", "inicio_comum_s", "fim_comum_s", "n_alvo", "n_ref_amostra",
                          "razao_alvo", "razao_mediana_outras", "cv_log_outras", "z", "percentil", "classe"],
                      fmt={c: ".2f" for c in ("inicio_comum_s", "fim_comum_s", "razao_alvo", "razao_mediana_outras", "cv_log_outras", "z", "percentil")}))
        for (pr, am), d in e.groupby(["proporcao", "amostra_alvo"]):
            m = d[d["grandeza"].isin(p["metricas_veredito"]) & d["classe"].notna()]
            if len(m):
                pior = m.loc[m["z"].abs().idxmax()]
                L.append(f"  {pr}/{am}: maior |z| entre as grandezas do veredito = {abs(pior['z']):.2f} ({pior['grandeza']}, canal {pior['canal']}, "
                         f"{pior['amostra_alvo']}/{pior['amostra_ref']}); classe {pior['classe']}.")
        for _, t in treino.merge(e[["proporcao", "amostra_alvo"]].drop_duplicates(), left_on=["proporcao", "amostra"],
                                 right_on=["proporcao", "amostra_alvo"]).iterrows():
            L.append(f"  {t['proporcao']}/{t['amostra']}: vídeos de treino {t['n_treino_hoje']} hoje; {t['n_treino_com_comparaveis']} com os comparáveis; "
                     f"{t['n_treino_com_comparaveis_e_duvidosos']} com comparáveis e duvidosos.")
    L += ["", "Observações do método: t50 e I0 são medidos a partir do início da janela (não de 2,5 s); a janela do saturado começa mais tarde que a do principal, "
          "então tau/t50 desses vídeos nunca são misturados com o principal sem a flag a_parte. Base 'outras_amostras' carrega a deriva de sessão e é limitada a duvidoso."]
    return "\n".join(L) + "\n"


def rodar(cfg: dict, pasta_pos=None, pasta_modelagem=None, saida=None) -> int:
    raiz = Path(cfg.get("pastas", {}).get("saida", "output"))
    pos = Path(pasta_pos or raiz / "posprocessamento")
    mod = Path(pasta_modelagem or raiz / "modelagem")
    for f in (pos / "series_pos.csv", pos / "videos_pos.csv"):
        if not f.exists():
            raise FileNotFoundError(f"{f} não existe (rode 'luminescencia consolidar' e 'luminescencia pos')")
    out = Path(saida or raiz / "saturados")
    out.mkdir(parents=True, exist_ok=True)
    series = pd.read_csv(pos / "series_pos.csv", encoding="utf-8-sig")
    vpos = pd.read_csv(pos / "videos_pos.csv", encoding="utf-8-sig")
    ajustes = pd.read_csv(mod / "ajustes.csv", encoding="utf-8-sig") if (mod / "ajustes.csv").exists() else None
    A = analisar(series, vpos, ajustes, cfg)
    treino = treino_por_grupo(vpos, A["veredito_video"])
    efeito = efeito_grupo(A, vpos, series)
    for nome, df in (("saturados_janelas", A["janelas"]), ("irmaos_ajustes", A["irmaos"]), ("comparacao", A["comparacao"]),
                     ("veredito_canal", A["veredito_canal"]), ("veredito_video", A["veredito_video"]),
                     ("treino_por_grupo", treino), ("efeito_grupo", efeito)):
        df.to_csv(out / f"{nome}.csv", index=False, encoding="utf-8-sig")
    texto = _relatorio(A, treino, efeito)
    (out / "relatorio_saturados.txt").write_text(texto, encoding="utf-8")
    print(texto)
    return 0
