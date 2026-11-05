"""Linhas de base SEM aprendizado e comando `avaliar_base` (BLOCO 1). Só A↔B e leave-one-proporção-out; nunca C.

Decisão do Pedro: nenhum modelo por enquanto (k-vizinhos adiado, nenhum ajuste de parâmetros). Linhas permitidas:
  (a) `media`: prevê a média das frações do treino da dobra (o único "parâmetro" é uma média);
  (b) `regra_fixa_<grandeza>`: fração do canal proporcional à sua grandeza medida (I0 e, como variante, auc_10s), normalizada para somar 1.
      Colunas vêm de `entradas.carregar_entradas` (caminho c0, descritores ligados, diagnósticos bloqueados); nunca `prop_*`.
      Ausências tratadas de forma explícita:
        * canal com `sinal_presente_<canal>` falso => fração 0 (a regra de SINAL MEDIDA diz que não há sinal; é o caso de R1.5);
        * canal com sinal presente mas grandeza ausente (NaN) ou negativa => o vídeo fica SEM previsão (não imputa; é contado e listado);
        * soma zero => sem previsão.
    Vídeos sem previsão saem da avaliação de TODAS as linhas (universo comum, para a comparação ser justa) e são contados no relatório.
A regra fixa é o piso do critério D3(c) enquanto não houver modelo.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .avaliacao import (CANAIS, NOMES_ESQUEMAS, carregar_ab, descrever_dobras, erros_por_preparacao, fracoes_alvo,
                        ganho_pareado, gerar_dobras, ic_bootstrap_mae, ic_bootstrap_razao, mae_por_fracao, mae_por_proporcao,
                        parametros as parametros_avaliacao, pior_proporcao, razao_ao_erro_da_media, validar_colunas_de_entrada)
from .descritores import PREFIXO
from .entradas import carregar_entradas

PADROES = {
    "esquemas": ["a_b", "leave_one_proporcao_out"],
    "regra_fixa": {"grandezas": ["I0", "auc_10s"]},        # a primeira é a primária (piso do critério D3(c))
    "entradas": {"caminhos": ["c0"], "descritores": True, "canais_sem_po": True, "bloquear_diagnosticos": True,
                 "descritores_sem_nao_confiaveis_R": False},
    "proporcoes_sensibilidade": ["G0.5_B1", "R1.5"],
    "saida": "avaliacao",
}
MEDIA = "media"


def parametros(cfg: dict) -> dict:
    p = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v) for k, v in PADROES.items()}
    dado = (cfg or {}).get("linhas_base") or {}
    for k, v in dado.items():
        if isinstance(v, dict) and isinstance(p.get(k), dict):
            p[k].update(v)
        else:
            p[k] = v
    if not p["regra_fixa"]["grandezas"]:
        raise ValueError("linhas_base.regra_fixa.grandezas não pode ser vazio")
    if p["entradas"].get("canais_sem_po") is not True:
        raise ValueError("linhas_base.entradas.canais_sem_po deve ser true: a regra fixa usa todos os canais medidos")
    return p


def nome_regra(grandeza: str) -> str:
    return f"regra_fixa_{grandeza}"


# ------------------------------------------------------------------------------------------------------ linhas de base

def prever_media(f_treino: pd.DataFrame, n: int) -> np.ndarray:
    """Média das frações do treino, repetida para os n vídeos de validação."""
    return np.tile(f_treino.to_numpy(float).mean(axis=0), (n, 1))


def prever_regra_fixa(entradas: pd.DataFrame, grandeza: str) -> tuple[pd.DataFrame, pd.Series]:
    """Regra fixa sem parâmetros: fração do canal = grandeza do canal / soma das grandezas. Devolve (previsões f_R..f_B com NaN onde
    não dá para prever, motivo por vídeo; '' = previsto). Só usa as colunas <canal>_md_<grandeza> e sinal_presente_<canal>."""
    precisa = [f"{c}_{PREFIXO}_{grandeza}" for c in CANAIS] + [f"sinal_presente_{c}" for c in CANAIS]
    falta = [c for c in precisa if c not in entradas.columns]
    if falta:
        raise ValueError(f"colunas ausentes nas entradas para a regra fixa '{grandeza}': {falta}")
    n = len(entradas)
    valores = np.zeros((n, 3))
    motivo = np.array([""] * n, dtype=object)

    def marca(mask, texto):
        livres = mask & (motivo == "")
        motivo[livres] = texto

    for j, c in enumerate(CANAIS):
        sinal_bruto = entradas[f"sinal_presente_{c}"]
        desconhecido = sinal_bruto.isna().to_numpy()
        tem_sinal = sinal_bruto.eq(True).to_numpy()          # NaN (desconhecido) não vira True; já marcado acima
        v = entradas[f"{c}_{PREFIXO}_{grandeza}"].to_numpy(float)
        marca(desconhecido, f"sinal_presente_{c} ausente")
        marca(tem_sinal & ~np.isfinite(v), f"{c}: sinal presente mas {grandeza} ausente (não imputado)")
        marca(tem_sinal & np.isfinite(v) & (v < 0), f"{c}: {grandeza} negativo")
        valores[:, j] = np.where(tem_sinal & np.isfinite(v), np.maximum(v, 0.0), 0.0)      # sem sinal medido => 0
    soma = valores.sum(axis=1)
    marca(soma <= 0, "soma das grandezas igual a zero")
    pred = np.full((n, 3), np.nan)
    ok = motivo == ""
    pred[ok] = valores[ok] / soma[ok, None]
    return pd.DataFrame(pred, columns=[f"f_{c}" for c in CANAIS], index=entradas.index), pd.Series(motivo, index=entradas.index, name="motivo")


# ------------------------------------------------------------------------------------------------------------ avaliação

def avaliar(df: pd.DataFrame, entradas: pd.DataFrame, cfg: dict) -> dict:
    """df e entradas: só A+B válidos, mesma ordem. Devolve tabelas e textos; nunca olha C."""
    p, pav = parametros(cfg), parametros_avaliacao(cfg)
    validar_colunas_de_entrada(entradas.columns)
    f = fracoes_alvo(df, pav["soma_proporcoes"], pav["tolerancia_soma"])
    grandezas = p["regra_fixa"]["grandezas"]
    regras = {nome_regra(g): prever_regra_fixa(entradas, g) for g in grandezas}
    ok_todas = np.ones(len(df), dtype=bool)
    for pred, _ in regras.values():
        ok_todas &= pred.notna().all(axis=1).to_numpy()
    chave = (df["proporcao"].astype(str) + "|" + df["amostra"].astype(str)).to_numpy()
    detalhe = pd.DataFrame([{"video": df["video"].iloc[i] if "video" in df else i, "preparacao": chave[i], "linha_base": n, "motivo": m.iloc[i]}
                            for n, (_, m) in regras.items() for i in np.flatnonzero((m != "").to_numpy())],
                           columns=["video", "preparacao", "linha_base", "motivo"])

    dobras_desc, erros, linhas, por_prop, ganhos = [], {}, [], [], []
    for esquema in p["esquemas"]:
        dobras = gerar_dobras(df, esquema, cfg)
        dobras_desc.append(descrever_dobras(df, dobras))
        previsoes = {MEDIA: np.full((len(df), 3), np.nan), **{n: pred.to_numpy() for n, (pred, _) in regras.items()}}
        for d in dobras:
            previsoes[MEDIA][d.validacao] = prever_media(f.iloc[d.treino], len(d.validacao))
        sel = np.flatnonzero(ok_todas)
        tabelas = {nome: erros_por_preparacao(df.iloc[sel], f.iloc[sel], pr[sel]) for nome, pr in previsoes.items()}
        erros[esquema] = tabelas
        for nome_sub, excl in [("todas", [])] + [(f"sem {x}", [x]) for x in p["proporcoes_sensibilidade"]]:
            sub = {n: t[~t["proporcao"].isin(excl)] for n, t in tabelas.items()}
            if len(sub[MEDIA]) == 0:
                continue
            for nome, t in sub.items():
                m = mae_por_fracao(t)
                pior, mae_pior = pior_proporcao(t)
                ic = ic_bootstrap_mae(t, cfg)
                razao = razao_ao_erro_da_media(t, sub[MEDIA])
                ric = ic_bootstrap_razao(t, sub[MEDIA], cfg) if nome != MEDIA else (1.0, 1.0)
                linhas.append({"esquema": esquema, "linha_base": nome, "subconjunto": nome_sub, "n_preparacoes": len(t),
                               "n_videos": int(t["n_videos"].sum()), **{f"mae_{c}": m[c] for c in CANAIS}, "mae_medio": m["medio"],
                               "mae_medio_ic_inf": ic[0], "mae_medio_ic_sup": ic[1], "razao_a_media": razao,
                               "razao_ic_inf": ric[0], "razao_ic_sup": ric[1], "pior_proporcao": pior, "mae_pior_proporcao": mae_pior})
            if nome_sub == "todas":
                for nome, t in sub.items():
                    pp = mae_por_proporcao(t).reset_index()
                    pp.insert(0, "linha_base", nome)
                    pp.insert(0, "esquema", esquema)
                    por_prop.append(pp)
            for nome in regras:
                g = ganho_pareado(sub[MEDIA], sub[nome], cfg, por="preparacao")
                ganhos.append({"esquema": esquema, "subconjunto": nome_sub, "referencia": MEDIA, "comparada": nome, **g})
    erros_prep = pd.concat([t.assign(esquema=e, linha_base=n).reset_index() for e, ts in erros.items() for n, t in ts.items()],
                           ignore_index=True)
    return {"dobras": pd.concat(dobras_desc, ignore_index=True), "tabela": pd.DataFrame(linhas),
            "por_proporcao": pd.concat(por_prop, ignore_index=True), "ganho_pareado": pd.DataFrame(ganhos),
            "erros_preparacao": erros_prep, "sem_previsao": detalhe, "n_videos_ab": len(df), "n_videos_avaliados": int(ok_todas.sum()),
            "n_preparacoes_ab": int(df.groupby(["proporcao", "amostra"]).ngroups), "grandezas": grandezas}


def _fmt(x, casas=4) -> str:
    return "nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{casas}f}"


def texto_relatorio(res: dict, cfg: dict) -> str:
    p = parametros(cfg)
    tab, gan = res["tabela"], res["ganho_pareado"]
    primaria = nome_regra(p["regra_fixa"]["grandezas"][0])
    L = ["Relatório de avaliação: linhas de base SEM aprendizado (BLOCO 1)", "=" * 64,
         f"versão do código: {__version__}   data: {datetime.now():%Y-%m-%d %H:%M}",
         "Só A+B (sem saturados); a amostra C não é aberta nem lida por este comando.", "",
         f"Universo: {res['n_videos_ab']} vídeos em {res['n_preparacoes_ab']} preparações (A+B sem saturados); "
         f"{res['n_videos_ab'] - res['n_videos_avaliados']} vídeo(s) sem previsão da regra fixa ficam fora de todas as linhas "
         f"(avaliados: {res['n_videos_avaliados']})."]
    for r in res["sem_previsao"].itertuples():
        L.append(f"  sem previsão: {r.video} ({r.preparacao}), {r.linha_base}: {r.motivo}")
    L += ["", "Frase calculada (MAE médio das 3 frações, cada preparação conta uma vez):"]
    for esquema in p["esquemas"]:
        t = tab[(tab.esquema == esquema) & (tab.subconjunto == "todas")].set_index("linha_base")
        if MEDIA not in t.index:
            continue
        partes = [f"a regra fixa ({g}) erra {_fmt(t.loc[nome_regra(g), 'mae_medio'])}" for g in res["grandezas"]]
        L.append(f"  {NOMES_ESQUEMAS[esquema]}: a média erra {_fmt(t.loc[MEDIA, 'mae_medio'])}; " + "; ".join(partes) + ".")
    L += ["", "As regras fixas não aprendem: o erro delas não depende do esquema, só o da média (ela usa o treino de cada dobra)."]
    L += ["Declaração: a regra fixa (primária = " + primaria + ") é o piso do critério D3(c) enquanto não houver modelo (k-vizinhos adiado)."]
    for esquema in p["esquemas"]:
        L += ["", "-" * 64, f"Esquema {NOMES_ESQUEMAS[esquema]}"]
        d = res["dobras"][res["dobras"].esquema == esquema]
        L += ["  dobras (vídeos e preparações; proporções sem treino):"]
        for r in d.itertuples():
            L.append(f"    {r.dobra:<16} treino {r.n_videos_treino:>3} v/{r.n_preparacoes_treino:>2} p | validação {r.n_videos_validacao:>3} v/"
                     f"{r.n_preparacoes_validacao:>2} p | sem treino: {r.proporcoes_sem_treino}")
        for sub in dict.fromkeys(tab[tab.esquema == esquema]["subconjunto"]):
            t = tab[(tab.esquema == esquema) & (tab.subconjunto == sub)]
            L += ["", f"  Subconjunto: {sub} ({int(t.n_preparacoes.iloc[0])} preparações)",
                  "    linha de base               MAE R   MAE G   MAE B   MAE med [IC95%]            razão à média [IC95%]    pior proporção"]
            for r in t.itertuples():
                L.append(f"    {r.linha_base:<26} {_fmt(r.mae_R):>7} {_fmt(r.mae_G):>7} {_fmt(r.mae_B):>7}   {_fmt(r.mae_medio)} [{_fmt(r.mae_medio_ic_inf)}; "
                         f"{_fmt(r.mae_medio_ic_sup)}]   {_fmt(r.razao_a_media, 3)} [{_fmt(r.razao_ic_inf, 3)}; {_fmt(r.razao_ic_sup, 3)}]   "
                         f"{r.pior_proporcao} ({_fmt(r.mae_pior_proporcao)})")
            g = gan[(gan.esquema == esquema) & (gan.subconjunto == sub)]
            for r in g.itertuples():
                veredito = "regra melhor que a média" if r.ic_inf > 0 else "média melhor que a regra" if r.ic_sup < 0 else "IC inclui 0 (sem diferença clara)"
                L.append(f"    ganho pareado por preparação, média - {r.comparada}: {_fmt(r.ganho_medio)} [{_fmt(r.ic_inf)}; {_fmt(r.ic_sup)}] "
                         f"(n={r.n_unidades}) -> {veredito}")
    L += ["", "-" * 64, "Pior proporção, por esquema (todas as preparações): ver pior_proporcao na tabela e mae_por_proporcao.csv.",
          "IC 95%: reamostragem de preparações (com reposição), semente fixa em avaliacao.semente.",
          "Convenção do ganho: erro(referência) - erro(comparada); positivo = a comparada erra menos.",
          "", "O que este relatório NÃO diz: nada sobre C, nada sobre modelos que aprendem, nada sobre generalização a outras preparações além de A↔B.",
          ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------------------- comando

def rodar(cfg: dict, pasta_tabela=None, saida=None, config_path="config.yaml") -> int:
    from .tudo import bloco_carimbo

    p = parametros(cfg)
    cfg_entradas = {**cfg, "entradas": {**(cfg.get("entradas") or {}), **p["entradas"]}}           # só em memória; o config não muda
    df, pos = carregar_ab(cfg, pasta_tabela)                                                      # A+B sem saturados; C nunca sai daqui
    todas = carregar_entradas(cfg_entradas, pasta_tabela)
    entradas = todas.iloc[pos].reset_index(drop=True)
    if len(todas) < (pos.max() + 1 if len(pos) else 0):
        raise ValueError("entradas e dataset.csv com tamanhos diferentes")
    res = avaliar(df, entradas, cfg)
    pasta = Path(saida or Path((cfg.get("pastas") or {}).get("saida", "output")) / p["saida"])
    pasta.mkdir(parents=True, exist_ok=True)
    for nome, chave in (("avaliacao_por_esquema.csv", "tabela"), ("mae_por_proporcao.csv", "por_proporcao"),
                        ("ganho_pareado.csv", "ganho_pareado"), ("dobras.csv", "dobras"), ("erros_por_preparacao.csv", "erros_preparacao"),
                        ("sem_previsao.csv", "sem_previsao")):
        res[chave].to_csv(pasta / nome, index=False, encoding="utf-8-sig", lineterminator="\n")
    texto = texto_relatorio(res, cfg)
    (pasta / "relatorio_avaliacao.txt").write_bytes((texto + "\n" + bloco_carimbo(config_path)).encode("utf-8"))
    print(res["dobras"].to_string(index=False))
    print()
    print("\n".join(l for l in texto.splitlines() if l.startswith("  ") and " a média erra " in l))
    print(f"Saída em {pasta}")
    return 0
