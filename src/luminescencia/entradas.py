"""Escolha das colunas de entrada da IA (D1, decidido em 02/11/2026).

D1: o tau dos canais SEM o pó (vazamento de cor da câmera) entra como entrada, mas deve ser possível desligar, para ter um
treino principal com e outro sem, e testar os dois na amostra C.

ATENÇÃO (limite que esta função respeita e que o treino também deve respeitar): na hora de usar a IA ninguém sabe qual canal
tem o pó, é isso que ela vai prever. Então "desligar o tau dos canais sem o pó" NÃO pode ser feito olhando prop_* ou
canal_ativo_* (isso entregaria o alvo pela ausência da coluna, o mesmo vazamento já corrigido na v0.8.1). O que dá para
desligar sem vazar é uma regra MEDIDA no vídeo. Aqui há duas chaves, ambas sem prop_*:

  entradas.canais_sem_po: true | false
      true  (padrão)  todo canal com ajuste entra (com o pó ou só com vazamento): é o treino "com".
      false           só entra o tau de canal cujo sinal medido passa de entradas.limiar_inicio_corr (níveis, por canal);
                      o resto fica NaN. É o treino "sem", mas em termos de SINAL FORTE, não de "ter o pó".
  entradas.limiar_inicio_corr: {R: .., G: .., B: ..}   nível mínimo de inicio_corr_<canal>

Cuidado com a interpretação: nos dados atuais o nível inicial do canal sem o pó se SOBREPÕE ao do canal com o pó (G: sem o pó
vai até ~212 níveis, com o pó começa em ~152; B: 203 contra 157; R: 83 contra 29). Nenhum limiar separa os dois grupos, então
o "sem" medido NÃO equivale a "sem vazamento". Use o resultado na amostra C para decidir, não a intenção do nome.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

CANAIS = ["R", "G", "B"]
CAMINHOS = ("cl", "c0")
PAPEIS_PROIBIDOS = {"meta", "alvo"}


def parametros(cfg: dict) -> dict:
    p = {"canais_sem_po": True, "limiar_inicio_corr": None, "caminhos": ["cl", "c0"]}
    p.update(cfg.get("entradas", {}) or {})
    if not set(p["caminhos"]) <= set(CAMINHOS) or not p["caminhos"]:
        raise ValueError(f"entradas.caminhos deve ser subconjunto de {CAMINHOS}")
    if not p["canais_sem_po"] and not p["limiar_inicio_corr"]:
        raise ValueError("entradas.canais_sem_po: false precisa de entradas.limiar_inicio_corr (regra MEDIDA; nunca prop_* ou canal_ativo_*)")
    return p


def colunas_de_entrada(dic: pd.DataFrame, p: dict) -> list[str]:
    """Colunas que podem ser entrada da IA: features dos caminhos pedidos e flags medidas. Nunca meta, alvo ou canal_ativo_*."""
    cols = []
    for _, r in dic.iterrows():
        c = r["coluna"]
        if r["papel"] in PAPEIS_PROIBIDOS or c.startswith("canal_ativo_") or c == "excluido_saturacao" or c == "sat_ate_s" or c.startswith("usar_no_ajuste_"):
            continue
        m = re.match(r"^[RGB]_(cl|c0)_", c)
        if m and m.group(1) not in p["caminhos"]:          # também as flags *_ok do caminho que ficou de fora
            continue
        cols.append(c)
    return cols


def aplicar_regra_de_sinal(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Com canais_sem_po = false, esvazia (NaN) o tau e demais colunas de ajuste do canal cujo sinal medido é fraco."""
    df = df.copy()
    if p["canais_sem_po"]:
        return df
    lim = p["limiar_inicio_corr"]
    for c in CANAIS:
        if f"inicio_corr_{c}" not in df or lim.get(c) is None:
            continue
        fraco = df[f"inicio_corr_{c}"] < float(lim[c])
        cols = [k for k in df.columns if k.startswith((f"{c}_cl_", f"{c}_c0_")) and not k.endswith("_ok")]
        df.loc[fraco, cols] = np.nan
        for pref in CAMINHOS:
            if f"{c}_{pref}_ok" in df:
                df.loc[fraco, f"{c}_{pref}_ok"] = False
    return df


def montar_entradas(df: pd.DataFrame, dic: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    p = parametros(cfg)
    cols = [c for c in colunas_de_entrada(dic, p) if c in df.columns]
    return aplicar_regra_de_sinal(df, p)[cols]
