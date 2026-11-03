"""Programa 6b - Tabela final por vídeo (para a IA e para o orientador).

Entrada: output/modelagem/ajustes.csv, output/posprocessamento/videos_pos.csv e output/consolidacao/videos.csv.
Saída (output/tabela/):
  dataset.csv              uma linha por vídeo aceito, formato largo
  dicionario.csv           cada coluna com seu papel (meta, alvo, feature, flag) e uma descrição
  relatorio_tabela.txt     contagens, faltantes e avisos

Colunas de ajuste: <canal>_<caminho>_<parâmetro>, com caminho
  cl  C livre (exp_dupla, C estimado)        c0  C = 0 (exp_dupla_c0), janela fixa (posprocessamento.inicio_ajuste_s, padrão 2,5 s)
Os dois caminhos vêm lado a lado (decidido em 02/11/2026) para poder treinar com só um deles ou com ambos.

CUIDADOS (a tabela não decide isto por você)
  - Papéis: 'meta' (identificam o vídeo: NÃO usar como entrada da IA), 'alvo' (prop_R/G/B), 'feature', 'flag'.
  - canal_ativo_* sai do ALVO (prop_* > 0): NÃO usar como entrada. Já "ajuste ausente" (*_ok = False) e sinal_presente_* são
    decididos pelo sinal medido no vídeo (perto do ruído, sinal com duração mínima), sem olhar a proporção: podem ser entrada.
  - Canal sem o pó também é ajustado quando tem sinal (vazamento de cor da câmera); o tau dele descreve esse vazamento,
    não o pó daquele canal, e é mais variável (sinais curtos, sobretudo no R).
  - Vídeo que satura depois do início da janela (posprocessamento.inicio_ajuste_s, padrão 2,5 s) não tem ajuste nos caminhos principais (colunas vazias; excluido_saturacao = True).
    O ajuste à parte (janela própria) fica só em ajustes.csv, não entra aqui.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .posprocessamento import parametros as parametros_pos

CANAIS = ["R", "G", "B"]
CAMINHOS = {"cl": "c_livre", "c0": "principal"}
PARAMS = ["modelo", "A1", "tau1", "A2", "tau2", "C", "fracao_rapida", "tau_medio", "r2", "rmse", "inicio_s", "fim_s", "n_pontos"]
META = ["video", "amostra", "grupo", "num_video", "proporcao"]
ALVO = ["prop_R", "prop_G", "prop_B"]
DESCRICAO = {
    "modelo": "modelo escolhido", "A1": "amplitude do componente rápido", "tau1": "tau rápido (s)", "A2": "amplitude do componente lento",
    "tau2": "tau lento (s)", "C": "patamar (só no caminho cl; no c0 é 0 por definição)", "fracao_rapida": "A1/(A1+A2)",
    "tau_medio": "tau médio ponderado pelas amplitudes (s); não inclui o patamar", "r2": "R² do ajuste", "rmse": "erro quadrático médio do ajuste (níveis)",
    "inicio_s": "início da janela (s)", "fim_s": "fim da janela, onde o sinal some no ruído (s)", "n_pontos": "pontos usados no ajuste",
}


def montar(aj: pd.DataFrame, vpos: pd.DataFrame, vcons: pd.DataFrame, inicio_ajuste_s: float = 2.5) -> pd.DataFrame:
    """inicio_ajuste_s: início da janela (posprocessamento.inicio_ajuste_s); vídeo que satura a partir dele é excluido_saturacao."""
    base = vpos[META + ALVO + ["sat_ate_s"] + [f"canal_ativo_{c}" for c in CANAIS] + [f"sinal_presente_{c}" for c in CANAIS]
                 + [f"usar_no_ajuste_{c}" for c in CANAIS]].copy()
    base["excluido_saturacao"] = base[[f"usar_no_ajuste_{c}" for c in CANAIS]].eq(False).any(axis=1) & (base["sat_ate_s"] >= inicio_ajuste_s - 1e-9)
    extra = vcons.set_index("video")
    for c in CANAIS:
        base[f"inicio_corr_{c}"] = base["video"].map(extra[f"inicio_corr_{c}"])
        base[f"final_corr_{c}"] = base["video"].map(extra[f"final_corr_{c}"])
    for pref, cam in CAMINHOS.items():
        d = aj[(aj["caminho"] == cam) & (aj["modelo"] != "")]
        for c in CANAIS:
            x = d[d["canal"] == c].set_index("video")
            for k in PARAMS:
                if k in x.columns and not (k == "C" and pref == "c0"):
                    base[f"{c}_{pref}_{k}"] = base["video"].map(x[k])
            base[f"{c}_{pref}_ok"] = base["video"].isin(x.index)
    return base


def dicionario(df: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for col in df.columns:
        if col in META:
            papel, desc = "meta", "identifica o vídeo; não usar como entrada da IA"
        elif col in ALVO:
            papel, desc = "alvo", "proporção do pó na mistura (rótulo)"
        elif col.startswith("canal_ativo_") or col.startswith("usar_no_ajuste_") or col.startswith("sinal_presente_") or col == "excluido_saturacao" or col == "sat_ate_s" or col.endswith("_ok"):
            papel = "flag"
            if col.startswith("canal_ativo_"):
                desc = "vaza o alvo se usada como entrada (vem de prop_*)"
            elif col.startswith("sinal_presente_") or col.endswith("_ok"):
                desc = "medido no vídeo (não usa prop_*): pode ser entrada"
            else:
                desc = "controle de qualidade/seleção"
        elif col.startswith(("inicio_corr_", "final_corr_")):
            papel, desc = "feature", "nível do sinal corrigido do preto, no início/fim do vídeo"
        else:
            c, pref, k = col.split("_", 2)
            papel, desc = "feature", f"canal {c}, caminho {'C livre' if pref == 'cl' else 'C = 0'}: {DESCRICAO.get(k, k)}"
        linhas.append({"coluna": col, "papel": papel, "descricao": desc})
    return pd.DataFrame(linhas)


def _relatorio(df: pd.DataFrame, inicio_ajuste_s: float = 2.5) -> str:
    L = ["TABELA FINAL", "=" * 60, f"Vídeos: {len(df)}   colunas: {df.shape[1]}", "", "Vídeos por amostra e grupo:"]
    L += [f"  {a} ({g}): {n}" for (a, g), n in df.groupby(["amostra", "grupo"]).size().items()]
    L += ["", "Ajustes presentes (vídeos com ajuste / total de vídeos), por caminho e canal; entre parênteses: com o pó / sem o pó (só vazamento de cor):"]
    for pref in CAMINHOS:
        partes = []
        for c in CANAIS:
            ok, at = df[f"{c}_{pref}_ok"], df[f"canal_ativo_{c}"]
            partes.append(f"{c}_{pref}: {int(ok.sum())}/{len(df)} ({int(ok[at].sum())}/{int(at.sum())} / {int(ok[~at].sum())}/{int((~at).sum())})")
        L.append("  " + "   ".join(partes))
    n = int(df["excluido_saturacao"].sum())
    L += ["", f"Vídeos sem ajuste principal por saturação depois de {inicio_ajuste_s:g} s: {n}"]
    if n:
        L.append("  " + ", ".join(df.loc[df["excluido_saturacao"], "video"]))
    L += ["", "Entradas da IA: não usar colunas meta nem canal_ativo_* (vazam o alvo). *_ok e sinal_presente_* são medidos no vídeo.",
          "Canal sem o pó com sinal é ajustado; o tau dele descreve o vazamento de cor, não o pó daquele canal."]
    return "\n".join(L) + "\n"



def rodar(cfg: dict, pasta_modelagem=None, pasta_pos=None, pasta_consolidacao=None, saida=None) -> int:
    inicio = parametros_pos(cfg)["inicio_ajuste_s"]
    raiz = Path(cfg.get("pastas", {}).get("saida", "output"))
    mod, pos, cons = (Path(pasta_modelagem or raiz / "modelagem"), Path(pasta_pos or raiz / "posprocessamento"),
                      Path(pasta_consolidacao or raiz / "consolidacao"))
    for f in (mod / "ajustes.csv", pos / "videos_pos.csv", cons / "videos.csv"):
        if not f.exists():
            raise FileNotFoundError(f"{f} não existe (rode 'consolidar', 'pos' e 'modelar')")
    out = Path(saida or raiz / "tabela")
    out.mkdir(parents=True, exist_ok=True)
    aj = pd.read_csv(mod / "ajustes.csv", encoding="utf-8-sig")
    aj["modelo"] = aj["modelo"].fillna("")
    df = montar(aj, pd.read_csv(pos / "videos_pos.csv", encoding="utf-8-sig"), pd.read_csv(cons / "videos.csv", encoding="utf-8-sig"), inicio)
    df.to_csv(out / "dataset.csv", index=False, encoding="utf-8-sig")
    dicionario(df).to_csv(out / "dicionario.csv", index=False, encoding="utf-8-sig")
    texto = _relatorio(df, inicio)
    (out / "relatorio_tabela.txt").write_text(texto, encoding="utf-8")
    print(texto)
    print(f"Arquivos em {out}: dataset.csv, dicionario.csv, relatorio_tabela.txt")
    return 0
