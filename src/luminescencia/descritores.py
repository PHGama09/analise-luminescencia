"""Programa 6d - Descritores que não dependem de modelo (P3, decidido em 03/11/2026: opção b, tabela paralela).

O tau muda com a janela e com o C (ver README). Aqui o decaimento é descrito SEM ajustar nenhuma equação, só pelo sinal
medido (corr_*), para ter números que não dependem da escolha de modelo. Nada disto altera o dataset.csv (106 colunas):
a tabela é PARALELA e se junta ao dataset pela coluna `video`.

Entrada: output/posprocessamento/ (series_pos.csv, videos_pos.csv). Opcional: output/modelagem/ajustes.csv (só para o
relatório comparar a repetibilidade dos descritores com a do tau).
Saída (output/tabela/):
  descritores.csv              uma linha por vídeo aceito (as mesmas 236 do dataset), formato largo
  dicionario_descritores.csv   cada coluna com papel (meta, feature, flag) e descrição
  relatorio_descritores.txt    contagens, valores típicos, repetibilidade entre regravações e sensibilidade ao início da janela

Quem tem descritor: o MESMO critério do ajuste principal (usar_no_ajuste_* = True, decidido pelo sinal medido, nunca por
prop_*), e o sinal precisa durar pelo menos modelagem.duracao_minima_s depois do início, pela mesma regra de fim no ruído
(modelar.fim_util). Os demais ficam vazios. Vídeo saturado depois de 2,5 s fica vazio, como no ajuste principal.

Definições (tempo SEMPRE em segundos desde o início da janela, posprocessamento.inicio_ajuste_s = 2,5 s do vídeo)
  I0        média dos 3 primeiros pontos da janela (níveis; mesma definição do gráfico 4)
  curva     mediana móvel de `suavizacao_pontos` pontos, só para achar cruzamentos e ler valores em tempos fixos
  t50, t10  primeiro instante em que a curva cai a 50% e a 10% de I0 (interpolação linear entre os dois pontos vizinhos).
            Fica vazio se não cruza dentro da janela ou se o nível-alvo (fração x I0) não passa de
            descritores.k_ruido_nivel x ruído do canal (aí o cruzamento seria ruído)
  r_<T>s    I(T)/I0 em tempo fixo T. Vazio se T passa do fim da janela daquele sinal (o sinal já sumiu no ruído)
  auc_<H>s  área sob I/I0 de 0 a H (segundos; é um "tempo equivalente": H x média de I/I0). Vazio se H passa do fim da janela
  razões entre canais: I0 e t50 de um canal dividido pelo de outro (R/G, G/B, R/B), só quando os dois canais têm descritor

CUIDADOS
  - Estes descritores também dependem do início da janela (I0 é lido nele) e do fim no ruído; o relatório mede quanto.
  - Canal sem o pó mas com sinal (vazamento de cor) também tem descritor; ele descreve o vazamento, não o pó daquele canal.
  - Nenhuma coluna usa prop_*. As colunas *_md_ok são medidas no vídeo (podem ser entrada); video/amostra/grupo/num_video/
    proporcao são meta e NÃO podem ser entrada da IA.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .modelar import fim_util, parametros as parametros_modelar
from .posprocessamento import CANAIS, parametros as parametros_pos

PREFIXO = "md"          # "modelo-free": colunas <canal>_md_<descritor>, como <canal>_cl_/<canal>_c0_ no dataset
META = ["video", "amostra", "grupo", "num_video", "proporcao"]
PARES = [("R", "G"), ("G", "B"), ("R", "B")]


def parametros(cfg: dict) -> dict:
    p = {"niveis_queda": [0.5, 0.1], "tempos_fixos_s": [2, 5, 10, 20, 40], "horizontes_area_s": [10, 30],
         "suavizacao_pontos": 5, "k_ruido_nivel": 3.0,
         "nao_confiaveis_R": ["t10", "r_20s", "r_40s"]}      # só aviso no dicionário (Q2, 03/11/2026): a coluna continua na tabela
    p.update(cfg.get("descritores", {}) or {})
    if not p["niveis_queda"] or not all(0 < float(x) < 1 for x in p["niveis_queda"]):
        raise ValueError("descritores.niveis_queda deve ter frações entre 0 e 1 (ex.: [0.5, 0.1])")
    return p


def _rot(x: float) -> str:
    """0.5 -> '50', 0.1 -> '10' (nome da coluna t50, t10)."""
    return f"{x * 100:g}".replace(".", "_")


def _numero(x: float) -> str:
    return f"{x:g}".replace(".", "_")


def descritores_serie(t, y, p: dict, k: float, janela: int, dur_min: float) -> dict | None:
    """Descritores de uma série já recortada a partir do início da janela (t em s do vídeo). None = sem descritor."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    if len(t) < 8:
        return None
    fim, ruido = fim_util(t, y, k, janela)
    if not fim == fim or fim - t[0] < dur_min:
        return None
    i0 = float(np.mean(y[:3]))
    if not i0 > 0:
        return None
    sel = t <= fim + 1e-9
    tt, yy = t[sel] - t[0], y[sel]
    n = max(1, int(p["suavizacao_pontos"]))
    curva = pd.Series(yy).rolling(n, center=True, min_periods=max(1, n // 2 + 1)).median().to_numpy()
    duracao = float(tt[-1])
    L = {"I0": i0, "ruido": ruido, "janela_s": duracao}

    for f in p["niveis_queda"]:
        nome = f"t{_rot(float(f))}"
        alvo = float(f) * i0
        L[nome] = np.nan
        if alvo <= p["k_ruido_nivel"] * ruido:
            continue
        abaixo = np.where(curva <= alvo)[0]
        if not len(abaixo):
            continue
        j = int(abaixo[0])
        if j == 0:
            L[nome] = float(tt[0])
        else:
            y0, y1 = curva[j - 1], curva[j]
            L[nome] = float(tt[j - 1] + (tt[j] - tt[j - 1]) * (y0 - alvo) / (y0 - y1)) if y0 != y1 else float(tt[j])

    for T in p["tempos_fixos_s"]:
        L[f"r_{_numero(T)}s"] = float(np.interp(T, tt, curva) / i0) if T <= duracao else np.nan

    for H in p["horizontes_area_s"]:
        nome = f"auc_{_numero(H)}s"
        if H > duracao:
            L[nome] = np.nan
            continue
        m = tt < H
        grade = np.append(tt[m], H)
        valores = np.append(yy[m], np.interp(H, tt, yy)) / i0
        try:
            from scipy.integrate import trapezoid
            L[nome] = float(trapezoid(valores, grade))
        except ImportError:
            # Se o scipy não estiver disponível, fazemos a integração trapezoidal manualmente por segurança
            L[nome] = float(np.sum((valores[:-1] + valores[1:]) * np.diff(grade) / 2))
    return L


def _colunas_descritor(p: dict) -> list[str]:
    cols = ["I0"] + [f"t{_rot(float(f))}" for f in p["niveis_queda"]]
    cols += [f"r_{_numero(T)}s" for T in p["tempos_fixos_s"]] + [f"auc_{_numero(H)}s" for H in p["horizontes_area_s"]]
    return cols


def calcular(series: pd.DataFrame, vpos: pd.DataFrame, cfg: dict, inicio: float | None = None) -> pd.DataFrame:
    """Tabela larga, uma linha por vídeo de vpos. `inicio` (s) muda o começo da janela (usado na sensibilidade)."""
    p, pm, pp = parametros(cfg), parametros_modelar(cfg), parametros_pos(cfg)
    inicio = pp["inicio_ajuste_s"] if inicio is None else float(inicio)
    por_video = {v: g for v, g in series.groupby("video")}
    cols = _colunas_descritor(p)
    linhas = []
    for _, v in vpos.iterrows():
        L = {k: v[k] for k in META}
        g = por_video.get(v["video"])
        achou = {}
        for c in CANAIS:
            r = None
            if g is not None and bool(v[f"usar_no_ajuste_{c}"]):
                d = g[g["tempo_s"] >= inicio - 1e-9]
                r = descritores_serie(d["tempo_s"].to_numpy(), d[f"corr_{c}"].to_numpy(), p,
                                      pm["ruido_k"], pm["janela_mediana_pontos"], pm["duracao_minima_s"])
            achou[c] = r
            L[f"{c}_{PREFIXO}_ok"] = r is not None
            for k in cols:
                L[f"{c}_{PREFIXO}_{k}"] = r.get(k, np.nan) if r else np.nan
        for a, b in PARES:                                       # razões entre canais
            for k in ("I0", "t50"):
                ra, rb = achou.get(a), achou.get(b)
                if (
                    ra is not None 
                    and rb is not None 
                    and isinstance(ra, dict) 
                    and isinstance(rb, dict)
                    and k in ra 
                    and k in rb 
                    and ra[k] == ra[k] 
                    and rb[k] == rb[k] 
                    and rb[k] != 0
                ):
                    L[f"razao_{k}_{a}_{b}"] = ra[k] / rb[k]
                else:
                    L[f"razao_{k}_{a}_{b}"] = np.nan
        linhas.append(L)
    return pd.DataFrame(linhas)


def dicionario(df: pd.DataFrame, p: dict, inicio: float) -> pd.DataFrame:
    jan = f"janela a partir de {inicio:g} s do vídeo"
    texto = {"I0": f"nível inicial da janela, média dos 3 primeiros pontos (níveis; {jan})"}
    for f in p["niveis_queda"]:
        texto[f"t{_rot(float(f))}"] = f"tempo para I/I0 cair a {float(f):.0%} (s desde o início da janela; vazio se não cruza ou o nível some no ruído)"
    for T in p["tempos_fixos_s"]:
        texto[f"r_{_numero(T)}s"] = f"I/I0 em {T:g} s depois do início da janela (vazio se o sinal já sumiu no ruído antes)"
    for H in p["horizontes_area_s"]:
        texto[f"auc_{_numero(H)}s"] = f"área sob I/I0 de 0 a {H:g} s (s; vazio se o sinal some no ruído antes)"
    linhas = []
    for col in df.columns:
        if col in META:
            papel, desc = "meta", "identifica o vídeo; não usar como entrada da IA"
        elif col.endswith(f"_{PREFIXO}_ok"):
            papel, desc = "flag", "o canal tem descritores (medido no vídeo, não usa prop_*): pode ser entrada"
        elif col.startswith("razao_"):
            partes = col.split("_")
            papel, desc = "feature", f"razão {partes[1]} do canal {partes[2]} pelo canal {partes[3]} (só quando os dois têm descritor)"
        else:
            c, _, k = col.split("_", 2)
            papel, desc = "feature", f"canal {c}, sem modelo: {texto.get(k, k)}"
            if c == "R" and k in set(p.get("nao_confiaveis_R") or []):
                desc += (" AVISO: no canal R não é confiável (o sinal do R chega ao ruído; CV entre regravações de 12% a 42% nos dados atuais, ver "
                         "relatorio_descritores.txt). A coluna foi mantida; decidir no treino se entra.")
        linhas.append({"coluna": col, "papel": papel, "descricao": desc})
    return pd.DataFrame(linhas)


def _cv(x: pd.Series) -> float:
    return float(x.std(ddof=1) / x.mean()) if len(x) > 1 and x.mean() else np.nan


def _cv_por_grupo(df: pd.DataFrame, col: str) -> float:
    """CV mediano entre as regravações (por proporção e amostra); só grupos com 2 ou mais valores."""
    x = df[["proporcao", "amostra", col]].dropna()
    return float(x.groupby(["proporcao", "amostra"])[col].agg(_cv).median()) if len(x) else np.nan


def _relatorio(df: pd.DataFrame, p: dict, inicio: float, sens: dict, aj: pd.DataFrame | None) -> str:
    L = ["DESCRITORES SEM MODELO (tabela paralela ao dataset.csv; junte pela coluna 'video')", "=" * 70,
         f"Vídeos: {len(df)}   colunas: {df.shape[1]}   janela: a partir de {inicio:g} s do vídeo (tempo dos descritores = s desde esse instante)", ""]
    L.append("Sinais com descritores (com ajuste principal = mesma regra de elegibilidade):")
    L.append("  " + "   ".join(f"{c}: {int(df[f'{c}_{PREFIXO}_ok'].sum())}/{len(df)}" for c in CANAIS))
    cols = _colunas_descritor(p)
    L += ["", "Valores por canal (mediana; entre parênteses: quantos sinais têm o valor / sinais com descritores):"]
    for c in CANAIS:
        ok = int(df[f"{c}_{PREFIXO}_ok"].sum())
        if not ok:
            continue
        partes = []
        for k in cols:
            x = df[f"{c}_{PREFIXO}_{k}"].dropna()
            if len(x):
                partes.append(f"{k} {x.median():.3g} ({len(x)}/{ok})")
            else:
                partes.append(f"{k} - (0/{ok})")
        L.append(f"  {c}: " + "; ".join(partes))
    L += ["", "Repetibilidade entre as 5 regravações (CV mediano por proporção e amostra; menor = mais estável). Para comparar: tau_medio do ajuste."]
    cab = f"    {'descritor':<22}" + "".join(f"{c:>9}" for c in CANAIS)
    L.append(cab)
    for k in cols:
        L.append(f"    {k:<22}" + "".join(f"{_cv_por_grupo(df, f'{c}_{PREFIXO}_{k}'):>9.1%}" for c in CANAIS))
    if aj is not None and len(aj):
        for cam, rot in (("principal", "tau_medio (C = 0)"), ("c_livre", "tau_medio (C livre)")):
            a = aj[(aj["caminho"] == cam) & (aj["modelo"].fillna("") != "")]
            if len(a):
                vals = []
                for c in CANAIS:
                    x = a[a["canal"] == c]
                    vals.append(f"{x.groupby(['proporcao', 'amostra'])['tau_medio'].agg(_cv).median():>9.1%}")
                L.append(f"    {rot:<22}" + "".join(vals))
    L += ["", "Razões entre canais (CV mediano entre regravações):"]
    for col in [c for c in df.columns if c.startswith("razao_")]:
        n = int(df[col].notna().sum())
        L.append(f"    {col:<18} n={n:<4} mediana {df[col].median():.3g}   CV {_cv_por_grupo(df, col):.1%}")
    L += ["", "Sensibilidade ao início da janela (diferença relativa mediana contra o início principal; mesmos vídeos e canais).",
          "  Aviso: I0 é lido no início da janela, então mover o início muda a referência; compare com o tau (~7% por 0,5 s) e com o CV acima."]
    for x, d in sens.items():
        L.append(f"  início {x:g} s (n comparável = {d.get('n', 0)} sinais):")
        for c in CANAIS:
            partes = [f"{k} {d[(c, k)]:.1%}" for k in ("t50", "t10", "r_10s", "auc_10s") if (c, k) in d]
            if partes:
                L.append(f"    {c}: " + "   ".join(partes))
    L += ["", "Como ler:",
          "  - Vazio (NaN) em t10, r_<T>s e auc_<H>s = o sinal (ou o nível-alvo) já está no ruído; é informação, não erro.",
          "  - Estes valores não são constantes do material: dependem da janela, do I0 e da regra de fim no ruído (declare-os sempre).",
          "  - Nenhuma coluna usa prop_*; meta (video, amostra, grupo, num_video, proporcao) não pode ser entrada da IA."]
    return "\n".join(L) + "\n"


def sensibilidade(series: pd.DataFrame, vpos: pd.DataFrame, cfg: dict, base: pd.DataFrame, p: dict) -> dict:
    """Recalcula com outros inícios (janelas_sensibilidade) e compara com o principal, nos sinais presentes nos dois."""
    pp = parametros_pos(cfg)
    saida = {}
    for x in pp["janelas_sensibilidade"]:
        if abs(x - pp["inicio_ajuste_s"]) < 1e-9:
            continue
        v = vpos[vpos[f"satura_apos_{f'{x:g}'.replace('.', '_')}s"] == False]       # noqa: E712  (só vídeos sem saturação depois de x)
        alt = calcular(series, v, cfg, inicio=x).set_index("video")
        b = base.set_index("video").loc[alt.index]
        d, n = {}, 0
        for c in CANAIS:
            ambos = alt[f"{c}_{PREFIXO}_ok"] & b[f"{c}_{PREFIXO}_ok"]
            n += int(ambos.sum())
            for k in ("t50", "t10", "r_10s", "auc_10s"):
                col = f"{c}_{PREFIXO}_{k}"
                if col in alt:
                    a_, b_ = alt.loc[ambos, col], b.loc[ambos, col]
                    ok = a_.notna() & b_.notna() & (b_ != 0)
                    if ok.sum() >= 5:
                        d[(c, k)] = float(((a_[ok] - b_[ok]) / b_[ok]).abs().median())
        d["n"] = n
        saida[x] = d
    return saida


def rodar(cfg: dict, pasta_pos=None, pasta_modelagem=None, saida=None) -> int:
    p, pp = parametros(cfg), parametros_pos(cfg)
    raiz = Path(cfg.get("pastas", {}).get("saida", "output"))
    pos = Path(pasta_pos or raiz / "posprocessamento")
    for f in (pos / "series_pos.csv", pos / "videos_pos.csv"):
        if not f.exists():
            raise FileNotFoundError(f"{f} não existe (rode 'luminescencia consolidar' e 'luminescencia pos')")
    out = Path(saida or raiz / "tabela")
    out.mkdir(parents=True, exist_ok=True)
    series = pd.read_csv(pos / "series_pos.csv", encoding="utf-8-sig")
    vpos = pd.read_csv(pos / "videos_pos.csv", encoding="utf-8-sig")
    df = calcular(series, vpos, cfg)
    caminho_aj = Path(pasta_modelagem or raiz / "modelagem") / "ajustes.csv"
    aj = pd.read_csv(caminho_aj, encoding="utf-8-sig") if caminho_aj.exists() else None
    sens = sensibilidade(series, vpos, cfg, df, p)
    df.round(4).to_csv(out / "descritores.csv", index=False, encoding="utf-8-sig")
    dicionario(df, p, pp["inicio_ajuste_s"]).to_csv(out / "dicionario_descritores.csv", index=False, encoding="utf-8-sig")
    texto = _relatorio(df, p, pp["inicio_ajuste_s"], sens, aj)
    (out / "relatorio_descritores.txt").write_text(texto, encoding="utf-8")
    print(texto)
    print(f"Arquivos em {out}: descritores.csv, dicionario_descritores.csv, relatorio_descritores.txt")
    return 0
