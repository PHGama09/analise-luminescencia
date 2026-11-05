"""Infraestrutura de avaliação (BLOCO 1, sem IA): alvo, dobras agrupadas, métricas e trava da amostra C.

Regras do protocolo (docs/protocolo_validacao.md) que este módulo faz cumprir:
  * alvo = fração f = prop/1,5 (soma 1). `fracoes_alvo` devolve o alvo SÓ para avaliar (colunas f_R, f_G, f_B); nunca é entrada.
  * unidade de validação = preparação (proporção x amostra), nunca o vídeo: as regravações da mesma preparação ficam sempre do mesmo lado.
  * treino = A+B; os vídeos com `excluido_saturacao` ficam fora do treino e da validação (D5).
  * C só é aberta no fim, uma vez, e só com `docs/preregistro.json` válido (`abrir_c`). Neste bloco nenhuma função calcula nada em C.

Todos os números saem dos CSVs; nada aqui guarda resultado escrito à mão. Os índices das dobras são POSIÇÕES (0..n-1) da tabela
passada, para casar com outras tabelas na mesma ordem (por exemplo, as entradas de `entradas.carregar_entradas`).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .entradas import eh_diagnostico

CANAIS = ("R", "G", "B")
ESQUEMAS = ("a_b", "leave_one_proporcao_out", "por_preparacao")
NOMES_ESQUEMAS = {"a_b": "A↔B", "leave_one_proporcao_out": "leave-one-proporção-out", "por_preparacao": "leave-one-preparação-out"}
META = ("video", "amostra", "grupo", "num_video", "proporcao")
PADROES = {
    "amostras_treino": ["A", "B"],
    "amostra_final": "C",
    "soma_proporcoes": 1.5,
    "tolerancia_soma": 1e-6,
    "bootstrap_n": 5000,
    "semente": 20261104,
    "nivel_ic": 0.95,
    "pasta_tabela": None,
}


class VazamentoDeDobras(ValueError):
    """Uma preparação (ou regravação) apareceu dos dois lados, ou entrou vídeo fora do universo permitido."""


# ----------------------------------------------------------------------------------------------------------------- config

def parametros(cfg: dict) -> dict:
    p = dict(PADROES)
    p.update((cfg or {}).get("avaliacao") or {})
    p["amostras_treino"] = list(p["amostras_treino"])
    if not p["amostras_treino"]:
        raise ValueError("avaliacao.amostras_treino não pode ser vazio")
    if p["amostra_final"] in p["amostras_treino"]:
        raise ValueError("avaliacao.amostra_final não pode estar em amostras_treino (C só no fim)")
    if not (0 < float(p["nivel_ic"]) < 1):
        raise ValueError("avaliacao.nivel_ic deve estar entre 0 e 1")
    if int(p["bootstrap_n"]) < 1:
        raise ValueError("avaliacao.bootstrap_n deve ser >= 1")
    return p


# ------------------------------------------------------------------------------------------------------------ leitura

def _pasta_tabela(cfg: dict, pasta_tabela=None) -> Path:
    p = parametros(cfg)
    return Path(pasta_tabela or p["pasta_tabela"] or Path((cfg.get("pastas") or {}).get("saida", "output")) / "tabela")


def _ler_dataset_bruto(cfg: dict, pasta_tabela=None) -> pd.DataFrame:
    """PRIVADA: lê dataset.csv inteiro (inclui C). Nada fora deste módulo deve usá-la; as funções públicas filtram ou travam."""
    caminho = _pasta_tabela(cfg, pasta_tabela) / "dataset.csv"
    if not caminho.is_file():
        raise FileNotFoundError(f"dataset.csv não encontrado em {caminho.parent.resolve()} (rode 'luminescencia tabela')")
    return pd.read_csv(caminho, encoding="utf-8-sig")


def chave_preparacao(df: pd.DataFrame) -> pd.Series:
    """Unidade independente = proporção x amostra (as regravações compartilham a chave)."""
    for c in ("proporcao", "amostra"):
        if c not in df:
            raise ValueError(f"coluna '{c}' ausente: não dá para definir a preparação")
    return (df["proporcao"].astype(str) + "|" + df["amostra"].astype(str)).rename("preparacao")


def mascara_ab(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Linhas do treino/validação interna: amostras de treino (A, B) sem os excluídos por saturação. Nunca C."""
    p = parametros(cfg)
    for c in ("amostra", "excluido_saturacao"):
        if c not in df:
            raise ValueError(f"coluna '{c}' ausente: não dá para excluir saturados/amostra C com segurança")
    sat = df["excluido_saturacao"].fillna(False).astype(bool)
    return df["amostra"].isin(p["amostras_treino"]) & ~sat


def carregar_ab(cfg: dict, pasta_tabela=None) -> tuple[pd.DataFrame, np.ndarray]:
    """Linhas de A+B válidas (sem saturados, sem C). Devolve (tabela com índice 0..n-1, posições dessas linhas em dataset.csv)."""
    bruto = _ler_dataset_bruto(cfg, pasta_tabela)
    pos = np.flatnonzero(mascara_ab(bruto, cfg).to_numpy())
    return bruto.iloc[pos].reset_index(drop=True), pos


# ------------------------------------------------------------------------------------------------------------- alvo

def fracoes_alvo(df: pd.DataFrame, soma: float = 1.5, tolerancia: float = 1e-6) -> pd.DataFrame:
    """f = prop/soma por vídeo (colunas f_R, f_G, f_B), conferindo que somam 1. SÓ para avaliar: nunca é coluna de entrada."""
    falta = [f"prop_{c}" for c in CANAIS if f"prop_{c}" not in df]
    if falta:
        raise ValueError(f"colunas ausentes para o alvo: {falta}")
    props = df[[f"prop_{c}" for c in CANAIS]].astype(float)
    if not np.isfinite(props.to_numpy()).all():
        raise ValueError("prop_R/G/B com valores ausentes ou infinitos")
    f = props / float(soma)
    f.columns = [f"f_{c}" for c in CANAIS]
    if (f.to_numpy() < 0).any():
        raise ValueError("fração negativa no alvo")
    desvio = float((f.sum(axis=1) - 1.0).abs().max()) if len(f) else 0.0
    if desvio > tolerancia:
        raise ValueError(f"as frações não somam 1 (desvio máximo {desvio:.3g} > tolerância {tolerancia:g}); soma esperada de prop = {soma}")
    return f


def validar_colunas_de_entrada(colunas) -> None:
    """Falha se alguma coluna de entrada for proibida pelo protocolo (alvo, rótulo, meta, seleção por saturação, diagnóstico)."""
    ruins = []
    for c in colunas:
        c = str(c)
        if (c.startswith(("prop_", "canal_ativo_", "usar_no_ajuste_", "f_")) or c in ("sat_ate_s", "excluido_saturacao", *META)
                or eh_diagnostico(c)):
            ruins.append(c)
    if ruins:
        raise ValueError(f"colunas proibidas como entrada (protocolo): {sorted(ruins)}")


# ------------------------------------------------------------------------------------------------------------ dobras

@dataclass(frozen=True)
class Dobra:
    esquema: str
    nome: str
    treino: np.ndarray        # posições na tabela passada
    validacao: np.ndarray


def _contexto(df: pd.DataFrame, cfg: dict):
    valido = mascara_ab(df, cfg).to_numpy()
    return (valido, chave_preparacao(df).to_numpy(), df["proporcao"].astype(str).to_numpy(), df["amostra"].astype(str).to_numpy())


def dobras_a_b(df: pd.DataFrame, cfg: dict) -> list[Dobra]:
    """Treina numa amostra e valida na outra (e o inverso)."""
    valido, _, _, amostra = _contexto(df, cfg)
    amostras = parametros(cfg)["amostras_treino"]
    if len(amostras) != 2:
        raise ValueError("A↔B precisa de exatamente 2 amostras em avaliacao.amostras_treino")
    saida = []
    for tr, va in (amostras, amostras[::-1]):
        t, v = np.flatnonzero(valido & (amostra == tr)), np.flatnonzero(valido & (amostra == va))
        if len(t) == 0 or len(v) == 0:
            raise ValueError(f"A↔B: sem vídeos válidos em {tr} ou {va}")
        saida.append(Dobra("a_b", f"{tr}→{va}", t, v))
    return saida


def dobras_leave_one_proporcao_out(df: pd.DataFrame, cfg: dict) -> list[Dobra]:
    """Estresse: tira uma proporção inteira (todas as preparações dela) e treina no resto de A+B."""
    valido, _, prop, _ = _contexto(df, cfg)
    saida = []
    for pr in sorted(set(prop[valido])):
        v, t = np.flatnonzero(valido & (prop == pr)), np.flatnonzero(valido & (prop != pr))
        if len(t):
            saida.append(Dobra("leave_one_proporcao_out", pr, t, v))
    return saida


def dobras_por_preparacao(df: pd.DataFrame, cfg: dict) -> list[Dobra]:
    """Leave-one-preparação-out: uma dobra por preparação (proporção x amostra)."""
    valido, prep, _, _ = _contexto(df, cfg)
    saida = []
    for pr in sorted(set(prep[valido])):
        v, t = np.flatnonzero(valido & (prep == pr)), np.flatnonzero(valido & (prep != pr))
        if len(t):
            saida.append(Dobra("por_preparacao", pr, t, v))
    return saida


_GERADORES = {"a_b": dobras_a_b, "leave_one_proporcao_out": dobras_leave_one_proporcao_out, "por_preparacao": dobras_por_preparacao}


def gerar_dobras(df: pd.DataFrame, esquema: str, cfg: dict) -> list[Dobra]:
    """Gera as dobras de um esquema e confere vazamento (levanta VazamentoDeDobras se houver)."""
    if esquema not in _GERADORES:
        raise ValueError(f"esquema desconhecido '{esquema}'; use um de {ESQUEMAS}")
    dobras = _GERADORES[esquema](df, cfg)
    verificar_sem_vazamento(df, dobras, cfg)
    return dobras


def verificar_sem_vazamento(df: pd.DataFrame, dobras: list[Dobra], cfg: dict, cobertura_unica: bool = True) -> None:
    """Confere, para todas as dobras: (1) só vídeos de A+B válidos (nada de C nem saturado); (2) nenhuma linha dos dois lados;
    (3) nenhuma preparação dos dois lados; (4) toda regravação de uma preparação do lado da sua preparação; (5) opcional: cada
    vídeo válido é validado exatamente uma vez no esquema."""
    valido, prep, _, _ = _contexto(df, cfg)
    n = len(df)
    vezes = np.zeros(n, dtype=int)
    for d in dobras:
        t, v = np.asarray(d.treino), np.asarray(d.validacao)
        rotulo = f"{d.esquema}/{d.nome}"
        if len(t) == 0 or len(v) == 0:
            raise VazamentoDeDobras(f"{rotulo}: treino ou validação vazio")
        for lado, idx in (("treino", t), ("validação", v)):
            if idx.min() < 0 or idx.max() >= n:
                raise VazamentoDeDobras(f"{rotulo}: índice fora da tabela no {lado}")
            if len(set(idx.tolist())) != len(idx):
                raise VazamentoDeDobras(f"{rotulo}: índice repetido no {lado}")
            if not valido[idx].all():
                raise VazamentoDeDobras(f"{rotulo}: o {lado} contém vídeo fora de A+B válido (amostra final ou saturado)")
        if set(t.tolist()) & set(v.tolist()):
            raise VazamentoDeDobras(f"{rotulo}: o mesmo vídeo está em treino e validação")
        comum = set(prep[t]) & set(prep[v])
        if comum:
            raise VazamentoDeDobras(f"{rotulo}: preparação em treino e validação ao mesmo tempo: {sorted(comum)[:5]}")
        for lado, idx in (("treino", t), ("validação", v)):
            esperado = set(np.flatnonzero(valido & np.isin(prep, prep[idx])).tolist())
            if esperado != set(idx.tolist()):
                raise VazamentoDeDobras(f"{rotulo}: regravação de uma preparação do {lado} ficou fora dele (cruza o lado)")
        vezes[v] += 1
    if cobertura_unica and dobras:
        if not (vezes[valido] == 1).all():
            raise VazamentoDeDobras("algum vídeo válido não é validado exatamente uma vez neste esquema")
        if (vezes[~valido] != 0).any():
            raise VazamentoDeDobras("vídeo fora do universo (C ou saturado) apareceu na validação")


def descrever_dobras(df: pd.DataFrame, dobras: list[Dobra]) -> pd.DataFrame:
    """Uma linha por dobra: nº de vídeos e de preparações (treino e validação) e proporções da validação sem nenhum vídeo no treino."""
    prep, prop = chave_preparacao(df).to_numpy(), df["proporcao"].astype(str).to_numpy()
    linhas = []
    for d in dobras:
        sem = sorted(set(prop[d.validacao]) - set(prop[d.treino]))
        linhas.append({"esquema": d.esquema, "dobra": d.nome,
                       "n_videos_treino": len(d.treino), "n_preparacoes_treino": len(set(prep[d.treino])),
                       "n_videos_validacao": len(d.validacao), "n_preparacoes_validacao": len(set(prep[d.validacao])),
                       "proporcoes_sem_treino": ", ".join(sem) if sem else "-"})
    return pd.DataFrame(linhas)


# ----------------------------------------------------------------------------------------------------------- métricas

def erros_por_preparacao(df: pd.DataFrame, f_verdade: pd.DataFrame, f_prevista) -> pd.DataFrame:
    """Erro absoluto por fração, médio sobre os vídeos de cada preparação (cada preparação conta uma vez). Índice = preparação.
    Colunas: proporcao, amostra, n_videos, erro_R, erro_G, erro_B, erro_medio."""
    pred = np.asarray(f_prevista, dtype=float)
    verdade = f_verdade.to_numpy(dtype=float)
    if pred.shape != verdade.shape or pred.shape[1] != 3:
        raise ValueError(f"previsão com forma {pred.shape}, esperado {verdade.shape}")
    if not np.isfinite(pred).all():
        raise ValueError("previsão com valores ausentes: trate-os antes de avaliar (não se avalia com NaN)")
    t = pd.DataFrame(np.abs(verdade - pred), columns=[f"erro_{c}" for c in CANAIS], index=df.index)
    t["preparacao"] = chave_preparacao(df).to_numpy()
    t["proporcao"] = df["proporcao"].astype(str).to_numpy()
    t["amostra"] = df["amostra"].astype(str).to_numpy()
    t["n_videos"] = 1
    tab = t.groupby("preparacao", sort=True).agg(
        proporcao=("proporcao", "first"), amostra=("amostra", "first"), n_videos=("n_videos", "sum"),
        erro_R=("erro_R", "mean"), erro_G=("erro_G", "mean"), erro_B=("erro_B", "mean"))
    tab["erro_medio"] = tab[[f"erro_{c}" for c in CANAIS]].mean(axis=1)
    return tab


def mae_por_fracao(tab: pd.DataFrame) -> dict:
    """MAE por fração e médio (cada preparação pesa igual)."""
    return {**{c: float(tab[f"erro_{c}"].mean()) for c in CANAIS}, "medio": float(tab["erro_medio"].mean())}


def mae_por_proporcao(tab: pd.DataFrame) -> pd.DataFrame:
    """MAE por proporção (média das preparações dela). Índice = proporção."""
    g = tab.groupby("proporcao", sort=True)
    saida = g[[f"erro_{c}" for c in CANAIS] + ["erro_medio"]].mean()
    saida["n_preparacoes"] = g.size()
    return saida


def pior_proporcao(tab: pd.DataFrame) -> tuple[str, float]:
    por = mae_por_proporcao(tab)["erro_medio"]
    return str(por.idxmax()), float(por.max())


def razao_ao_erro_da_media(tab_modelo: pd.DataFrame, tab_media: pd.DataFrame) -> float:
    """MAE médio do modelo / MAE médio da linha de base 'média do treino' (< 1 = melhor que a média)."""
    base = mae_por_fracao(tab_media)["medio"]
    return float(mae_por_fracao(tab_modelo)["medio"] / base) if base > 0 else float("nan")


def _indices_bootstrap(n: int, p: dict) -> np.ndarray:
    """Reamostra UNIDADES (preparações ou proporções) com reposição; semente fixa do config, independente da ordem das chamadas."""
    rng = np.random.default_rng(int(p["semente"]))
    return rng.integers(0, n, size=(int(p["bootstrap_n"]), n))


def _limites(amostras: np.ndarray, p: dict) -> tuple[float, float]:
    a = (1.0 - float(p["nivel_ic"])) / 2.0
    lo, hi = np.nanquantile(amostras, [a, 1.0 - a])
    return float(lo), float(hi)


def ic_bootstrap_mae(tab: pd.DataFrame, cfg: dict, coluna: str = "erro_medio") -> tuple[float, float]:
    """IC (percentil) do MAE por reamostragem de preparações."""
    v = tab[coluna].to_numpy(float)
    idx = _indices_bootstrap(len(v), parametros(cfg))
    return _limites(v[idx].mean(axis=1), parametros(cfg))


def ic_bootstrap_razao(tab_modelo: pd.DataFrame, tab_media: pd.DataFrame, cfg: dict) -> tuple[float, float]:
    """IC da razão MAE(modelo)/MAE(média) reamostrando as MESMAS preparações nas duas (pareado)."""
    if list(tab_modelo.index) != list(tab_media.index):
        raise ValueError("as duas tabelas precisam ter as mesmas preparações, na mesma ordem")
    a, b = tab_modelo["erro_medio"].to_numpy(float), tab_media["erro_medio"].to_numpy(float)
    idx = _indices_bootstrap(len(a), parametros(cfg))
    with np.errstate(divide="ignore", invalid="ignore"):
        return _limites(a[idx].mean(axis=1) / b[idx].mean(axis=1), parametros(cfg))


def ganho_pareado(tab_ref: pd.DataFrame, tab_novo: pd.DataFrame, cfg: dict, por: str = "preparacao") -> dict:
    """Ganho = erro(ref) - erro(novo), pareado por unidade (positivo = o novo erra menos), com IC por reamostragem das unidades.
    por='preparacao' (padrão) ou 'proporcao' (média das preparações de cada proporção; é o pareamento do critério D3(c) em C)."""
    if por not in ("preparacao", "proporcao"):
        raise ValueError("por deve ser 'preparacao' ou 'proporcao'")
    if por == "proporcao":
        tab_ref, tab_novo = mae_por_proporcao(tab_ref), mae_por_proporcao(tab_novo)
    if set(tab_ref.index) != set(tab_novo.index):
        raise ValueError("as duas tabelas precisam cobrir exatamente as mesmas unidades")
    d = (tab_ref["erro_medio"] - tab_novo["erro_medio"].reindex(tab_ref.index)).to_numpy(float)
    p = parametros(cfg)
    lo, hi = _limites(d[_indices_bootstrap(len(d), p)].mean(axis=1), p)
    return {"unidade": por, "n_unidades": int(len(d)), "ganho_medio": float(d.mean()), "ic_inf": lo, "ic_sup": hi,
            "exclui_zero": bool(lo > 0 or hi < 0)}


# ---------------------------------------------------------------------------------------------------- trava do C

def abrir_c(cfg: dict, pasta_tabela=None) -> pd.DataFrame:
    """ÚNICA porta para as linhas da amostra C. Só funciona com docs/preregistro.json válido e hashes iguais aos arquivos atuais
    (dataset, descritores, dicionários, seções do config); senão RuntimeError. Só devolve linhas; não calcula nada."""
    from .preregistro import verificar_preregistro

    verificar_preregistro(cfg, pasta_tabela=pasta_tabela)          # RuntimeError com mensagem clara se recusar
    p = parametros(cfg)
    bruto = _ler_dataset_bruto(cfg, pasta_tabela)
    sat = bruto["excluido_saturacao"].fillna(False).astype(bool) if "excluido_saturacao" in bruto else False
    return bruto[(bruto["amostra"] == p["amostra_final"]) & ~sat].reset_index(drop=True)
