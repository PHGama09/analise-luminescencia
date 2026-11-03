"""Ajuste de curvas de decaimento da intensidade RGB.

Orientação do professor:
  1. Cada canal (R, G, B) é ajustado por uma equação de decaimento.
  2. O ideal é a exponencial simples (difícil de obter em dados reais).
     Se ela não servir, usa-se a exponencial dupla.

Como a escolha é feita
  Os modelos são testados na ordem de preferência. Fica o mais simples que
  (a) atinge R² mínimo e (b) não é claramente superado por um modelo mais
  complexo (ganho de BIC maior que delta_bic). Se nenhum atingir o R² mínimo,
  fica o de menor BIC e a observação avisa.

Para adicionar um modelo novo: escreva a função, crie um Modelo e registre
em MODELOS (veja exp_simples e exp_dupla como exemplo).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# ---------------------------------------------------------------- equações
def exp_simples(t, A, tau, C):
    """I(t) = A * exp(-t/tau) + C"""
    return A * np.exp(-t / tau) + C


def exp_dupla(t, A1, tau1, A2, tau2, C):
    """I(t) = A1 * exp(-t/tau1) + A2 * exp(-t/tau2) + C"""
    return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + C


# ---------------------------------------------------------------- registro
@dataclass
class Modelo:
    nome: str
    func: Callable
    params: tuple[str, ...]
    chutes: Callable      # (t, y) -> lista de chutes iniciais
    limites: Callable     # (t, y) -> (inferior, superior)
    validar: Callable | None = None   # (params_dict, cfg) -> mensagem de erro ou ""


def _estimativas(t, y):
    """Base, amplitude e tau aproximados, usados como chute inicial."""
    n_cauda = max(3, len(y) // 10)
    base = float(np.mean(y[-n_cauda:]))
    amp = float(y[0] - base)
    if amp <= 0:
        amp = max(float(np.ptp(y)), 1e-6)
    alvo = base + amp / np.e
    abaixo = np.where(y <= alvo)[0]
    duracao = float(t[-1] - t[0])
    tau = float(t[abaixo[0]] - t[0]) if len(abaixo) and t[abaixo[0]] > t[0] else duracao / 3
    return base, amp, max(tau, duracao / 200)


def _limites_comuns(t, y):
    passo = float(np.median(np.diff(t))) if len(t) > 1 else 1e-3
    duracao = float(t[-1] - t[0])
    amplitude_total = float(np.ptp(y)) or 1.0
    return passo, 100 * duracao, amplitude_total


def _chutes_simples(t, y):
    base, amp, tau = _estimativas(t, y)
    return [[amp, tau, base], [amp, tau * 3, base], [amp, tau / 3, base]]


def _limites_simples(t, y):
    passo, tau_max, span = _limites_comuns(t, y)
    return ([0, passo, float(np.min(y)) - span], [10 * span + 10 * float(np.max(y)), tau_max, float(np.max(y))])


def _chutes_dupla(t, y):
    base, amp, tau = _estimativas(t, y)
    return [
        [amp / 2, tau / 3, amp / 2, tau * 3, base],
        [amp / 2, tau / 10, amp / 2, tau * 2, base],
        [amp / 2, tau / 2, amp / 2, tau * 10, base],
    ]


def _limites_dupla(t, y):
    passo, tau_max, span = _limites_comuns(t, y)
    teto = 10 * span + 10 * float(np.max(y))
    return (
        [0, passo, 0, passo, float(np.min(y)) - span],
        [teto, tau_max, teto, tau_max, float(np.max(y))],
    )


def _validar_dupla(p, cfg):
    total = p["A1"] + p["A2"]
    if total <= 0:
        return "amplitudes nulas"
    menor_fracao = min(p["A1"], p["A2"]) / total
    if menor_fracao < cfg["fracao_minima_componente"]:
        return f"um componente tem só {menor_fracao:.1%} da amplitude (ajuste degenerado)"
    if p["tau2"] / p["tau1"] < cfg["separacao_minima_tau"]:
        return "os dois tempos de decaimento são quase iguais (ajuste degenerado)"
    return ""


MODELOS: dict[str, Modelo] = {
    "exp_simples": Modelo("exp_simples", exp_simples, ("A", "tau", "C"), _chutes_simples, _limites_simples),
    "exp_dupla": Modelo("exp_dupla", exp_dupla, ("A1", "tau1", "A2", "tau2", "C"), _chutes_dupla, _limites_dupla, _validar_dupla),
}

PADRAO = {
    "r2_minimo": 0.99,
    "delta_bic": 10.0,
    "fracao_minima_componente": 0.05,
    "separacao_minima_tau": 1.5,
    "minimo_pontos": 8,
}


# ---------------------------------------------------------------- ajuste
@dataclass
class Ajuste:
    modelo: str
    ok: bool
    motivo: str = ""
    params: dict = field(default_factory=dict)
    r2: float = float("nan")
    rmse: float = float("nan")
    bic: float = float("nan")


def ajustar_modelo(modelo: Modelo, t: np.ndarray, y: np.ndarray, cfg: dict) -> Ajuste:
    """Ajusta um modelo, testando vários chutes iniciais e ficando com o menor erro."""
    melhor = None
    for p0 in modelo.chutes(t, y):
        try:
            popt, _ = curve_fit(modelo.func, t, y, p0=p0, bounds=modelo.limites(t, y), max_nfev=20000)
        except (RuntimeError, ValueError):
            continue
        sse = float(np.sum((y - modelo.func(t, *popt)) ** 2))
        if melhor is None or sse < melhor[1]:
            melhor = (popt, sse)
    if melhor is None:
        return Ajuste(modelo.nome, False, "o ajuste não convergiu")

    popt, sse = melhor
    params = dict(zip(modelo.params, (float(v) for v in popt)))
    if modelo.nome == "exp_dupla" and params["tau1"] > params["tau2"]:   # componente rápido primeiro
        params["A1"], params["A2"] = params["A2"], params["A1"]
        params["tau1"], params["tau2"] = params["tau2"], params["tau1"]

    n, k = len(y), len(popt)
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - sse / sst if sst > 0 else float("nan")
    bic = n * np.log(max(sse / n, 1e-300)) + k * np.log(n)
    motivo = modelo.validar(params, cfg) if modelo.validar else ""
    return Ajuste(modelo.nome, motivo == "", motivo, params, r2, float(np.sqrt(sse / n)), float(bic))


def escolher(ajustes: list[Ajuste], cfg: dict) -> tuple[Ajuste | None, str]:
    """Aplica a regra: o mais simples que atende, a menos que um mais complexo o supere bem."""
    validos = [a for a in ajustes if a.ok]
    if not validos:
        return None, "nenhum modelo convergiu com parâmetros válidos"
    for i, a in enumerate(validos):
        if not a.r2 >= cfg["r2_minimo"]:
            continue
        if all(a.bic - b.bic <= cfg["delta_bic"] for b in validos[i + 1:]):
            return a, "" if i == 0 else f"{validos[0].modelo} não serviu; usado {a.modelo}"
    melhor = min(validos, key=lambda a: a.bic)
    return melhor, f"nenhum modelo atingiu R² ≥ {cfg['r2_minimo']}; usado o de menor BIC ({melhor.modelo})"


def ajustar_serie(t, y, modelos=("exp_simples", "exp_dupla"), **opcoes) -> dict:
    """Ajusta uma série (t em segundos desde o início do decaimento, y = intensidade)."""
    cfg = {**PADRAO, **{k: v for k, v in opcoes.items() if v is not None}}
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    boa = np.isfinite(t) & np.isfinite(y)
    t, y = t[boa], y[boa]
    if len(y) < cfg["minimo_pontos"]:
        return {"modelo": "", "observacao": f"poucos pontos ({len(y)})", "n_pontos": len(y)}

    ajustes = [ajustar_modelo(MODELOS[nome], t, y, cfg) for nome in modelos]
    escolhido, obs = escolher(ajustes, cfg)

    linha = {"modelo": escolhido.modelo if escolhido else "", "n_pontos": len(y)}
    if escolhido:
        linha.update(escolhido.params)
        linha.update(r2=escolhido.r2, rmse=escolhido.rmse, bic=escolhido.bic)
        p = escolhido.params
        if "A1" in p:
            linha["fracao_rapida"] = p["A1"] / (p["A1"] + p["A2"])
            linha["tau_medio"] = (p["A1"] * p["tau1"] + p["A2"] * p["tau2"]) / (p["A1"] + p["A2"])
        else:
            linha["tau_medio"] = p["tau"]
    for a in ajustes:                              # comparação entre modelos, sempre visível
        linha[f"r2_{a.modelo}"] = a.r2
        linha[f"bic_{a.modelo}"] = a.bic
        if not a.ok and a.motivo:
            obs = (obs + "; " if obs else "") + f"{a.modelo}: {a.motivo}"
    linha["observacao"] = obs
    return linha


def ajustar_canais(df: pd.DataFrame, instante_queda_s: float, canais: dict, modelos=("exp_simples", "exp_dupla"), **opcoes) -> pd.DataFrame:
    """Ajusta o decaimento de cada canal de um CSV de extração (um vídeo).

    Usa só a fase de decaimento. O tempo é contado a partir do primeiro frame do
    decaimento. Retorna uma linha por canal.
    """
    if "fase" in df.columns:
        decai = df[df["fase"].astype(str).str.lower().str.startswith("decai")]
    else:
        decai = df[df["tempo_s"] >= instante_queda_s]
    linhas = []
    if decai.empty:
        return pd.DataFrame([{"canal": c, "modelo": "", "observacao": "sem frames de decaimento"} for c in canais])
    t = decai["tempo_s"].to_numpy(dtype=float)
    t = t - t[0]
    for canal, coluna in canais.items():
        resultado = ajustar_serie(t, decai[coluna].to_numpy(dtype=float), modelos, **opcoes)
        linhas.append({"canal": canal, **resultado})
    return pd.DataFrame(linhas)