"""Leitura do config.yaml compartilhado por todas as etapas."""
from __future__ import annotations

from pathlib import Path

import yaml


def carregar_config(caminho: str | Path = "config.yaml") -> dict:
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo de configuração não encontrado: {caminho.resolve()}\n"
            "Rode o comando na pasta do projeto ou use --config."
        )
    with caminho.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}