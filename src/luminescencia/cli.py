"""Ponto de entrada: luminescencia <comando>."""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

from .config import carregar_config


def _cmd_auditar(args, cfg):
    from .auditoria import auditar_pasta

    pasta = args.pasta or cfg.get("pastas", {}).get("videos", "data")
    auditar_pasta(
        pasta, cfg, saida=args.saida, cor=sys.stdout.isatty(),
        completo=args.completo, workers=args.workers, refazer=args.refazer,
    )


def _cmd_ajustar(args, cfg):
    import pandas as pd

    from .modelos import ajustar_canais

    aj = cfg.get("ajuste", {})
    df = pd.read_csv(args.csv)
    tabela = ajustar_canais(
        df, cfg.get("protocolo", {}).get("instante_queda_s", 20), aj.get("canais", {}),
        modelos=aj.get("modelos", ["exp_simples", "exp_dupla"]),
        r2_minimo=aj.get("r2_minimo"), delta_bic=aj.get("delta_bic"),
        fracao_minima_componente=aj.get("fracao_minima_componente"),
        separacao_minima_tau=aj.get("separacao_minima_tau"), minimo_pontos=aj.get("minimo_pontos"),
    )
    print(tabela.round(4).to_string(index=False))
    if args.saida:
        Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
        tabela.to_csv(args.saida, index=False, encoding="utf-8-sig")
        print(f"\nSalvo em {args.saida}")


def _cmd_roi(args, cfg):
    from .roi import rodar

    return rodar(cfg, pasta=args.pasta, saida=args.saida, refazer=args.refazer, workers=args.workers,
                 so_mosaico=args.so_mosaico, so_suspeitos=args.so_suspeitos)


def _cmd_extrair(args, cfg):
    from .extracao import rodar

    return rodar(cfg, pasta=args.pasta, saida=args.saida, roi_json=args.roi, refazer=args.refazer,
                 workers=args.workers, filtro=args.videos)


def _nao_implementado(nome):
    def f(args, cfg):
        print(f"O comando '{nome}' ainda não foi implementado.")
    return f


def main(argv=None) -> int:
    if os.name == "nt":
        os.system("")                                   # habilita cores ANSI no terminal do Windows
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    comum = argparse.ArgumentParser(add_help=False)
    comum.add_argument("--config", default="config.yaml", help="arquivo de configuração (padrão: config.yaml)")

    parser = argparse.ArgumentParser(prog="luminescencia", description="Análise de luminescência por vídeo")
    sub = parser.add_subparsers(dest="comando")

    p = sub.add_parser("auditar", parents=[comum], help="1. Auditoria dos vídeos")
    p.add_argument("pasta", nargs="?", help="pasta com os vídeos (padrão: pastas.videos do config)")
    p.add_argument("--saida", help="pasta do relatório (padrão: output/auditoria)")
    p.add_argument("--completo", action="store_true", help="lê todos os frames (mede jitter e frames repetidos; mais lento)")
    p.add_argument("--workers", type=int, help="vídeos lidos ao mesmo tempo (padrão: auditoria.workers do config)")
    p.add_argument("--refazer", action="store_true", help="ignora o cache e audita todos os vídeos de novo")
    p.set_defaults(func=_cmd_auditar)

    p = sub.add_parser("ajustar", parents=[comum], help="Ajuste de decaimento RGB em um CSV de extração")
    p.add_argument("csv", help="CSV de um vídeo (colunas tempo_s, fase e as colunas de canal do config)")
    p.add_argument("--saida", help="salvar a tabela de ajuste neste CSV")
    p.set_defaults(func=_cmd_ajustar)

    p = sub.add_parser("roi", parents=[comum], help="2. Detecção automática da amostra (ROI) e mosaico de conferência")
    p.add_argument("pasta", nargs="?", help="pasta com os vídeos (padrão: pastas.videos do config)")
    p.add_argument("--saida", help="pasta de saída (padrão: output/roi)")
    p.add_argument("--workers", type=int, help="vídeos lidos ao mesmo tempo (padrão: roi.workers do config)")
    p.add_argument("--refazer", action="store_true", help="ignora o que já foi detectado e lê todos os vídeos de novo")
    p.add_argument("--so-mosaico", action="store_true", help="refaz só o mosaico, a partir das miniaturas (não lê vídeos)")
    p.add_argument("--so-suspeitos", action="store_true", help="mosaico só com os vídeos suspeitos")
    p.set_defaults(func=_cmd_roi)

    p = sub.add_parser("extrair", parents=[comum], help="3. Extração: um CSV por vídeo (precisa do roi.json)")
    p.add_argument("pasta", nargs="?", help="pasta com os vídeos (padrão: pastas.videos do config)")
    p.add_argument("--saida", help="pasta de saída (padrão: output/extracao)")
    p.add_argument("--roi", help="roi.json a usar (padrão: output/roi/roi.json)")
    p.add_argument("--workers", type=int, help="vídeos lidos ao mesmo tempo (padrão: extracao.workers do config)")
    p.add_argument("--refazer", action="store_true", help="ignora o cache e extrai todos os vídeos de novo")
    p.add_argument("--videos", nargs="+", help="só os vídeos cujo caminho contém algum destes textos (ex.: B_15 RB_075)")
    p.set_defaults(func=_cmd_extrair)

    for nome, texto in [("consolidar", "4. Consolidação"), ("pos", "5. Pós-processamento")]:
        p = sub.add_parser(nome, parents=[comum], help=f"{texto} (ainda não implementado)")
        p.set_defaults(func=_nao_implementado(nome))

    args = parser.parse_args(argv)
    if not args.comando:
        parser.print_help()
        return 0
    try:
        codigo = args.func(args, carregar_config(args.config))
    except (FileNotFoundError, NotADirectoryError) as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 1
    return codigo if isinstance(codigo, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())