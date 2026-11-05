"""Ponto de entrada: luminescencia <comando>."""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

from .config import carregar_config
from .tudo import NOMES_ETAPAS


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
        df, cfg.get("protocolo", {}).get("instante_queda_s", 0.5), aj.get("canais", {}),
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


def _cmd_consolidar(args, cfg):
    from .consolidacao import rodar

    return rodar(cfg, pasta_extracao=args.extracao, saida=args.saida, curadoria=args.curadoria)


def _cmd_pos(args, cfg):
    from .posprocessamento import rodar

    return rodar(cfg, pasta_consolidacao=args.consolidacao, saida=args.saida)


def _cmd_modelar(args, cfg):
    from .modelar import rodar

    return rodar(cfg, pasta_pos=args.pos, pasta_consolidacao=args.consolidacao, saida=args.saida)


def _cmd_tabela(args, cfg):
    from .tabela import rodar

    return rodar(cfg, pasta_modelagem=args.modelagem, pasta_pos=args.pos, pasta_consolidacao=args.consolidacao, saida=args.saida)


def _cmd_comparar_modelos(args, cfg):
    from .comparacao_modelos import rodar

    return rodar(cfg, pasta_pos=args.pos, saida=args.saida)


def _cmd_descritores(args, cfg):
    from .descritores import rodar

    return rodar(cfg, pasta_pos=args.pos, pasta_modelagem=args.modelagem, saida=args.saida)


def _cmd_offset(args, cfg):
    from .offset import rodar

    return rodar(cfg, pasta_extracao=args.extracao, pasta_consolidacao=args.consolidacao, saida=args.saida)


def _cmd_graficos(args, cfg):
    from .graficos import rodar

    return rodar(cfg, pasta_modelagem=args.modelagem, pasta_pos=args.pos, saida=args.saida)

def _cmd_deriva(args, cfg):
    from .deriva import rodar

    return rodar(cfg, pasta_tabela=args.tabela, saida=args.saida)


def _cmd_colinearidade(args, cfg):
    from .colinearidade import rodar

    return rodar(cfg, pasta_tabela=args.tabela, saida=args.saida, so_treino=args.so_treino)


def _cmd_janela_r(args, cfg):
    from .janela_r import rodar

    return rodar(cfg, pasta_pos=args.pos, pasta_modelagem=args.modelagem, pasta_tabela=args.tabela, saida=args.saida)


def _cmd_saturados(args, cfg):
    from .saturados import rodar

    return rodar(cfg, pasta_pos=args.pos, pasta_modelagem=args.modelagem, saida=args.saida)


def _cmd_ic_simulacao(args, cfg):
    from .ic_simulacao import rodar

    return rodar(cfg, pasta_deriva=args.deriva, saida=args.saida)


def _cmd_qualidade(args, cfg):
    from .qualidade import rodar

    return rodar(cfg, pasta_tabela=args.tabela, saida=args.saida)


def _cmd_avaliar_base(args, cfg):
    from .linhas_base import rodar

    return rodar(cfg, pasta_tabela=args.tabela, saida=args.saida, config_path=args.config)


def _cmd_preregistrar(args, cfg):
    from .preregistro import rodar

    return rodar(cfg, forcar=args.forcar, pasta_tabela=args.tabela, config_path=args.config)


def _cmd_tudo(args, cfg):
    from .tudo import rodar

    return rodar(
        cfg, de=args.de, ate=args.ate, com_offset=args.com_offset, parar_no_erro=args.parar_no_erro,
        config_path=args.config, conferir=args.conferir, gravar_referencia=args.gravar_referencia,
        so_conferir=args.so_conferir,
    )


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

    p = sub.add_parser("consolidar", parents=[comum], help="4. Consolidação: vídeos aceitos, preto corrigido por vídeo, formato longo")
    p.add_argument("--extracao", help="pasta da extração (padrão: output/extracao)")
    p.add_argument("--saida", help="pasta de saída (padrão: output/consolidacao)")
    p.add_argument("--curadoria", help="curadoria.csv a usar (padrão: curadoria.arquivo do config)")
    p.set_defaults(func=_cmd_consolidar)

    p = sub.add_parser("pos", parents=[comum], help="5. Pós-processamento: janela fixa do ajuste, saturação, canais usáveis")
    p.add_argument("--consolidacao", help="pasta da consolidação (padrão: output/consolidacao)")
    p.add_argument("--saida", help="pasta de saída (padrão: output/posprocessamento)")
    p.set_defaults(func=_cmd_pos)

    p = sub.add_parser("modelar", parents=[comum], help="6. Ajuste em lote (C = 0) com sensibilidades; precisa de 'consolidar' e 'pos'")
    p.add_argument("--pos", help="pasta do pós-processamento (padrão: output/posprocessamento)")
    p.add_argument("--consolidacao", help="pasta da consolidação (padrão: output/consolidacao)")
    p.add_argument("--saida", help="pasta de saída (padrão: output/modelagem)")
    p.set_defaults(func=_cmd_modelar)

    p = sub.add_parser("tabela", parents=[comum], help="6b. Tabela final por vídeo (C livre e C = 0 lado a lado) e dicionário de colunas")
    p.add_argument("--modelagem", help="pasta da modelagem (padrão: output/modelagem)")
    p.add_argument("--pos", help="pasta do pós-processamento (padrão: output/posprocessamento)")
    p.add_argument("--consolidacao", help="pasta da consolidação (padrão: output/consolidacao)")
    p.add_argument("--saida", help="pasta de saída (padrão: output/tabela)")
    p.set_defaults(func=_cmd_tabela)

    p = sub.add_parser("comparar_modelos", parents=[comum], help="6c. Compara exp simples, dupla, esticada e potência (só análise)")
    p.add_argument("--pos", help="pasta do pós-processamento (padrão: output/posprocessamento)")
    p.add_argument("--saida", help="pasta de saída (padrão: output/modelagem)")
    p.set_defaults(func=_cmd_comparar_modelos)

    p = sub.add_parser("descritores", parents=[comum], help="6d. Descritores sem modelo (t50, t10, razões em tempos fixos, área); tabela paralela ao dataset")
    p.add_argument("--pos", help="pasta do pós-processamento (padrão: output/posprocessamento)")
    p.add_argument("--modelagem", help="pasta da modelagem, só para comparar a repetibilidade com o tau (padrão: output/modelagem)")
    p.add_argument("--saida", help="pasta de saída (padrão: output/tabela)")
    p.set_defaults(func=_cmd_descritores)

    p = sub.add_parser("graficos", parents=[comum], help="7. Gráficos: semilog com os dois ajustes, tau cl x c0, resíduos, normalizada, log-log, ln I x t, 1/I x t")
    p.add_argument("--modelagem", help="pasta da modelagem (padrão: output/modelagem)")
    p.add_argument("--pos", help="pasta do pós-processamento (padrão: output/posprocessamento)")
    p.add_argument("--saida", help="pasta de saída (padrão: output/graficos)")
    p.set_defaults(func=_cmd_graficos)

    p = sub.add_parser("offset", parents=[comum], help="P5. Testa a suposição do preto por vídeo (offset cantos - ROI) nos canais sem sinal (só análise)")
    p.add_argument("--extracao", help="pasta da extração (padrão: output/extracao)")
    p.add_argument("--consolidacao", help="pasta da consolidação (padrão: output/consolidacao)")
    p.add_argument("--saida", help="pasta de saída (padrão: output/offset)")
    p.set_defaults(func=_cmd_offset)

    p = sub.add_parser("deriva", parents=[comum], help="Q4/B5. Deriva de sessão A/B/C por proporção, canal e grandeza (só análise; lê output/tabela)")
    p.add_argument("--tabela", help="pasta com dataset.csv e descritores.csv (padrão: output/tabela)")
    p.add_argument("--saida", help="pasta de saída (padrão: output/deriva)")
    p.set_defaults(func=_cmd_deriva)

    p = sub.add_parser("colinearidade", parents=[comum], help="Q4/B13. Correlação, VIF, agrupamento e dimensões efetivas das colunas de entrada (só análise)")
    p.add_argument("--tabela", help="pasta com dataset.csv, descritores.csv e os dicionários (padrão: output/tabela)")
    p.add_argument("--saida", help="pasta de saída (padrão: output/colinearidade; com --so-treino: output/colinearidade_AB)")
    p.add_argument("--so-treino", action="store_true", help="só as amostras de treino (divisao.treino, A+B) e sem os saturados; compara com output/colinearidade")
    p.set_defaults(func=_cmd_colinearidade)

    p = sub.add_parser("janela_r", parents=[comum], help="Q4/B12. Fim de janela específico do R: reajusta o R com outras regras de fim e compara tau, t50, CV e ajustabilidade (só análise)")
    p.add_argument("--pos", help="pasta com series_pos.csv e videos_pos.csv (padrão: output/posprocessamento)")
    p.add_argument("--modelagem", help="pasta com ajustes.csv, só para conferir a regra atual (padrão: output/modelagem)")
    p.add_argument("--tabela", help="pasta com dataset.csv, só para listar colunas afetadas (padrão: output/tabela)")
    p.add_argument("--saida", help="pasta de saída (padrão: output/janela_r)")
    p.set_defaults(func=_cmd_janela_r)

    p = sub.add_parser("saturados", parents=[comum], help="Q4/B11. Saturados (a_parte) x irmãos reajustados na mesma janela: tau, t50, I0, veredito por vídeo e efeito no grupo (só análise)")
    p.add_argument("--pos", help="pasta com series_pos.csv e videos_pos.csv (padrão: output/posprocessamento)")
    p.add_argument("--modelagem", help="pasta com ajustes.csv, só para conferir o caminho a_parte (padrão: output/modelagem)")
    p.add_argument("--saida", help="pasta de saída (padrão: output/saturados)")
    p.set_defaults(func=_cmd_saturados)
    p = sub.add_parser("ic_simulacao", parents=[comum], help="Bloco 2C/D6. Simula o IC 95%% do MAE em C com 1, 2 e 3 preparações e a chance de passar o critério D3 por sorte (só análise)")
    p.add_argument("--deriva", help="pasta com os CSVs de deriva (padrão: output/deriva)")
    p.add_argument("--saida", help="pasta de saída (padrão: output/ic_simulacao)")
    p.set_defaults(func=_cmd_ic_simulacao)

    p = sub.add_parser("qualidade", parents=[comum], help="Bloco 2B. Qualidade e viés dos diagnósticos do ajuste (r2, rmse, n_pontos, fim_s, ok): η² por proporção e amostra, vídeos de baixo peso candidatos (só análise; nenhum peso é aplicado)")
    p.add_argument("--tabela", help="pasta com dataset.csv e descritores.csv (padrão: output/tabela)")
    p.add_argument("--saida", help="pasta de saída (padrão: output/qualidade)")
    p.set_defaults(func=_cmd_qualidade)


    p = sub.add_parser("avaliar_base", parents=[comum], help="Bloco 1. Linhas de base SEM aprendizado (média do treino e regra fixa por I0/auc_10s) em A<->B e leave-one-proporção-out; nunca abre C")
    p.add_argument("--tabela", help="pasta com dataset.csv, descritores.csv e dicionários (padrão: output/tabela)")
    p.add_argument("--saida", help="pasta de saída (padrão: output/avaliacao)")
    p.set_defaults(func=_cmd_avaliar_base)

    p = sub.add_parser("preregistrar", parents=[comum], help="Bloco 1. Grava docs/preregistro.json/.md (hashes, variante primária, critério D3, validação); recusa sobrescrever sem --forcar")
    p.add_argument("--tabela", help="pasta com dataset.csv, descritores.csv e dicionários (padrão: output/tabela)")
    p.add_argument("--forcar", action="store_true", help="substitui um pré-registro existente (o antigo é guardado com data)")
    p.set_defaults(func=_cmd_preregistrar)

    p = sub.add_parser("tudo", parents=[comum], help="6e. Roda a cadeia consolidar -> ... -> graficos e carimba a versão nos relatórios")
    p.add_argument("--de", choices=NOMES_ETAPAS, help="primeira etapa a rodar (padrão: consolidar)")
    p.add_argument("--ate", choices=NOMES_ETAPAS, help="última etapa a rodar (padrão: graficos)")
    p.add_argument("--com-offset", action="store_true", help="inclui a etapa offset depois de descritores")
    p.add_argument("--parar-no-erro", action=argparse.BooleanOptionalAction, default=True,
                   help="interrompe na primeira etapa com código != 0 (padrão: sim; --no-parar-no-erro continua)")
    p.add_argument("--conferir", action="store_true", help="após rodar, confere o output com tests/referencia/referencia.json (código != 0 se divergir)")
    p.add_argument("--gravar-referencia", action="store_true", help="após rodar, grava tests/referencia/referencia.json a partir do output atual")
    p.add_argument("--so-conferir", action="store_true", help="só confere o output atual com a referência, sem rodar etapas")
    p.set_defaults(func=_cmd_tudo)

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