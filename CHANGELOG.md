# CHANGELOG

Versões a partir do histórico do Git e dos trechos de README. Detalhes de cada decisão em `docs/DECISOES.md`.

## [0.16.4] — 2026-11-04 (não commitada)
- Bloco 2C: `luminescencia ic_simulacao` (simulação do IC 95% do MAE em C com 1, 2 e 3 preparações) e `docs/plano_segunda_preparacao.md`.

## [0.16.3] — 2026-11-04 (não commitada)
- Bloco 2B: `luminescencia qualidade` (viés dos diagnósticos por proporção/amostra, só A+B, sem saturados).

## [0.16.2] — 2026-11-04 (não commitada)
- Bloco 2A: `colinearidade --so-treino` e `colinearidade.amostras` (B13 só com A+B), saída em `output/colinearidade_AB/`.

## [0.16.x] — 2026-11-04 (versão exata não identificada no trecho)
- E1: bloqueio das colunas de diagnóstico em `entradas.py` (`bloquear_diagnosticos`, padrão true no código).

## [0.16.1] — 2026-11-04 (não commitada)
- Bloco 1: `avaliacao.py` (alvo, dobras agrupadas, métricas, trava do C), `linhas_base.py` (`luminescencia avaliar_base`), `preregistro.py` (`luminescencia preregistrar`).

## [0.16.0] — 2026-11-04 (commit `9b72f3e`)
- Q4 fase 2: `janela_r` (B12), `saturados` (B11), `tests/test_integracao.py` e `docs/reproduzir.md` (C15).

## [0.13.0] — 2026-11-04 (commit `b3984f4`)
- Q4 fase 1: `deriva` (B5), `colinearidade` (B13), `docs/protocolo_validacao.md` (C17).

## [0.12.0] — 2026-11-04 (commit `4cd84f7`)
- Arrumação: módulos de fachada do Q4 removidos; versão revertida para a da P6; README atualizado. Referência de regressão (`tests/referencia/`) gravada nesta versão em 2026-11-03 14:40.

## Até 0.11.x — 2026-10-21 a 2026-11-03
- Estrutura inicial, auditoria, ROI, extração, consolidação, pós-processamento, modelagem, tabela, comparação de modelos, descritores, offset, gráficos e `tudo` (P1–P6, Q1–Q5).
