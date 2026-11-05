# CHANGELOG

Versões a partir do histórico do Git e dos trechos de README. Detalhes de cada decisão em `docs/DECISOES.md`.

## [docs] — 2026-11-04 (verificação completa da v0.19.0)
- `scripts/anomalias_amostras.py`: seções 8 a 14 (o que sobra depois da correção da luz, I0 por sessão, ausentes e modelos por sessão, duplicatas, assinatura de decaimento, ordem das regravações depois da correção).
- README: itens 9 a 15 de "Anomalias entre amostras" e a pendência 3 (arredondamento do canto) no topo; guia e DECISOES atualizados.
- Proposta (não aplicada): `propostas/01_canto_sem_arredondamento.diff`.

## [0.19.0] — 2026-11-04
- `entradas.corrigir_luz: true` no `config.yaml` (decisão do Pedro; aguarda o orientador). Padrão do código continua `false`.
- `luz.separacao_cruzada`: separação entre composições sem e com a correção, cruzada A↔B; `output/luz/separacao.csv` e seção no relatório.
- Referência de regressão regravada (piso + luz).
- 2 testes novos em `test_luz.py`.

## [0.18.0] — 2026-11-04
- Etapa nova `luminescencia luz` (`luz.py`): correção da luz de fundo em colunas paralelas (`output/tabela/luz.csv`), com inclinações calibradas só em A+B, dentro das preparações e sem saturados; validação cruzada A↔B em `output/luz/`. Entra no `tudo` depois de `descritores` e na conferência de regressão.
- `entradas.corrigir_luz` (padrão false) troca as colunas de forma do decaimento pelas corrigidas; `entradas.trocar_por_luz`.
- Pré-registro: `luz.csv` em `arquivos_tabela`; seções `consolidacao` e `luz` em `secoes_config`.
- 10 testes novos (`test_luz.py`); `test_tudo.py` com a cadeia nova.

## [0.17.0] — 2026-11-04
- `consolidacao.canto_sem_piso` (padrão no config: true): o preto usa o canto real, relido dos CSVs por vídeo, sem o `max` com `extracao.piso_minimo`. Três travas (`canto_min_quadros`, `canto_max_variacao`, `canto_minimo_absoluto`); colunas novas `piso_canto_extracao_*` e `canto_fonte_*` em `videos.csv`. Muda 53 vídeos no B; a referência de regressão precisa ser regravada.
- 6 testes novos em `test_consolidacao.py`; tolerância de `test_integracao.py::test_preto_por_video_recuperado` de 0,01 para 0,02 (o canto agora vem dos quadros com ruído).
- README: pendências antes do treino em destaque no topo; anomalias 4, 7 e 8 (piso corrigido, anel rejeitado, regravação).

## [docs] — 2026-11-04 (só documentação)
- `docs/GUIA_DO_PROJETO.md`: guia em linguagem simples (glossário, fluxo, cada funcionalidade).
- `scripts/anomalias_amostras.py` e seção "Anomalias entre amostras" no README: possível rótulo trocado em `RB_05_1_v1a`, luz nos cantos x decaimento, canto preso no piso mínimo, tendência com a ordem das regravações.
- README: seção "B5: deriva de sessão e a amostra C" (o C destoa mais quando há deriva e apaga um pouco mais rápido; cuidados).
- README: ponteiro para o guia, fluxo em 4 fases, todos os comandos, árvore do repositório atualizada.

## [0.16.5] — 2026-11-04
- Trava do aval: `preregistro.aval_orientador` (padrão `null`). Sem a data do aval, ou com declaração "provisória", `preregistrar` recusa até com `--forcar` e `abrir_c` fica travado (`preregistro.pendencias_aval`, 5 testes novos).
- `entradas.bloquear_diagnosticos: true` no `config.yaml` (E1; valor igual ao padrão do código).
- Pré-registro de 20:42 guardado como `docs/preregistro_rascunho.json/.md`.

## [0.16.4] — 2026-11-04 (commit `0d72628`)
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
