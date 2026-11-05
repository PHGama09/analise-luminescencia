# Protocolo de validação (C17) — escrito ANTES de qualquer resultado de IA

Versão 0.13.0. Rascunho com padrões recomendados; as decisões marcadas **[D1…D6]** e **[E1]** foram respondidas pelo Pedro em 04/11/2026 (ver o quadro abaixo); D2 e D6 são provisórias até o orientador. Contagens calculadas de `dataset.csv` (236 vídeos); números de B5/B13 vêm de `output/deriva` e `output/colinearidade`. **Nada abaixo foi treinado ou testado com IA.**

## 0. Decisões do Pedro (04/11/2026)

| | Decisão | Estado |
|---|---|---|
| D1 | Alvo: regressão das 3 frações; classificação das 16 classes só como leitura secundária | decidida |
| D2 | Validação interna A↔B, leave-one-proporção-out só como estresse, C aberto uma vez no fim | **provisória** (levar as alternativas ao orientador) |
| D3 | Critério em dois níveis declarados antes de ver C: mínimo ≤ 70% e forte ≤ 50% do erro da média, mantendo (b) e (c) | decidida |
| D4 | Variante primária: `c0` + descritores, priorizando razões entre canais, `t50`, `r_5s` e `auc_10s` | decidida (declarar a dependência do início da janela) |
| D5 | Os 9 saturados ficam fora do treino e da validação | decidida para agora; rever depois |
| D6 | Declarar o limite de 1 preparação de validação; se o critério passar, planejar uma 2ª preparação (D) | **provisória** |
| E1 | Colunas de diagnóstico (`*_ok`, `*_r2`, `*_rmse`, `*_inicio_s`, `*_fim_s`, `*_n_pontos`) **bloqueadas como entrada**; uso só para controle de qualidade e pesos | decidida; implementada em `entradas.py` (`entradas.bloquear_diagnosticos: true`) |

Sobre E1: pesos por qualidade do ajuste só entram depois de conferir a distribuição do peso por proporção e por amostra (vídeos brilhantes ajustam melhor e o peso poderia criar outro viés), com teto no peso e comparação com e sem peso em A↔B. Se um diagnóstico parecer útil, testá-lo como variante **exploratória declarada** (`bloquear_diagnosticos: false`), nunca como padrão.

## 1. Objetivo e alvo
Prever a composição R/G/B do pó a partir do decaimento. As 16 proporções somam sempre 1,5, então o alvo é a fração `f = prop/1,5` (soma 1; 2 graus de liberdade). **[D1] Padrão: regressão das 3 frações** (saída normalizada para somar 1); a classificação das 16 classes entra só como leitura secundária (a fração prevista é encaixada na composição mais próxima; matriz de confusão). Justificativa: há só 2 preparações de treino por classe (uma classe, `G0.5_B1`, tem 1), e a regressão dá erro graduado (errar para a vizinha custa menos) e permite ver extrapolação. `prop_*` é só rótulo e agrupador, nunca entrada.

## 2. Divisão e dobras (sempre agrupadas)
Treino **A+B**, teste **C**, usado **uma vez**. Unidade independente = *preparação* (proporção × amostra); as 5 regravações são quase duplicatas e **nunca** se separam entre treino e validação. Treino real: 148 vídeos (157 menos 9 saturados) = **31 preparações**; C: 79 vídeos = 16 preparações. Vídeos por proporção em A+B: 5 (`G0.5_B1`) a 10; em C: 4 a 5.

| Esquema (só dentro de A+B) | Dobras | Treino / validação (vídeos) | O que mede |
|---|---|---|---|
| Leave-one-sample-out (A→B, B→A) | 2 | 74 / 74 | nova preparação de composição já vista (o mesmo que C fará); pessimista, treina com 1 preparação por classe |
| Leave-one-preparação-out | 31 | 143–145 / 3–5 | idem, com mais treino; variância alta por dobra |
| Leave-one-proporção-out | 16 | 138–143 / 5–10 | **extrapolação** a composição nunca vista; só para regressão; não serve para escolher |

**[D2] Padrão:** escolher entradas e hiperparâmetros por leave-one-sample-out (A↔B), com leave-one-proporção-out como teste de estresse; C só no fim. Nesses dois esquemas `G0.5_B1` nunca está em A (saturada): relatar as métricas com e sem essa proporção. Todo pré-processamento (padronização, imputação, poda por B13, escolha de variantes) é **ajustado só no treino de cada dobra**. Ressalva: os conjuntos de B13 usaram as 236 linhas (incl. C), sem o alvo; antes do treino final, refazê-los só com A+B.

## 3. Métricas e linhas de base
MAE por fração (R, G, B) e médio; R² por fração (contra a média do treino); erro por proporção (MAE e viés); IC 95% por *bootstrap de preparações* (16 unidades em C), nunca de vídeos; matriz de confusão e acurácia se classificação. **Bases:** (i) prever a média do treino (em C: MAE médio **0,255**; frações médias 0,349/0,331/0,320); (ii) k-vizinhos só com razões `razao_I0_*` e `razao_t50_*` (k escolhido em A↔B). Acaso na classificação: 1/16 = 6,25%.

## 4. Critério de sucesso (fixado agora) **[D3]**
Todos em C, com a variante escolhida antes: (a) MAE médio ≤ **70%** do MAE da média (nível **mínimo**) e ≤ **50%** (≈ 0,13, nível **forte**), os dois declarados agora; reportar qual nível foi atingido; (b) nenhuma proporção com MAE pior que o da média; (c) melhora sobre uma linha de base **sem aprendizado**, a regra fixa física (fração do canal proporcional ao seu `I0`, normalizada para somar 1, sem parâmetros ajustados; a variante `auc_10s` fica só como sensibilidade), em diferença pareada por proporção (16 pares), com IC 95% que não inclui 0. Os k-vizinhos ficam adiados por decisão do Pedro (nenhum modelo por enquanto); enquanto não houver modelo, a regra fixa é o piso deste critério. Falhou em (b) ou (c), ou ficou acima de 70% = não demonstrado. Os limiares são o padrão proposto; mudá-los depois de ver C invalida o teste.

## 5. Entradas
Variantes: caminho `cl`, `c0` ou ambos × sem/com descritores (6). **[D4] Padrão primário:** `c0` + descritores, conjuntos mínimos de B13 (`so_descritores`, `so_tau`, `misto`); `cl` e ambos como sensibilidade. As 6 variantes e os 3 conjuntos são comparados em A↔B; **uma** vai para C (as demais, se rodadas em C, ficam rotuladas como exploratórias: com 6+ variantes sobre 16 preparações, o melhor valor é otimista). `canais_sem_po: true` (todo canal com ajuste); o "sem" é sensibilidade, não equivale a "sem vazamento" (níveis se sobrepõem, ver `entradas.py`).
**Vazamento:** nunca `prop_*`, `canal_ativo_*`, `usar_no_ajuste_*`, `sat_ate_s`, `excluido_saturacao`, meta (`video`, `amostra`, `grupo`, `num_video`, `proporcao`) — `entradas.py` bloqueia; imputação e normalização não podem usar rótulo. Ausência estrutural: `R1.5` não tem G/B (e os saturados nada), então exigir todas as colunas derruba as 10 de `R1.5` (sobram 29 preparações); tratar a ausência como informação medida (`sinal_presente_*`/ajuste existente), nunca imputar com o rótulo. Colunas de diagnóstico do ajuste (`*_ok`, `*_n_pontos`, `*_fim_s`, `*_r2`, `*_rmse`) ficam fora dos conjuntos (descrevem a janela/ruído, não o decaimento).

## 6. Saturados e deriva
**[D5] Padrão:** os 9 saturados (A: 5, B: 4, C: 0) saem do treino e da validação (sem ajuste principal, sem imputar); reavaliar após B11. Consequência: `G0.5_B1` fica só com B (5 vídeos). **Deriva A/B/C (B5):** existe e supera a variação entre regravações; por isso C é outra preparação, a avaliação **nunca** pode ser por vídeo aleatório, e o CV entre regravações subestima o erro. Verificar, no resultado de C, se o erro por proporção segue o sentido da deriva (ex.: R com B/A e C/A < 1).

## 7. O que NÃO dá para concluir com 1 preparação de validação
C é **uma** preparação por composição: as 5 regravações dela medem só ruído de câmera, não variação entre preparações, então o erro em C é uma estimativa com n efetivo 16 e IC largo. Não se pode afirmar: generalização a composições fora da grade de 16, a outro lote de pó, outra câmera/iluminação/tempo de excitação ou outro operador; nem separar "deriva de sessão" de "variação de preparação". Alegação máxima: na grade de 16 composições, para uma nova preparação gravada com este protocolo. **[D6]** Padrão: declarar esse limite em qualquer resultado e, se o critério passar, planejar uma 2ª preparação de validação (D) antes de afirmar generalização.
