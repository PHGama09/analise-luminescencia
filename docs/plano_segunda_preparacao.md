# Plano da 2ª preparação de validação (D)

Versão 0.16.4. Rascunho só de texto: nada foi gravado nem alterado em vídeos ou CSVs. Números vêm de `output/deriva`, `relatorio_saturados.txt`, `protocolo_validacao.md` e da simulação `luminescencia ic_simulacao` (config padrão). Itens marcados **[ORIENTADOR]** dependem de decisão dele; D6 continua provisória.

## 1. Por que uma 2ª preparação
Em C há 1 preparação por composição (16 unidades). A simulação do IC mostra que, com a variação medida entre preparações, ir de 1 para 3 preparações reduz a largura do IC 95% do MAE para cerca de 0,57 a 0,58 da largura com 1 preparação (cenários mediana e p75), e que a parte sistemática por composição não encolhe com mais preparações (suposição 3 do relatório). Portanto D melhora o IC, não o elimina. D deve ser uma preparação **nova**, usada só para validar (nunca para escolher entradas ou hiperparâmetros) **[ORIENTADOR: confirmar que D entra como validação e não como treino]**.

## 2. Proporções a gravar
Critérios: (i) células com mais deriva A/B/C; (ii) composições sem preparação A utilizável; (iii) onde o critério D3(b) é mais sensível.

Deriva medida (fração das células proporção x grandeza com p_perm < 0,05 e eta² ≥ 0,3): G0.75_B0.75 e R0.75_B0.75 (96%; diferença relativa mediana entre amostras 30% e 28%), G1.5, R1_B0.5 e R0.25_G0.25_B1 (88%), B1.5 (64%, só 14 células). As de menor deriva: R1_G0.5 (8%), R1_G0.25_B0.25 (12%), G1_B0.5 (17%), R1.5 (17%, só 6 células por ausência estrutural de G/B).

Sem amostra A: `G0.5_B1` (os 5 vídeos de A estão entre os 9 saturados; fica só com B, 5 vídeos).

**Conjunto mínimo (8 composições, 40 vídeos com 5 regravações):**
1. `G0.5_B1` (sem A; hoje 1 só preparação utilizável).
2. `G0.75_B0.75` e 3. `R0.75_B0.75` (maior deriva; G0.75_B0.75 também teve 2 saturados em B).
4. `G1.5`, 5. `R1_B0.5`, 6. `R0.25_G0.25_B1` (deriva em ≥ 88% das células).
7. `B1.5` (deriva alta e R sem sinal: composição pura).
8. `R0.5_G0.5_B0.5` (a média das 16 frações é igual à sua fração; o erro da média nela é 0,000, então o critério (b) nela só passa com erro nulo; gravar D permite ver isso com uma 2ª medida, ver README_trecho_2C).

**Ideal:** as 16 composições (80 vídeos), para o D6 ficar simétrico com C. Se só der para o mínimo, o IC e o critério são reportados sobre as 8, nunca misturados com as 16 sem aviso **[ORIENTADOR]**.

## 3. Regravações
Manter 5 por preparação (igual a A/B/C) para o resultado ser comparável. O CV entre regravações mediano nas grandezas dos descritores é pequeno (ver relatório da simulação) e as regravações são quase duplicatas: mais preparações valem mais que mais regravações. Se o tempo apertar, 3 regravações é uma opção; o efeito no IC pode ser medido antes com `n_regravacoes: 3` no config (não foi rodado aqui) **[ORIENTADOR]**.

## 4. O que anotar por sessão
Uma linha por vídeo e um bloco por sessão, em arquivo próprio (`sessoes.csv`, que **não** entra como entrada de modelo, como os demais metadados):
- data e hora de início e fim; operador;
- lote do pó (cada canal R, G, B), validade, massa/volume de cada pó e do meio; id da preparação (proporção x amostra);
- câmera: modelo, resolução, fps, ganho/ISO, tempo de exposição, balanço de branco, foco (fixos e manuais, valores anotados);
- iluminação de excitação: fonte, distância, tempo de excitação, intensidade; luz ambiente residual;
- posição e distância da amostra, tempo entre excitação e início da gravação, tempo no escuro antes;
- temperatura e umidade, se houver termômetro;
- ordem das gravações na sessão e qualquer intercorrência (troca de bateria, reposicionamento, vídeo repetido e por quê).

## 5. Vídeos de controle (resolve a suposição do controle)
1 a 2 vídeos por sessão de uma **amostra de referência única e selada**, a mesma em todas as sessões, gravada no início e no fim da sessão. Serve para separar deriva do equipamento/sessão (câmera, excitação) da deriva da preparação, que hoje não se separam (o relatório de deriva diz que A/B/C é deriva total). Suposição declarada: a referência não envelhece de forma relevante entre sessões; checar a tendência dos controles antes de usá-los **[ORIENTADOR: definir qual composição é a referência; a de menor deriva medida, R1_G0.5, é um candidato, não uma decisão]**. Controles não entram na validação nem no treino.

## 6. Como evitar saturação
Medido: 9 vídeos saturados (sat_ate_s ≥ 2,5 s), 5 em A (todos de `G0.5_B1`) e 4 em B (`B1.5`, `G0.75_B0.75` x2, `RG_05_1`), com `sat_ate_s` perto de 2,7 a 3,5 s nos que o relatório lista. Medidas:
1. Antes da sessão, gravar 1 teste curto das composições mais brilhantes (`G0.5_B1`, `G0.75_B0.75`, `B1.5`) com os ajustes fixos e checar o pico por canal com o mesmo critério do pipeline (`sat_ate_s`).
2. Escolher uma **única** exposição/ganho para a sessão inteira que deixe a mais brilhante abaixo do limite; não usar ajuste automático.
3. Se mesmo assim saturar, regravar com menos excitação ou menos ganho e anotar; não misturar exposições dentro de uma preparação.
4. Mudar exposição em relação a A/B/C cria outra fonte de deriva: registrar nos controles **[ORIENTADOR: aceitar exposição diferente de A/B/C para D?]**.
5. Saturados continuam fora (D5), então saturar em D custa uma preparação; reservar 1 a 2 regravações extras nas composições do item 1.

## 7. Dependem do orientador
D como validação (não treino); conjunto mínimo de 8 contra 16; 3 contra 5 regravações; composição de referência do controle; exposição diferente de A/B/C; a regra para (b) onde o erro da média é 0 (R0.5_G0.5_B0.5).
