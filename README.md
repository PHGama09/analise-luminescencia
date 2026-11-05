# Análise de Luminescência por Vídeo

> **Projeto em desenvolvimento (v0.18.0; 293 testes).** Estão implementadas as etapas **1 (auditoria), 2 (ROI), 3 (extração), 4 (consolidação), 5 (pós-processamento), 6a (modelagem), 6b (tabela final)**, a **comparação de modelos** (6c), os **descritores sem modelo** (6d), os **gráficos** (7, agora com log-log, ln I × t e 1/I × t), a escolha de **entradas da IA** (`entradas.py`) e o **comando `tudo` (6e)** (cadeia de etapas, carimbo de versão e conferência de regressão). Também estão feitos o lote Q4 (análises de controle), os blocos 1 e 2 (avaliação sem IA, qualidade, simulação do IC) e a trava do aval do orientador. **Ainda faltam** (não feitos, por decisão): treino da IA e app. O que está marcado como "previsto" abaixo é plano e pode mudar.

> ⚠️ **Pendências que precisam ser resolvidas ANTES de qualquer treino** (detalhes em "Anomalias entre amostras"):
> 1. **`RB_05_1_v1a` (amostra A, treino) parece ser de outra composição (`RG_05_1`).** Assistir ao vídeo e decidir no `curadoria.csv` (`incluir = nao` ou nome corrigido em `nome_aprovado`). Enquanto isso, ele **entra no dataset com um rótulo provavelmente errado**, porque está como `pendente` com auditoria `ok`.
> 2. **Luz de fundo durante a gravação:** é o maior artefato conhecido da medição. A correção estatística já existe (`luminescencia luz`, v0.18.0), mas está **desligada** (`entradas.corrigir_luz: false`). Falta decidir se o treino usa as colunas corrigidas, antes do pré-registro e de preferência com o orientador. O teste de luz acesa × apagada, que confirmaria a causa, continua pendente.

> **Novo no projeto?** Comece por [`docs/GUIA_DO_PROJETO.md`](docs/GUIA_DO_PROJETO.md): explica em linguagem simples o que é o projeto, o glossário, o fluxo de trabalho e o porquê de cada funcionalidade. As decisões, com data, estão em [`docs/DECISOES.md`](docs/DECISOES.md).

## Objetivo

Transformar vídeos de amostras luminescentes em **dados numéricos limpos e comparáveis**: curvas de brilho e de cor ao longo do tempo, durante a excitação e o decaimento da luz.

Os dados serão usados para **treinar um modelo de IA**. Por isso a prioridade do projeto não é só extrair números, e sim garantir que **todas as amostras sejam medidas do mesmo jeito**. Qualquer diferença sistemática entre grupos de vídeos (outro fps, outra exposição, outra distância da câmera) pode ser aprendida pelo modelo no lugar da diferença real entre as amostras.

**O que são as amostras:** pós luminescentes em proporções diferentes de três componentes, identificados por amostra (A/B/C) e número do vídeo (1 a 5). O protocolo grava a fase de excitação e depois o decaimento (luz de excitação apagada logo no início, entre 0,2 e 0,4 s de vídeo), em arquivos de 15 fps (cerca de 8 quadros novos por segundo reais, pois ~47% dos quadros se repetem).

## Visão geral

O trabalho tem quatro fases. Cada etapa lê a saída da anterior e grava seus resultados em `output/<etapa>/`, com um `relatorio_*.txt`. Entre as etapas há pausas: você lê o que foi gerado e decide se avança ou se volta para refazer só aquela etapa, sem repetir as outras.

```
Fase 1 (vídeos, lenta)   auditar → roi → extrair
Fase 2 (dados, `tudo`)   consolidar → pos → modelar → tabela → comparar_modelos → descritores → luz → graficos
Fase 3 (só análise)      offset · deriva · colinearidade · qualidade · janela_r · saturados · ic_simulacao
Fase 4 (avaliação, sem IA)  entradas.py → avaliacao.py → avaliar_base → preregistrar (bloqueado até o aval) → C, uma vez, no fim
```

Diagrama completo e explicação de cada parte: `docs/GUIA_DO_PROJETO.md`, seção 3.

### Princípios

- **Dado puro primeiro.** O CSV da extração guarda só o que depende dos pixels. Transformações (HSV, Lab, gama, normalização) ficam para depois e são opcionais e reversíveis.
- **A etapa lenta roda uma vez.** Ler os vídeos é a parte mais demorada. Tudo que precisa dos pixels da ROI (mediana, desvio padrão, mín., máx., saturação) é calculado na extração, porque não dá para recuperar essas métricas a partir das médias.
- **Nada é alterado em silêncio.** A auditoria não modifica arquivos, e cada resultado guarda os parâmetros e a versão do código que o gerou.
- **O tempo do protocolo é o eixo oficial.** Todos os vídeos usam o mesmo eixo de tempo, sem realinhamento individual.
- **Um vídeo com problema não derruba o lote.** Os erros são registrados e o processamento continua.

## As etapas

### 1. Auditoria dos vídeos
**Entrada:** pasta de vídeos. **Saída:** `relatorio_auditoria.csv/.txt` em `output/auditoria/`, resumo no terminal e `curadoria.csv` atualizado.

Roda antes de tudo e pode ser repetida sempre que chegarem vídeos novos (só os novos ou alterados são lidos de novo). Cada vídeo recebe um `status`:

- **ok:** nada a apontar;
- **revisar:** legível, mas pede conferência manual (a decisão é sua e nada é renomeado, movido ou excluído);
- **fora_do_padrao:** fora do protocolo (fps, duração, resolução, frames repetidos acima da maioria etc.);
- **ilegivel / truncado:** não pôde ser lido por inteiro.

O que a auditoria confere:

- **Nome do arquivo** (proporção, amostra A/B/C, número do vídeo). Erros de digitação simples (espaço, `__`, falta de `_` antes do `v`) são lidos, mas o vídeo vai para **revisar** com um nome sugerido. Diferença só de maiúscula/minúscula (ex.: `v4A`) vira apenas um aviso; com `nomes.maiusculas_sao_erro: true` ela volta a ir para a revisão.
- **Pasta contra nome:** se a pasta (`Amostras C/4/`) não bate com o nome (`..._v4a`), o vídeo vai para **revisar** com a opinião do programa (nome errado, pasta errada ou vídeo extra), baseada em quais combinações já existem.
- **Cópias e repetidos:** mesma combinação (proporção + amostra + vídeo) em mais de um arquivo, com veredito de conteúdo idêntico ou diferente (por hash).
- **Detecção da queda da luz:** procura, só nos primeiros segundos (`deteccao_queda`, hoje 0 a 1 s), o instante em que a luz de excitação apaga. É só verificação: não alinha nada, e quando falha o vídeo vai para **revisar**.
- **Duração, fps e resolução** contra o protocolo (60 s ± 1 s, 15 fps). A **moda** do lote funciona como alerta secundário.
- **Frames repetidos (modo `--completo`):** conta frames idênticos ao anterior e calcula `fps_efetivo`. Nos vídeos atuais ~47% dos frames se repetem (~8 frames novos por segundo). Se quase todos têm a mesma fração, é um alerta do lote (característica do captador); só os que fogem da maioria viram problema.
- **Grupo (treino/validação):** coluna `grupo`, derivada da letra da amostra (`divisao` no config: A e B treino, C validação). Só para conferência visual; vídeo com pasta diferente do nome fica `indefinido`.
- **Pastas ignoradas** (`auditoria.pastas_ignoradas`): não são lidas. Hoje: `Proporcao de Branco 6000` (gravada com dados errados). Os arquivos continuam no disco.
- **Exposição automática:** ISO, exposição e balanço de branco só são conferidos se estiverem nos metadados (precisa do `ffprobe`; AVIs do OpenCV não os guardam, e então o relatório avisa).

**Curadoria (`curadoria.csv`, na raiz do projeto):** suas decisões por vídeo (`incluir` sim/nao/pendente, `motivo` de uma lista fechada, `nome_aprovado`, `observacao`). O vídeo é identificado pelo **hash do conteúdo**, então renomear ou mover o arquivo não perde a decisão. A auditoria só acrescenta vídeos novos e atualiza caminho, status e `nome_sugerido`; as suas colunas nunca são sobrescritas, e a versão anterior fica em `curadoria.csv.bak`. As etapas 2 e 3 pulam os vídeos com `incluir = nao`; `sim` e `pendente` entram.

### 2. Detecção da amostra (ROI)
**Entrada:** vídeos e `curadoria.csv`. **Saída:** `output/roi/roi.json` (uma ROI por vídeo), `roi.csv`, miniaturas e mosaicos de conferência.

Tudo automático:

- Depois que a luz de excitação apaga, o único brilho do quadro é o da própria amostra. A média dos quadros **novos** entre 0,6 e 1,5 s (`roi.janela_s`) é binarizada (Otsu) e o maior objeto vira o círculo. A ROI usada é 88% do raio detectado (`roi.margem_interna`), para fugir da borda borrada.
- O vídeo de **controle** (sem brilho) herda a ROI mediana dos outros vídeos.
- Um vídeo é **suspeito** por regras objetivas, com o motivo escrito: forma pouco circular, contraste baixo, objeto encostado na borda, área fora do esperado, raio diferente da mediana do lote, ROI herdada porque a detecção falhou, ou controle com brilho. O **centro não é critério**, porque o recipiente muda de posição entre vídeos.
- **Mosaico** (`mosaico_*.png`): as primeiras folhas reúnem os suspeitos; círculo verde = ROI usada, amarelo = detectado. Pode ser refeito sem reler os vídeos (`--so-mosaico`).
- **Correção manual:** edite a entrada em `roi.json` (campo `roi`: cx, cy, r) e ponha `"manual": true`; ela passa a ser preservada.
- Previsto, ainda não implementado: conferência um por um com Enter/Espaço e definição da ROI por clique.
- A ROI guarda uma `forma`, hoje só o círculo; outras formas podem ser adicionadas depois.

### 3. Extração dos dados (o núcleo)
**Entrada:** vídeos, `roi.json` e `curadoria.csv`. **Saída:** `output/extracao/<nome do vídeo>.csv` e `extracao_resumo.csv`.

Uma linha por quadro **novo** (quadro idêntico ao anterior é descartado, e `repeticoes` guarda quantas vezes ele se repetiu). Colunas:

- `frame`, `tempo_s` (índice do quadro / fps do contêiner, na primeira ocorrência) e `fase` (excitacao ou decaimento);
- `media_R/G/B` e `mediana_R/G/B` dos pixels da ROI, `desvio_cinza` (irregularidade espacial) e `saturados_pct`;
- `fundo_R/G/B` (anel logo fora da ROI, que pode conter halo da própria amostra) e `canto_R/G/B` (4 cantos do quadro, nível de preto da câmera).

**Corte entre excitação e decaimento:** um valor fixo e igual para todos os vídeos (`protocolo.instante_queda_s`, hoje 0,5 s). A queda real medida fica entre 0,20 e 0,40 s, e o corte não é realinhado vídeo a vídeo, para não criar diferenças entre grupos.

**Piso de ruído:** o vídeo de **controle** (`Amostra_referencia`, sem amostra luminescente) dá o nível de preto da câmera: média pixel a pixel dos quadros novos depois do corte, lida **dentro da ROI de cada vídeo**. Sem controle, usa os cantos de cada vídeo (`extracao.piso_fonte: cantos` força isso). O piso **não é subtraído** dos dados: alimenta só o aviso `perto_do_ruido_*` e a `deriva_preto_*` (se o preto mudou entre as gravações). Isso é refeito a cada execução a partir do resumo.

`extracao_resumo.csv` traz, por vídeo, a ROI usada, quadros novos e repetidos, saturação, níveis inicial e final por canal, e `status_auditoria` e `incluir`, para filtrar depois sem apagar nada.

### 4. Consolidação
**Entrada:** `output/extracao/` (um CSV por vídeo + `extracao_resumo.csv`) e o `curadoria.csv`. **Saída** (`output/consolidacao/`):
`videos.csv` (uma linha por vídeo aceito), `series.csv` (formato longo, uma linha por quadro novo) e `relatorio_consolidacao.txt`.

- **Quem entra:** `incluir = sim`; `pendente` com auditoria `ok` (`consolidacao.incluir_pendentes_ok`); ficam de fora `nao`, o controle e `pendente` com alerta (ex.: `C/1/RGB_05_05_05_v1a`).
- **Identificação:** proporção dos 3 pós, amostra (A/B/C), grupo (treino/validação) e número do vídeo, lidos do `nome_aprovado` (ou do nome do arquivo). Avisa se a pasta e a letra do nome divergem. Nada é renomeado.
- **Correção do preto por vídeo** (`consolidacao.piso: video`): `piso_video = piso_canto_do_vídeo - (piso_canto_do_controle - piso_ref_do_controle)`. O controle só mede a diferença cantos/ROI (~0,9); o nível de preto de cada sessão vem dos cantos do próprio vídeo. Os valores brutos continuam nas colunas sem sufixo; os corrigidos são `inicio_corr_*`, `final_corr_*` (resumo) e `corr_*` (séries). **Suposição testada na P5 (v0.11.2, ver "P5" abaixo):** a diferença cantos/ROI vale igual para todas as sessões; vale em média, mas não é constante em todos os vídeos.
- **Canto sem o piso mínimo** (v0.17.0, `consolidacao.canto_sem_piso: true`): o canto usado no preto é o real, relido dos CSVs por vídeo, e não o `max(canto, 1,0)` da extração. Um canto abaixo de 1,0 só vale se passar em três travas (`canto_min_quadros`, `canto_max_variacao`, `canto_minimo_absoluto`); senão volta a 1,0 e o motivo fica em `canto_fonte_*`. O valor da extração continua em `piso_canto_extracao_*`. Afeta só o B (53 vídeos). Ver "Anomalias entre amostras", item 4.
- **Conferência no relatório:** desvio padrão do nível final entre as 5 regravações de cada proporção/amostra, com piso fixo x piso por vídeo, só nos canais cujo pó está na mistura. Nos 236 vídeos aceitos caiu de 1,32 para 0,45 (R), 2,18 para 1,29 (G) e 3,62 para 2,91 (B). Sobra dispersão no B, que tem outras fontes (saturação, posição da amostra).
- Alertas aceitos aparecem no relatório. Não há modo incremental: a consolidação é refeita inteira a cada execução (é rápida).

### 5. Pós-processamento e análise
**Entrada:** CSV consolidado. **Saída:** CSV com colunas derivadas e tabelas de métricas.

**Implementado (v0.6.0, `luminescencia pos`; sem mudanças na v0.9.0):** lê `output/consolidacao/` e grava em `output/posprocessamento/` `series_pos.csv` (só decaimento, a partir de 2,0 s), `videos_pos.csv` e `relatorio_posprocessamento.txt`. Não altera valores nem reamostra o tempo.

- **Janela fixa:** todos os vídeos começam em `posprocessamento.inicio_ajuste_s` = 2,5 s (decidido em 02/11/2026), depois da saturação da grande maioria. Sem janela por vídeo, para não criar diferença entre grupos.
- **Saturação:** 9 de 236 vídeos ainda saturam depois de 2,5 s (35 depois de 2,0 s; 5 depois de 3,0 s). Ficam marcados (`satura_apos_*`) e fora do ajuste principal (`usar_no_ajuste_* = False`); os dados continuam em `series_pos.csv` para ajuste à parte.
- **Canais (revisado em 02/11/2026):** entra o canal com sinal medido (fora do ruído e sem saturação depois de 2,5 s), com ou sem o pó na mistura (`usar_no_ajuste_*`, com `motivo_fora_*`; `sinal_presente_*` é medido). A decisão **não olha a proporção do nome do arquivo**, para a ausência do ajuste não revelar o alvo à IA. `canal_ativo_*` continua nas tabelas só para conferência.
- **Sensibilidade ao instante inicial** (2,0 / 2,5 / 3,0 s): o recorte é feito no ajuste, por `tempo_s`.
- **Aviso, não exclusão:** canal com mais de 10% dos pontos corrigidos <= 0 (cauda no ruído) recebe `aviso_cauda_no_ruido_*`.
- Ainda não feitos (opcionais, dependem de decisão): linearização de gama, métricas de cor, checagem de sincronização.

Tudo é opcional, configurável e reversível:

- **Linearização de gama:** vídeos guardam o brilho em curva (gama ~2,2), não proporcional à luz. Como o decaimento físico é descrito em termos de luz, converter para valores lineares evita distorcer a forma da curva. Câmeras de celular aplicam processamento próprio, então o resultado nunca é perfeitamente linear.
- **Métricas de cor** (escolha no arquivo de configuração):
  - **HSV, matiz (H):** a cor como ângulo no círculo cromático. Mostra bem a mudança de cor, mas fica ruidoso no escuro.
  - **Lab:** luminosidade (L) e dois eixos de cor (a: verde↔vermelho, b: azul↔amarelo). A distância entre cores se aproxima da diferença percebida (ΔE).
  - **Razão B/G:** divisão entre canais. Mostra direto qual componente decai mais rápido. Só faz sentido depois de linearizar a gama e subtrair o fundo.
- **Métricas de análise:** desvio padrão, intensidade máxima, tempo para cair a X% e comparações entre amostras.
- **Checagem de sincronização** (versão 1, forma simples): detecta o instante em que a luz apaga (nos primeiros 1 s do vídeo) e compara com a mediana do lote. O desvio fica numa coluna de qualidade (`desvio_queda_s`), e vídeos acima da tolerância são marcados para revisão. **Isso não realinha o tempo.** Realinhar só entra se o usuário escolher, igual para todo o conjunto e registrado no arquivo auxiliar, porque realinhar só alguns vídeos criaria diferenças artificiais entre grupos.

### 6a. Ajuste em lote (`luminescencia modelar`, v0.7.0)
Lê o pós-processamento e grava `output/modelagem/ajustes.csv` (um ajuste por vídeo x canal x caminho) e `relatorio_modelagem.txt`. Modelos com `C = 0` (`exp_simples_c0`, `exp_dupla_c0`) e os de `C` livre (`exp_simples`, `exp_dupla`) em `modelos.py`.

Caminhos: `principal` (C = 0, janela fixa de 2,5 s), `inicio_2s` / `inicio_3s`, `piso_fixo` (C = 0 sobre o piso fixo do controle), `c_livre` (todos os vídeos) e `a_parte` (vídeos que saturam depois de 2,5 s, começando depois da saturação; janela própria, **não misturar com o principal**). O fim da janela de cada sinal é onde a mediana móvel deixa de passar de 3x o ruído do canal (decidido para o R; a mesma regra vale para G e B, onde quase não atua).

**Resultados com os 236 vídeos (02/11/2026):**
- A exponencial dupla venceu em 100% dos ajustes principais (a simples quase nunca atinge R² >= 0,99 nem vence no BIC).
- **Preto:** trocar o piso por vídeo pelo piso fixo muda o tau em 4 a 10% (viés ~ -1%), no nível da variação entre regravações (CV 8 a 11% no principal). O tau não depende da correção do preto além do ruído.
- **Instante inicial:** começar em 2,0 s dá tau ~7% menor, e em 3,0 s ~6% maior. Viés sistemático e parecido nos três canais: o decaimento não é uma dupla exponencial perfeita, então o tau depende da janela. Vale igual para todos os vídeos, mas o tau não é uma constante intrínseca.
- **`C = 0` x `C` livre:** o `C` livre acha um patamar positivo (G ~4,7 e B ~6,5 níveis, estável entre amostras) e erra a série cerca de 2x menos (rmse 0,24 contra 0,55 no G), embora ainda acima do ruído (0,05). O tau muda 26% (G) e 32% (B), muito mais que o ruído. O `C` livre foi **mais** repetível entre regravações que o `C = 0` (CV 3,7% contra 7,5% no G; 4,5% contra 9,1% no B) e tem menos diferença entre amostras (B: 10,9-11,2 s contra 15,5-16,9 s com `C = 0`). No R (patamar ~0) os dois são equivalentes e o `C` livre é um pouco pior (CV 11,8% contra 11,3%). Números da rodada de 02/11/2026 com os 3282 ajustes (versão anterior deste README trazia B ~8,7, de uma rodada antiga).


**Limite do patamar no C livre (P2, v0.10.0; valor final decidido na Q1, v0.11.0: -1,35).** `modelagem.c_livre_c_minimo` (padrão **-1.35** desde a v0.11.0, era `0.0` na v0.10.0; `null` volta ao comportamento da v0.9.0) impede o patamar `C` do caminho `c_livre` de ficar abaixo do valor dado. `ajustes.csv` ganha a coluna `C_no_limite` (C a menos de 0,01 nível do limite) e o relatório conta quantos ajustes ficaram colados. Vale também para o conjunto `cl` de `comparar_modelos`, para os dois falarem do mesmo "C livre". Os outros caminhos (principal, inicio_2s/3s, piso_fixo, gama, a_parte) ficaram **idênticos** aos da v0.9.0 (diferença máxima de tau 0,00002 em 3933 ajustes).

*O que foi medido (só o R muda de forma relevante; G não muda, B muda em 1 vídeo):*

| Limite de C (R, 227 ajustes) | colados no limite | ajustes que mudam | tau > 30 s |
|---|---|---|---|
| sem limite (v0.9.0) | n/a | n/a | 3 |
| **C >= 0 (padrão da v0.10.0)** | **106 (47%)** | **100** | 0 |
| C >= -1 | 13 | 11 | 0 |
| C >= -2 | 3 | 3 | 0 |
| **C >= -1,35 (padrão da v0.11.0)** | **6** | **6** | 0 |

- **A premissa da decisão não se confirmou:** o limite C >= 0 **não** fica restrito aos 3 vídeos degenerados (`GB_075_075_v1a`, `GB_1_05_v1a`, `RB_05_1_v3a`). 103 dos 227 ajustes de R tinham C < 0 (33 abaixo de -0,5; o resto entre -0,5 e -1,6, ordem de grandeza do erro do preto por vídeo). Com C >= 0, 100 ajustes de R mudam (mediana de |delta tau| 7,9%, máximo 35,7%).
- Os 3 degenerados ficam normais (tau 80-120 s -> 6,4-9,8 s; C -23 a -33 -> 0), mas com C = 0 o ajuste deles **é igual ao `c0`** (mesmo tau).
- **Consequência para a tabela:** em 47% dos sinais de R, `R_cl_*` passa a ser cópia de `R_c0_*` (54% com tau dentro de 2%). Mais redundância entre os dois caminhos (risco 2 abaixo). CV do tau no R: 11,8% -> 11,3%; o rmse mediano do R com C livre sobe de 0,151 para 0,166 (C = 0: 0,180).
- **Um vídeo troca de modelo:** `B_15_v3b` (R) passa de dupla para simples (tau 7,6 -> 4,9 s). O `RG_1_05_v2a` (B) tem C -0,69 e vira C = 0 (tau muda 9%).
- **Decisão Q1 (03/11/2026): `c_livre_c_minimo: -1.35`.** Primeiro se pensou em -2, que passa nos critérios fixados antes do teste (os 3 degenerados com tau normal, nenhum outro sinal muda, poucos colados) e deixa o resto do dataset como na v0.9.0. Mas -2 vem de uma folga nos dados, não de uma regra. O valor adotado é **3 x 0,45**, onde 0,45 é o erro do preto medido no R (desvio do nível final entre regravações com o piso por vídeo, `relatorio_consolidacao.txt`); é uma estimativa aproximada, porque esse desvio inclui outras fontes. Teste feito com os dados reais, só no caminho `c_livre`:
  - Em relação à v0.9.0 (sem limite) mudam **6 de 651 sinais**, todos no R, nenhum troca de modelo: os 3 degenerados (tau 93/80/120 s -> 9,1/7,7/11,4 s) e `RGB_025_1_025_v4a`, `G_15_v4b`, `RGB_025_1_025_v4b` (2 a 3,5%). Os outros 6 caminhos ficam idênticos ao da v0.10.0 (diferença máxima de tau 0,00002).
  - **Colados no limite: R 6 de 227 (3%), G 0, B 0.** O `R_cl_*` é cópia do `R_c0_*` (tau dentro de 2%) em cerca de 15% dos sinais do R, contra 54% com C >= 0.
  - **Limite do método:** o tau desses 6 sinais **depende do limite**. Os 3 degenerados querem C de -22 a -33; com limite 0, -1, -1,35 e -2 o tau deles é 7,4 / 8,6 / 9,1 / 10,2 s (exemplo `GB_075_075_v1a`). O limite só evita o absurdo (tau de 80 a 120 s); não faz o ajuste identificar o patamar. Eles ficam marcados por `C_no_limite = True` em `ajustes.csv`.
  - Números de referência (v0.11.0): CV do tau `c_livre` R 11,8 / G 3,7 / B 4,5%; tau por amostra (A/B/C) no R 7,4/6,3/6,2; rmse do R com C livre 0,15 (ruído 0,05); sensibilidade `c_livre` contra o principal no R 6,2% (viés -0,6%); `c_livre`: dupla 650, simples 1 (`B_15_v4b`, R). Se a sessão de gravação mudar, refazer o teste.

### 6b. Tabela final (`luminescencia tabela`, v0.8.1)
Grava em `output/tabela/` `dataset.csv` (uma linha por vídeo aceito, formato largo), `dicionario.csv` (papel de cada coluna: meta, alvo, feature, flag) e `relatorio_tabela.txt`.

- **Decididos em 02/11/2026:** os dois caminhos (`cl` = C livre, `c0` = C = 0) vêm lado a lado para treinar só com um ou com ambos e para comparar com o orientador; o mesmo caminho vale nos três canais (R, G, B), para padronizar e reaproveitar em projetos futuros. Antes de entregar à IA, rodar o treino com três conjuntos (só `cl`, só `c0`, ambos) e comparar na amostra C.
- **Não usar como entrada da IA:** colunas `meta` (nome do vídeo, amostra, grupo) e `canal_ativo_*` (vem de `prop_*`, ou seja, do alvo).
- **Vídeos que saturam depois do início da janela** (2,5 s; 9 vídeos; desde a v0.11.0 o `tabela.py` lê `posprocessamento.inicio_ajuste_s` em vez de ter o 2,5 fixo) ficam sem ajuste principal; o ajuste à parte fica só em `ajustes.csv`.
- **Vazamento do alvo (resolvido em 02/11/2026, v0.8.1):** o ajuste é decidido pelo sinal medido. Canal sem o pó mas com sinal (vazamento de cor da câmera) também é ajustado: 67/75 no R, 59/75 no G e 57/73 no B. O tau dele descreve o vazamento de cor, não o pó daquele canal, mas é repetível entre regravações (CV mediano 3 a 10%, R² > 0,999). Entradas da IA: não usar `meta` nem `canal_ativo_*`; `*_ok` e `sinal_presente_*` são medidos no vídeo e podem ser usados.

### 6c. Comparação de modelos (`luminescencia comparar_modelos`, v0.9.0, decisão D3)
Compara **só para análise** a exponencial simples, a dupla, a **exponencial esticada** (`A*exp(-(t/tau)^beta)`) e a **lei de potência** (`A*(1+t/t0)^-alpha`), na mesma janela do ajuste principal, em dois conjuntos: `c0` (C = 0) e `cl` (C livre, incluindo a esticada e a potência com C). Grava `output/modelagem/comparacao_modelos.csv` e `relatorio_comparacao_modelos.txt`. **Nada disso entra no caminho principal nem na tabela final** (o principal continua sendo a exponencial dupla, como decidido; a simples quase nunca serve).

Como comparar formas diferentes: BIC (penaliza parâmetros) e o erro em relação ao ruído do canal; a repetibilidade é medida em `t_meio` (tempo em que a curva ajustada cai à metade), que não depende da forma do modelo, porque o tau de uma esticada ou o t0 de uma potência não são o `tau_medio` da dupla.

**ATENÇÃO (v0.10.0, P2):** o conjunto `cl` agora usa o mesmo limite do `c_livre` (`modelagem.c_livre_c_minimo`, padrão 0). Os números de `cl` abaixo são **os da v0.9.0 (sem limite)**; com C >= 0 o quadro muda (ver o fim desta seção). Os de `c0` não mudam.

**Resultado nos 236 vídeos (651 sinais; 02/11/2026) e o que ele contraria:**
- A exponencial **dupla vence a simples e a esticada** em todos os canais, mas **não é o melhor descritor em G e B**. No conjunto `c0`, a **lei de potência com 3 parâmetros vence a dupla de 4 parâmetros em B** (177 contra 32 vitórias no BIC; R² mediano 0,99938 contra 0,99910) e fica empatada em G (94 contra 107). Com C livre, a potência com C (4 parâmetros) vence em G (139 contra 70) e em B (118 contra 92) e a dupla (5) vence em R (176 contra 40). O expoente da potência fica perto de 1 em B/G (mediana 1,1-1,2 com C = 0; 0,7-0,8 com C livre).
- O **R** não pede cauda lenta: a dupla vence (186 de 227 com C = 0).
- A repetibilidade de `t_meio` é parecida entre dupla, potência e esticada em G e B (CV 2 a 4%). Com C = 0 a simples é a menos repetível (8 a 12%); com C livre a potência é a menos repetível (4 a 13%).
- **Cuidado com a leitura:** com expoente próximo de 1 a lei de potência não tem escala de tempo própria (o tempo para cair à metade depende de onde a janela começa). Isso é coerente com o que já se viu (tau muda ~7% ao mover o início da janela) e é um indício de que o decaimento tem cauda tipo potência, mas **com 60 s de vídeo e só R² não prova** o mecanismo. Nenhum modelo extra foi adotado; isso é decisão sua (ver "Pontos em aberto").

**O que a v0.10.0 mudou nesse resultado (C >= 0 no conjunto `cl`):** a vantagem da lei de potência **com C** em G e B **vinha de um patamar C negativo**. Na v0.9.0, as 139 vitórias da potência com C em G e as 118 em B usaram C < 0 (mediana -7,3 níveis em G e -6,3 em B; na potência com C, C < 0 em 100% dos sinais de G), um patamar sem sentido físico (o sinal corrigido do preto não pode ficar abaixo de ~0 além do erro do preto, ~1 a 3 níveis). Com C >= 0 (v0.10.0, sem tocar em nada mais), o placar `cl` vira: G: dupla 207 x potência 2 (antes 70 x 139); B: dupla 198 x potência 12 (antes 92 x 118); R: dupla 218 x potência 4 (antes 176 x 40). O rmse/ruído da dupla com C livre continua 3,6 (R), 4,7 (G) e 2,4 (B). **O que fica de pé:** com C = 0 (`c0`) a potência ainda vence em B (177 x 32) e empata em G, porque ali não há C para imitar a cauda. Ou seja, a indicação de cauda tipo potência existe só com C = 0; com C livre e fisicamente plausível a dupla é o melhor modelo. Isso reforça a decisão P1(a) (a potência fica só como comparação).

**Placar `cl` final (v0.11.0, C >= -1,35):** R dupla 201 x potência 14 x esticada 12 (simples 0); G dupla 197 x potência 12 x esticada 3; B dupla 190 x potência 20 x esticada 2. Com C >= 0 a potência quase não vencia (G 2, B 12); com -1,35 ela pode usar um C ligeiramente negativo e ganha um pouco, mas a dupla continua vencendo em 88 a 93% dos sinais (na v0.9.0, sem limite, a potência vencia 139 em G e 118 em B). O resto do conjunto `c0` não muda.

### 6d. Descritores sem modelo (`luminescencia descritores`, v0.10.0, decisão P3 opção b)
Grava em `output/tabela/` `descritores.csv` (uma linha por vídeo aceito, as mesmas 236 do `dataset.csv`; junte pela coluna `video`), `dicionario_descritores.csv` e `relatorio_descritores.txt`. **É uma tabela paralela: o `dataset.csv` continua com 106 colunas.** Só usa o sinal medido (`corr_*`), sem ajustar equação nenhuma, e tem descritor nos mesmos sinais que têm ajuste principal (regra do sinal medido, nunca `prop_*`; 227 R, 212 G, 212 B). Colunas `<canal>_md_<descritor>`:
- `I0` (média dos 3 primeiros pontos da janela), `t50` e `t10` (tempo para I/I0 cair a 50% e a 10%, interpolado; vazio se não cruza na janela ou se o nível-alvo não passa de 3x o ruído), `r_2s`, `r_5s`, `r_10s`, `r_20s`, `r_40s` (I/I0 em tempos fixos; vazio se o sinal já sumiu no ruído antes) e `auc_10s`, `auc_30s` (área sob I/I0, em s). Mais as razões entre canais `razao_I0_*` e `razao_t50_*` (R/G, G/B, R/B) e a flag `<canal>_md_ok`.
- **Declare sempre a janela:** tempo em s desde 2,5 s do vídeo (`posprocessamento.inicio_ajuste_s`). Parâmetros em `descritores:` no `config.yaml`.

*Resultado nos 236 vídeos (03/11/2026):*
- **Repetibilidade entre as 5 regravações (CV mediano):** `t50` R 7,8% / G 3,0% / B 3,9%, contra o tau principal 11,3 / 7,5 / 9,1% e o tau C livre 11,3 / 3,7 / 4,5%. `r_2s`, `r_5s`, `auc_10s` e `I0` ficam entre 0,8% e 5,7%. Mais estáveis que o tau com C = 0 nos três canais, e no nível do tau com C livre sem depender de modelo.
- **Os valores longos não servem para o R:** `r_20s` e `r_40s` têm CV de 35% e 42% no R (o sinal está no ruído), e `r_40s` existe só em 118 dos 227 sinais de R; `t10` fica entre 9% e 15% em todos os canais. Para o R, os descritores confiáveis são `I0`, `t50`, `r_2s`, `r_5s`, `r_10s` e `auc_10s`. **Decisão Q2 (03/11/2026, v0.11.0):** as colunas `R_md_t10`, `R_md_r_20s` e `R_md_r_40s` **continuam na tabela**; só o `dicionario_descritores.csv` ganhou um AVISO na descrição delas (lista em `descritores.nao_confiaveis_R`; `[]` desliga o aviso).
- **Eles dependem do início da janela, e mais que o tau:** mover o início para 2,0 ou 3,0 s muda `t50` em 9 a 12% (o tau muda ~7%), `t10` 4 a 8%, `r_10s` 5 a 9% e `auc_10s` 3 a 6%. É esperado: I0 é lido no início da janela, e o decaimento não é uma exponencial pura. **Mais repetíveis entre regravações não significa menos dependentes da janela.**
- Os valores de `t50` aqui (mediana R 3,6 s, G 5,4 s, B 5,9 s; por vídeo e todos os canais com sinal) **não** são os do `graficos_resumo.txt` (4,5 / 5,5 / 7,5 s): o gráfico 4 usa a mediana por proporção em intervalos de 1 s e só os canais com o pó.
- **Cuidados:** canal sem o pó com sinal também tem descritor (descreve o vazamento de cor); `video`, `amostra`, `grupo`, `num_video` e `proporcao` são `meta` e não podem ser entrada da IA. Desde a v0.11.1 eles podem entrar nas entradas da IA por uma chave **desligada por padrão** (`entradas.descritores`, ver "Entradas da IA").

### P5. Teste do offset cantos - ROI (`luminescencia offset`, v0.11.2)
Só análise (nada do pipeline muda). Grava em `output/offset/` `offset_por_video.csv` e `relatorio_offset.txt`. **A suposição:** a correção do preto por vídeo (etapa 4) tira dos cantos de cada vídeo o offset "cantos - dentro da ROI" medido no controle (R 0,87, G 0,79, B 1,03). **O teste, sem vídeo novo:** num vídeo em que um canal não tem sinal nenhum, o nível final dentro da ROI é só o preto da câmera, então `offset medido = piso_canto - final_ROI`. "Sem sinal" é regra **medida** (`perto_do_ruido_*` e nível inicial a menos de `offset.limiar_vazamento` = 3 níveis acima do canto), nunca a proporção do nome. Só o **B das gravações de R_15** passa (14 vídeos); o G de R_15 começa ~5 níveis acima do preto (vazamento de cor) e não serve.

Resultado nos 236 vídeos (03/11/2026):
- **Vale em média no B:** offset medido médio 1,10 (mediana 0,95) contra 1,03 do controle: erro médio **+0,07**, dp 0,62; 13 de 14 vídeos a menos de 1 nível. **Sem diferença detectável entre amostras** (A 0,94, B 1,53, C 0,91; Kruskal-Wallis p = 0,50, mas com 4 a 5 vídeos por amostra o poder é baixo).
- **Não é constante:** o pior vídeo é `R_15_v4b` (cantos 5,17, ou 2,9 níveis acima do controle), com offset medido 3,03: a correção deixa o piso **2,0 níveis acima do real** (a série corrigida fica 2 níveis baixa). Sem ele o dp do erro cai de 0,62 para 0,29. Na reta `final_ROI = a + b x canto` (cantos acima do piso mínimo, n = 9), **b = 0,45** (r = 0,97): quando o canto sobe 1 nível, o ROI sobe só ~0,45. Se o offset fosse constante, b seria 1.
- **A alternativa linear não é melhor:** trocar a correção por `a + b x canto` (calibrada nesse B) **piora** a dispersão do nível final entre regravações (R 0,85 contra 0,45; G 1,75 contra 1,29; B 3,26 contra 2,91; piso fixo: 1,32 / 2,18 / 3,62). Com 9 pontos de um só canal, não fica provado que ela generalize. **A correção atual fica como está.**
- **Quantos vídeos estão expostos:** 22 (R), 20 (G) e 29 (B) vídeos têm cantos mais de 2 níveis acima dos do controle (máx. 3,4 a 3,6), concentrados nas gravações `v4b` (todas as proporções) e `v5a`. Para saber se a correção erra nelas também no R, usei o patamar C do ajuste livre sem limite (análise pontual, **não reproduzida pelo comando**): se o preto estivesse 2 níveis alto, o C do R seria ~-2; o C mediano fica em 0,04 nos vídeos com cantos altos e 0,06 nos demais (Mann-Whitney p = 0,92; sessões `4b` 0,15 e `5a` 0,01). **No R não há sinal de erro nessas sessões**; o erro de +2,0 apareceu no B de `R_15_v4b`.
- **Limite que a análise expôs:** 53 dos 236 vídeos (22% no total; 22 na amostra C) têm o canto do B **no piso mínimo da extração** (`extracao.piso_minimo` = 1,0; no R são 4, no G 2). O canto real pode ser menor e a correção não consegue seguir a sessão mais escura; além disso, o nível dentro da ROI não pode ficar abaixo de 0. Ligação com a Q1: o erro do preto de 0,3 a 0,6 nível (B em R_15) é da mesma ordem dos 0,45 usados no limite de C (3 x 0,45 = 1,35); um vídeo como `R_15_v4b` (erro 2,0) passaria dele, mas só no B.
- **O que não se sabe:** se o offset varia por canal, por posição da amostra ou por sessão inteira; só há 14 vídeos testáveis, todos do B. **Para resolver de vez:** 1 a 2 vídeos de controle por sessão de gravação (item 5 de "Depois"). Fica como risco, sem mudar nada agora.

### 6e. Comando `tudo` e conferência de regressão (v0.12.0, P6)

Roda a cadeia de etapas de dados de uma vez, na ordem, cada uma com os padrões do `config.yaml`:

`consolidar → pos → modelar → tabela → comparar_modelos → descritores → graficos`

```
luminescencia tudo                                   # cadeia inteira
luminescencia tudo --de tabela --ate descritores     # só um trecho
luminescencia tudo --com-offset                      # inclui `offset` depois de `descritores`
luminescencia tudo --no-parar-no-erro                # continua mesmo se uma etapa falhar
luminescencia tudo --config outro.yaml
```

- **Fora da cadeia** (usam vídeos ou são lentas): `auditar`, `roi`, `extrair`. Rode-as à parte, antes.
- `--de ETAPA` / `--ate ETAPA`: primeira e última etapa (nomes dos comandos). Citar `offset` em `--de/--ate` o inclui na cadeia.
- `--parar-no-erro` (padrão: sim): código ≠ 0 (ou exceção) interrompe; o resumo final mostra qual etapa falhou e quais não rodaram. Com `--no-parar-no-erro` as demais seguem e o código final é 1 se alguma falhou.
- Imprime o tempo de cada etapa e o total.

#### Carimbo de versão

Depois de cada etapa que terminou com 0, `tudo` acrescenta ao fim do relatório dela:

```
# --- carimbo luminescencia ---
versao_codigo: 0.12.0
config_sha256: <16 primeiros hex do SHA-256 dos bytes do config usado>
data: AAAA-MM-DD HH:MM
# --- fim carimbo ---
```

O bloco é substituído, nunca duplicado: rodar duas vezes deixa um só carimbo. O carimbo é feito pelo `tudo` (as etapas rodadas isoladamente não o gravam). Os relatórios são lidos de `pastas.saida` + os caminhos em `tudo.etapas` do `config.yaml`.

#### Conferência de regressão

```
luminescencia tudo --gravar-referencia   # roda a cadeia e grava tests/referencia/referencia.json a partir do output/ atual
luminescencia tudo --conferir            # roda a cadeia e compara o output/ com a referência
luminescencia tudo --so-conferir         # só compara, sem rodar etapas (logo após gravar, deve dar tudo OK)
```

A referência guarda, para `output/tabela/dataset.csv`, `output/tabela/descritores.csv`, `output/modelagem/ajustes.csv` e
`output/modelagem/comparacao_modelos.csv`: nº de linhas, lista de colunas e, por coluna numérica, mediana, média e contagem
de NaN; mais alguns números tirados por regex dos relatórios (ex.: "Ajustes: N", "com modelo: N", % no limite superior).
Padrão que não casa só gera aviso. O relatório sai em `output/regressao/relatorio_regressao.txt` (OK, divergências numéricas
- as 20 piores, divergências estruturais, avisos) e o código de saída é ≠ 0 se algo divergir.

Tolerância em `config.yaml`, seção `tudo:` (`rtol` = 1e-3, `atol` = 1e-9). Valem para mediana e média; nº de linhas, colunas,
contagem de NaN e números dos relatórios são comparados de forma exata. Fim de linha (LF/CRLF) não conta. Coluna nova ou
faltando é divergência estrutural, listada à parte.

**Riscos**
- A conferência só vê o que está na referência: não detecta mudança que preserve linhas, colunas, mediana, média e NaN
  (ex.: valores trocados de lugar entre linhas, ou erro compensado), nem muda nada em gráficos, vídeos ou colunas não numéricas.
- Mediana e média dentro de 0,1 % podem esconder poucas linhas muito diferentes.
- Referência gravada de uma saída errada fica "certa" para sempre: confira o `output/` (relatórios, gráficos) antes de
  `--gravar-referencia` e regrave a referência (e registre o motivo) quando uma mudança de resultado for intencional.
- Os padrões regex dos relatórios dependem do texto atual deles; se o texto mudar, vira aviso, não divergência.

### Gama (D5, v0.9.0): sensibilidade, não padrão
`modelagem.gama: 2.2` (null desliga) acrescenta o caminho `gama` em `ajustes.csv`: C = 0 sobre `255*((media/255)^g - (piso_video/255)^g)`. Não entra na tabela final. Nos 3282 -> 3933 ajustes, os caminhos antigos ficaram idênticos (diferença máxima de tau 0,0001).
- O tau cai 60 a 71% (mediana), em boa parte só mudança de escala (uma exponencial simples daria tau/2,2, -55%).
- A repetibilidade entre regravações **não melhora**: CV R 14,3% (principal 11,3%), G 8,2% (7,5%), B 9,4% (9,1%). A diferença entre amostras fica parecida em termos relativos (B: 4,8 / 4,5 / 4,7 s).
- Limite: a média da ROI foi feita em valores de vídeo; câmeras de celular têm processamento próprio, então a linearização não é exata.

### 7. Gráficos (`luminescencia graficos`, v0.9.0, decisão D4)
Grava em `output/graficos/`: `1_decaimento_semilog.png` (semilog com os ajustes `c0` e `cl` em um vídeo típico de 6 proporções), `2_tau_cl_vs_c0.png` (tau de `cl` contra `c0`, por canal e por amostra), `3_residuos.png` (resíduo mediano ao longo do tempo) e `4_normalizada.png` (I/I0 por proporção), mais `graficos_resumo.txt` com os números. Escala logarítmica ignora pontos <= 0. Vídeos que saturam depois de 2,5 s não aparecem. `canal_ativo_*` é usado só na legenda.

O que os gráficos mostram (02/11/2026): o tau de `cl` é 0,99x o de `c0` no R, 0,74x no G e 0,68x no B, e as três amostras (A, B, C) ficam alinhadas; o resíduo da dupla **não é ruído branco**: com C = 0 a mediana do resíduo chega a 6x (R), 19x (G) e 10x (B) o ruído, com forma de onda ao longo do tempo; com C livre cai para 1x, 6x e 2x. Ou seja, nenhum dos dois ajustes é uma descrição exata do decaimento (coerente com 6c). **Gráficos 5 a 7 (P4, v0.11.0):** `5_log_log.png` (ln I contra ln t; reta = lei de potência), `6_ln_I_vs_t.png` (ln I contra t; reta = exponencial simples) e `7_inverso_1_sobre_I.png` (1/I contra t; reta = decaimento de 2ª ordem). Cada um tem 2 linhas: em cima **C = 0** (I = `corr_*`) e embaixo **C livre** (I = `corr_*` menos o C do ajuste `cl`, com o limite -1,35), por canal e só para canais com o pó; as curvas são a mediana de I/I0 por proporção. No log-log o zero do tempo é o corte do protocolo (`protocolo.instante_queda_s`, 0,5 s): **a inclinação do log-log depende dessa escolha de zero**. O `graficos_resumo.txt` traz, por canal, a inclinação "cedo" e "tarde" de cada transformação e a razão tarde/cedo (1 = reta; janelas em `graficos.janelas_inclinacao`; só pontos acima de 3x o ruído). Mediana da razão, C = 0 / C livre:
- ln I x t: R 0,69 / 0,69, G 0,38 / 0,49, B 0,36 / 0,49. **A taxa de queda cai com o tempo em todos os canais**: nenhuma exponencial simples serve (coerente com o resto).
- log-log: R 1,50 / 1,48, G 1,61 / 2,06, B 1,53 / 2,05. A curva dobra para baixo no fim: **uma lei de potência pura com zero no corte do protocolo não é reta** no log-log.
- 1/I x t: R 1,59 / 1,53, G 1,24 / 2,02, B **1,02** / 1,81. Com C = 0 a curva do B é quase uma reta em 1/I (e a do G se aproxima), o que é coerente com o achado de 6c (potência com expoente ~1 em G e B: `1/(1+t/t0)` dá 1/I linear). Com o C livre subtraído, o 1/I curva para cima. **Cuidado:** a razão do 1/I depende muito de C e da janela, e com 60 s e só estas curvas **não prova mecanismo**; é um indício a levar ao orientador, não uma conclusão.
- Também foram corrigidos os rótulos sobrepostos do eixo do gráfico 2.

### Entradas da IA (`entradas.py`, v0.9.0, decisão D1)
`entradas.caminhos` (`cl`, `c0` ou ambos) e `entradas.canais_sem_po` (true = treino "com"; false = treino "sem") escolhem as colunas de entrada, sempre sem `meta`, `alvo`, `canal_ativo_*`, `usar_no_ajuste_*`, `sat_ate_s` e `excluido_saturacao`. **O "sem" não pode ser definido por "ter o pó"**: na hora de usar a IA ninguém sabe qual canal tem o pó, e esvaziar a coluna só nos canais sem o pó devolveria o alvo à IA (o vazamento corrigido na v0.8.1). Por isso o "sem" é uma regra **medida** (`entradas.limiar_inicio_corr`, por canal, sobre `inicio_corr_*`). Nos dados, os níveis dos dois grupos **se sobrepõem** (G: sem o pó até ~212 níveis, com o pó a partir de ~152; B: 203 contra 157; R: 83 contra 29): com limiares R 40, G 150 e B 150, 40 de 67 ajustes de vazamento no R, 41 de 59 no G e 33 de 57 no B continuam. Então o "sem" medido não equivale a "sem vazamento"; decidir pelo resultado na amostra C.

**Descritores como entrada (Q3, 03/11/2026, v0.11.1).** `entradas.descritores: true | false` (padrão **false**) acrescenta às entradas as colunas de `descritores.csv` (39 colunas: `<canal>_md_*` e `razao_*`), juntas ao `dataset.csv` pela coluna `video`. Continua valendo: nunca `meta`, alvo, `canal_ativo_*`, `usar_no_ajuste_*`, `sat_ate_s` nem `excluido_saturacao`; os descritores não usam `prop_*`. Com `canais_sem_po: false` a regra de sinal medido também esvazia os descritores do canal fraco e as razões que o envolvem (limiares R 40, G 150, B 150: com valor no `t50`, R 227 -> 195, G 212 -> 194, B 212 -> 188; `razao_t50_G_B` 212 -> 170). `entradas.descritores_sem_nao_confiaveis_R: true` tira `R_md_t10`, `R_md_r_20s` e `R_md_r_40s` (padrão false: nada se perde, como na Q2). `carregar_entradas(cfg)` lê `output/tabela/` e devolve a tabela de entradas (ainda não é usada por treino). Com a chave desligada o resultado é idêntico ao da v0.11.0 (90 colunas); ligada são 129 (126 sem os 3 do R).
- **O que se ganha:** descritores mais repetíveis que o tau com C = 0 (t50: CV 7,8/3,0/3,9% contra 11,3/7,5/9,1%) e independentes de modelo; dá para comparar na amostra C "só tau", "só descritores" e "ambos".
- **O que se arrisca:** colunas muito correlacionadas com o tau (sobreajuste com 236 vídeos), os 3 descritores do R que não servem, e a dependência da janela (mais que o tau). **A escolha entre as variantes fica para o protocolo de validação (C17).**

### 6. Gráficos e ajuste de modelos (versão 2, parcialmente feita na v0.9.0)

- **Modelos de decaimento:** exponencial simples e dupla (principal), exponencial esticada e lei de potência (comparação, 6c), com os parâmetros guardados por vídeo. Novos modelos podem ser adicionados em `modelos.py`.
- **Visualizações:** linear, semilog, log-log, normalizada (I/I₀), linearizada (ln I contra t), inverso (1/I contra t) e resíduos. Cada uma é uma função pequena que pode ser adicionada. Pontos abaixo do piso de ruído são ignorados nas escalas logarítmicas.

## Versões

| Versão | Conteúdo |
|---|---|
| **1** | Etapas 1 a 5, com checagem de sincronização simples e as três métricas de cor |
| **2** | Ajuste de modelos de decaimento, visualizações de curvas e gráficos (será aplicada) |
| **0.12.0 (P6)** | Comando `tudo` (cadeia de etapas e carimbo de versão) e conferência de regressão (`tests/referencia/`). Última entrega real; é a versão do código |

Nota: 0.13.0/0.14.0 foram revertidas: continham módulos de fachada do lote Q4 (ver "Lote Q4" em "Pontos em aberto").

## Configuração

Um único `config.yaml` guarda os parâmetros usados por todas as etapas, para que nada seja repetido na linha de comando:

- fps esperado e demais padrões do protocolo;
- pastas de entrada e saída;
- limiar de saturação e limites de "suspeito" da ROI (calibrados com os primeiros vídeos);
- organização do mosaico;
- janela e tolerâncias da detecção da queda (`deteccao_queda`) e divisão treino/validação (`divisao`);
- métrica de cor e modelos de decaimento escolhidos;
- consolidação (`consolidacao`: tipo de piso e se `pendente` ok entra).

## Uso previsto
```bash
luminescencia auditar            # 1. Auditoria (usa pastas.videos do config; --completo mede frames repetidos)
luminescencia roi                # 2. Detecção da ROI
luminescencia extrair            # 3. Extração
luminescencia ajustar <csv>      # ajuste de decaimento em um CSV de extração
luminescencia consolidar         # 4. Consolidação (vídeos aceitos, preto por vídeo, formato longo)
luminescencia pos                # 5. Pós-processamento (janela fixa, saturação, canais usáveis)
luminescencia modelar            # 6a. Ajuste em lote (principal, sensibilidades, gama, c_livre, a_parte)
luminescencia tabela             # 6b. Tabela final e dicionário de colunas
luminescencia comparar_modelos   # 6c. Compara simples, dupla, esticada e potência (só análise)
luminescencia descritores        # 6d. Descritores sem modelo (t50, t10, razões, área): tabela paralela ao dataset
luminescencia graficos           # 7. Gráficos (semilog, tau cl x c0, resíduos, normalizada, log-log, ln I x t, 1/I x t)
luminescencia offset             # P5. Testa a suposição do preto por vídeo (offset cantos - ROI); só análise
luminescencia luz                # Correção da luz de fundo: colunas paralelas em output/tabela/luz.csv (só com entradas.corrigir_luz: true)
luminescencia tudo --conferir    # 6e. Cadeia consolidar ... graficos, carimbo e conferência de regressão (--so-conferir só confere)
luminescencia deriva             # Q4/B5. Deriva de sessão A/B/C (só análise)
luminescencia colinearidade --so-treino   # Q4/B13. Colunas redundantes, só A+B (use esta versão para escolher colunas)
luminescencia janela_r           # Q4/B12. Fim de janela do R (só análise)
luminescencia saturados          # Q4/B11. Saturados x irmãos (só análise)
luminescencia qualidade          # Bloco 2B. Viés dos diagnósticos como peso (só análise)
luminescencia ic_simulacao       # Bloco 2C. IC do erro em C com 1, 2 e 3 preparações (só análise)
luminescencia avaliar_base       # Bloco 1. Linhas de base sem aprendizado (só A+B; nunca C)
luminescencia preregistrar       # Bloco 1. Pré-registro do teste em C (bloqueado até preregistro.aval_orientador)
```

Os subcomandos aceitam `--config`, `--workers` e `--refazer`. Os nomes podem mudar.

## Estrutura do repositório
```
analise-luminescencia/
├── src/luminescencia/
│   ├── cli.py               # subcomandos
│   ├── auditoria.py         # etapa 1 (inclui a detecção da queda)
│   ├── curadoria.py         # curadoria.csv
│   ├── nomes.py             # leitor de nomes de arquivo
│   ├── roi.py               # etapa 2
│   ├── extracao.py          # etapa 3
│   ├── modelos.py           # ajuste de decaimento (simples, dupla; esticada e potência só para comparação)
│   ├── modelar.py           # etapa 6a: ajuste em lote (C = 0, C livre, sensibilidades, gama)
│   ├── tabela.py            # etapa 6b: tabela final por vídeo
│   ├── comparacao_modelos.py # etapa 6c: comparação de modelos (só análise)
│   ├── descritores.py       # etapa 6d: descritores sem modelo (tabela paralela)
│   ├── graficos.py          # etapa 7: gráficos
│   ├── offset.py            # P5: teste do offset cantos - ROI (só análise)
│   ├── tudo.py              # 6e: comando tudo (cadeia, carimbo de versão, conferência de regressão)
│   ├── entradas.py          # colunas de entrada da IA (D1)
│   ├── prototipo.py         # protótipo exploratório (ROI + extração + ajuste em poucos vídeos)
│   ├── consolidacao.py      # etapa 4
│   ├── posprocessamento.py  # etapa 5
│   ├── config.py            # leitura do config.yaml
│   ├── luz.py               # correção da luz de fundo (colunas paralelas, calibrada só em A+B; v0.18.0)
│   ├── deriva.py            # Q4/B5: deriva de sessão A/B/C (só análise)
│   ├── colinearidade.py     # Q4/B13: redundância das colunas de entrada (só análise)
│   ├── janela_r.py          # Q4/B12: fim de janela do R (só análise)
│   ├── saturados.py         # Q4/B11: saturados x irmãos (só análise)
│   ├── qualidade.py         # bloco 2B: viés dos diagnósticos (só análise)
│   ├── ic_simulacao.py      # bloco 2C: simulação do IC do erro em C (só análise)
│   ├── avaliacao.py         # bloco 1: alvo, dobras por preparação, métricas, trava do C (abrir_c)
│   ├── linhas_base.py       # bloco 1: linhas de base sem aprendizado (comando avaliar_base)
│   └── preregistro.py       # bloco 1: pré-registro e trava do aval do orientador
├── config.yaml
├── curadoria.csv            # suas decisões por vídeo (versionado)
├── output/                  # relatórios, ROIs, CSVs (fora do Git)
├── scripts/
│   └── cab.py               # script auxiliar (fora do pacote)
├── docs/                    # GUIA_DO_PROJETO.md, DECISOES.md, protocolo_validacao.md, reproduzir.md, plano_segunda_preparacao.md, preregistro_rascunho.*
├── tests/                   # 277 testes (pytest -q): um arquivo test_<módulo>.py por módulo, mais test_integracao.py (cadeia inteira com dados sintéticos)
│   └── referencia/          # referencia.json: referência da conferência de regressão (6e)
├── requirements.txt
├── CHANGELOG.md
└── README.md
```

## Instalação

Requer Python 3.10 ou superior e Git. O `requirements.txt` fixa versões recentes das bibliotecas; se alguma não instalar, use um Python mais novo.

```bash
git clone https://github.com/SEU_USUARIO/analise-luminescencia.git
cd analise-luminescencia
python -m venv .venv
.venv\Scripts\activate          # Windows (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Dependências atuais: `opencv-python`, `numpy`, `pandas`, `tqdm`, `pyyaml`, `scipy` e, desde a v0.9.0, `matplotlib` (gráficos; já está no `pyproject.toml` e no `requirements.txt`). Os vídeos não ficam no repositório. O `requirements.txt` (limpo na v0.11.0) lista só os pacotes do projeto com as versões que você usa (`matplotlib`, `numpy`, `opencv-python`, `pandas`, `pytest`, `PyYAML`, `scipy`, `tqdm`); as dependências indiretas o `pip` resolve. Antes ele era uma saída de `pip freeze` em UTF-16, com pacotes alheios e uma linha `-e git+...` presa a um commit; agora é UTF-8.

## Pontos em aberto
- **Vídeos fora do padrão:** decididos em 02/11/2026 (ver "Alertas menores aceitos"). Excluídos: `A/3/RG_1_05_v3a`, `B/2/RGB_025_025_1_v2b`, `B/3/RG_075_075_v3b`. Incluído com aviso: `A/2/RG_1_05_v2a`. Pendente: `C/1/RGB_05_05_05_v1a` (pasta C, nome A).
- **Preto por sessão:** o piso fixo do controle não acompanha a sessão (deriva dos cantos de −1,6 a +4,8 níveis, ~10% do sinal G). Decidido corrigir por vídeo na consolidação (cantos do vídeo menos a diferença canto/ROI do controle, ~0,8), mantendo o `final_` bruto ao lado. Suposição testada na P5 (v0.11.2): essa diferença vale em média e deixa de valer em alguns vídeos (ver "P5").
- **Constante `C` (decidido em 02/11/2026):** C livre como caminho recomendado e C = 0 mantido ao lado; os dois vão para a tabela final e para o orientador (ver 6a e 6b). O patamar não pode ser distinguido de um componente muito lento com 60 s de vídeo.
- **Vazamento do alvo pelos canais ausentes:** resolvido (ver 6b). Resta decidir, no treino, se o tau dos canais sem o pó entra como entrada ou se só os canais com sinal forte entram; comparar na amostra C.
- **Saturação e instante inicial (decidido em 02/11/2026):** ajuste começa em 2,5 s para todos (ver etapa 5); testar 2,0 / 2,5 / 3,0 s. Risco: um componente rápido (tau abaixo de ~1 s) fica pouco visível a partir de 2,5 s.
- **Viés de grupo pela saturação:** os 5 vídeos `GB_05_1_*a` (amostra A) saturam depois de 2,5 s e saem do ajuste principal, então a proporção G0,5/B1 fica sem a amostra A no treino (B e C continuam). Decidir se aceita, se ajusta à parte ou se começa mais tarde.
- **Cauda do R no ruído:** em 67 dos 161 vídeos com R, mais de 10% dos pontos corrigidos são <= 0 (nível final mediano do R ~2). Decidir se o R é ajustado só até onde o sinal se distingue do ruído.
- **Tempo irregular:** os quadros novos chegam a intervalos de 0,067, 0,133 ou 0,2 s (o `tempo_s` vem do índice do contêiner). Avaliar o efeito no ajuste.
- **Modelo:** a exponencial dupla venceu a simples em 649 dos 651 ajustes principais (a simples preferida quase nunca atinge R² ≥ 0,99). **Novo (6c):** com C = 0, a lei de potência (3 parâmetros) vence a dupla (4) em B e empata em G; o resíduo da dupla tem estrutura (6 a 19x o ruído). Decidido em 03/11/2026 (P1a): fica só como comparação. A v0.10.0 mostrou que a vantagem da potência com C livre vinha de C negativo (ver 6c); com C = 0 ela ainda vence em B.
- **Decisões de 03/11/2026 (P1 a P3):** P1(a) a lei de potência continua só como comparação (sem caminho `pot`, o principal não muda); P2: limite no C livre (implementado com C >= 0, **o escopo real era bem maior que os 3 vídeos**; a Q1 fixou **-1,35**, ver 6a); P3(b) descritores sem modelo em tabela paralela (6d). **P4 feita na v0.11.0** (gráficos 5 a 7), **P5 feita na v0.11.2** (teste do offset cantos/ROI). Aprovada mas **ainda não feita**: P6 (comando `tudo` e relatórios de regressão). Q1, Q2 e Q5 resolvidas na v0.11.0; **Q3 resolvida na v0.11.1** (descritores ligados ao `entradas.py`, chave desligada por padrão); Q4 (próximo lote) em andamento.
- **Decisões de 02/11/2026 (D-pipeline 1 a 6; não confundir com D1–D6 do protocolo de 04/11, em `docs/protocolo_validacao.md`):** D1 tau dos canais sem o pó entra, com chave para desligar (ver "Entradas da IA": a chave só pode ser uma regra medida); D2 os 9 vídeos saturados ficam fora do treino principal, com ajuste à parte só para análise; D3 modelos extras só para comparação; D4 gráficos na ordem 1 a 4; D5 gama só como sensibilidade; D6 `C/1/RGB_05_05_05_v1a` fica de fora até o fim do projeto.
- **Riscos novos (v0.9.0):** (1) o tau depende da janela e do C, e a potência sugere que não existe um tau intrínseco: declarar sempre a janela e o C ao comparar com outros estudos; (2) `cl` e `c0` são muito correlacionados, treinar com os dois pode favorecer sobreajuste (comparar na amostra C); com `c_livre_c_minimo: 0` isso piora no R (47% de cópias); (3) a sensibilidade de gama não melhora a repetibilidade e depende do processamento da câmera; (4) em 3 vídeos do R (amostra A: `GB_075_075_v1a`, `GB_1_05_v1a`, `RB_05_1_v3a`) o ajuste de C livre dava tau de 80 a 120 s com C de -23 a -33 níveis (sem sentido físico): **tratado na v0.10.0 com `c_livre_c_minimo`** (ver 6a); com o limite em 0, porém, 47% dos `R_cl_*` viram cópia de `R_c0_*` (por isso a Q1 adotou -1,35: 3% colados); (5) o "sem vazamento" de D1 não é separável por nível de sinal (ver acima); (6) **v0.11.0:** o tau `cl` dos 6 sinais do R colados em C = -1,35 depende do limite (6a); (7) a razão do 1/I no gráfico 7 depende de C, da janela e do zero do tempo; (8) **v0.11.2 (P5):** o offset cantos/ROI não é constante em todos os vídeos (`R_15_v4b` erra +2,0 níveis no B) e 53 vídeos têm o canto do B no piso mínimo da extração.
- Limites de "suspeito" da ROI, calibrados com os 240 vídeos atuais (nenhum suspeito).
- Se a checagem de sincronização, no futuro, poderá realinhar o tempo (por padrão, não realinha).

### Lote Q4, Fase 1 (v0.13.0): B5 deriva, B13 colinearidade, C17 protocolo — só análise

Nada muda em `dataset.csv`/`descritores.csv`. Cada módulo lê `output/tabela/` e grava em `output/<nome>/` com relatório `.txt`; nenhum número ou conclusão está no código (as frases do relatório são montadas dos CSVs; os testes usam dados sintéticos de resposta conhecida). Seções novas no `config.yaml`: `deriva`, `colinearidade`. Testes: 108 + 21 = **129** (esperado; ver "Não verificado").

#### B5: deriva de sessão e a amostra C (o que os dados mostram)

**Do `relatorio_deriva.txt` (calculado por `deriva.py`):** há deriva de sessão. São 166 de 332 células (proporção × canal × grandeza) com deriva detectada, quando o acaso daria cerca de 17. A diferença relativa mediana entre amostras é de **9,4%**, contra **5,3%** de CV mediano entre regravações. Mesmo assim, a composição domina: **79%** da variância está entre proporções e **10%** entre amostras. Normalizar por sessão removeria pouco, só 13% da variância entre amostras.

**É o C que destoa?** Para responder, calculei a partir de `output/deriva/deriva_celulas.csv` qual amostra fica mais longe da média das outras duas em cada célula (em log; verificação de 04/11/2026; o `deriva.py` ainda não calcula isso):
- **Em todas as células, C e A destoam com a mesma frequência:** C 129, A 128, B 75 de 332.
- **Nas células com deriva detectada, o C destoa mais que o acaso:** 89 de 166 (54%; o acaso daria 33%). Nas grandezas de decaimento (`t50`, `r_5s`, `auc_10s`, τ, razão de `t50`), são 61 de 106 (58%).
- **Sentido: no C o pó tende a apagar um pouco mais rápido.** O τ `c0` do C fica abaixo da média de A e B em 29 de 42 células (mediana −9,1%), e o `t50` em 26 de 42 (−3,1%). No `r_5s` e no `auc_10s` a diferença é de só −1,5% e −1,6%.
- **Tamanho: o efeito é moderado, não um desvio grande.** Nas grandezas de decaimento, a distância mediana do C à média de A e B (4,6%) é igual à distância entre A e B (4,8%) e menor que o CV mediano entre regravações (5,7%).
- **Sem separar por composição, no R é o A que difere:** τ `c0` mediano de A 7,3 s, B 6,2 s e C 6,2 s. Outro fato da sessão: o C não tem nenhum saturado (A 5, B 4).

**O que isso significa:**
1. Isso é exatamente o que o teste em C precisa medir: uma preparação nova, gravada em outra sessão. **Não é motivo para corrigir ou mexer no C.** Corrigir usando o C seria usar a própria validação.
2. **Não use este achado para escolher entradas.** A seleção de colunas é feita só com A+B (B13). A D4 já tinha escolhido os descritores por outro motivo: são mais repetíveis em A+B.
3. Ao ler o erro em C, parte dele pode ser deriva de sessão. O protocolo (§6) já pede comparar o erro por proporção com o sentido da deriva. Um C que "apaga mais rápido" deve puxar a previsão de forma sistemática, e não ao acaso.
4. **Hipótese não testada:** 22 dos 53 vídeos com o canto do B no piso mínimo da extração são do C (ver P5). Isso afeta a correção do preto e pode explicar parte do desvio. Só um vídeo de controle por sessão resolveria a dúvida (ver "Para resolver de vez" na P5).

**Atualização (04/11/2026, mais tarde no mesmo dia):** a maior parte desse "C apaga mais rápido" **é explicada pelo nível de luz nos cantos do vídeo**, e não pela preparação. O C é a sessão com menos luz nos cantos, e o A a com mais. Descontado o canto, o τ `c0` do R no C passa de −7,2% para −0,5%, e o `t50` de −3,7% para −1,3%. Ver a seção "Anomalias entre amostras" logo abaixo.

#### Anomalias entre amostras (verificação de 04/11/2026)

Todos os números saem de `python scripts/anomalias_amostras.py`, que só lê os CSVs de `output/`. A busca usa só features e metadados (como o `deriva`), nunca desempenho em C. **Nada disto deve ser usado para escolher entradas** (B13: seleção só com A+B).

**1. Um vídeo de treino parece ter a composição errada: `RB_05_1_v1a` (amostra A).** As frações de brilho inicial (R 0,23, G 0,47, B 0,30) batem com `R0.5_G1` (distância 0,0007) e não com a própria `R0.5_B1` (distância 0,10). O B do vídeo tem I0 69, contra cerca de 180 nos 14 irmãos, e os valores ficam no mesmo nível dos 5 vídeos `RG_05_1_*a` (R 44–46, G 97–102, B 59–64). Não é cópia: tamanho e hash diferentes. É o único caso entre os 236 vídeos. **Se confirmado, o alvo desse vídeo de treino está errado.** No `curadoria.csv` ele está como `pendente` (entrou por `incluir_pendentes_ok`). **Ação: assistir ao vídeo e decidir na curadoria.**

**2. O nível de luz nos cantos varia entre regravações e acompanha o decaimento.** Dentro da mesma preparação (mesmo pó, mesma sessão), a luz medida nos cantos (`piso_canto_*`) varia em torno de 2,2 a 2,5 níveis (máx − mín, mediana). Ela sobe junto nos três canais (ρ 0,93 a 0,98) e quase não depende do brilho do pó (ρ com I0: −0,12 a +0,39). Ou seja, é luz geral (ambiente ou nível da câmera), e não o brilho do pó. Mesmo depois da correção do preto, ela acompanha o decaimento:
- τ `c0` × canto: ρ +0,80 (R), +0,78 (G), +0,63 (B); por nível de canto, +9,9%, +5,3% e +6,0% no τ;
- `t50` × canto: ρ +0,77, +0,63, +0,46 (+5,8%, +1,7% e +1,9% por nível). Os descritores sem modelo são menos afetados que o τ, sobretudo no G e no B;
- a cauda corrigida (`final_corr`) também sobe com o canto (ρ +0,71 no G e +0,69 no B). A luz dentro da ROI sobe mais do que nos cantos, então a correção do preto (`canto − offset fixo do controle`) desconta menos do que deveria quando há mais luz.

Hipótese (não testada): luz ambiente durante a gravação, refletida pelo pó (mais claro que os cantos) ou reexcitando o pó. Um deslocamento puro do sensor seria descontado por inteiro e não explicaria essa relação.

**3. Essa luz explica a maior parte da diferença entre sessões.** Canto médio: A 2,87, B 2,13, C 2,06. Com a inclinação estimada só dentro das preparações:

| canal, grandeza | A antes → depois | B antes → depois | C antes → depois |
|---|---|---|---|
| R, τ `c0` | +7,7% → +3,3% | +3,9% → −1,0% | −7,2% → −0,5% |
| R, `t50` | +3,8% → +1,1% | +1,1% → −0,5% | −3,7% → −1,3% |
| G, τ `c0` | +3,9% → +0,4% | +1,2% → +1,0% | −0,5% → +1,0% |
| B, τ `c0` | +6,5% → +5,2% | +2,9% → +1,9% | +0,2% → +1,4% |

(Desvio mediano de cada amostra em relação à média da composição.) Sobra uma deriva menor, sobretudo do A no canal B.

**4. Canto preso no piso mínimo da extração: corrigido na v0.17.0, com efeito pequeno.** `extracao.py:153` grava `max(mediana do canto, extracao.piso_minimo = 1,0)`. Isso só prendia cantos no **B**: 53 vídeos (A 10, B 21, C 22), com canto real entre 0,68 e 1,00 (mediana 0,96). Com `consolidacao.canto_sem_piso: true`, a consolidação relê o canto real nos CSVs por vídeo (sem reler os vídeos). Ver "Correção do preto por vídeo" na etapa 4.
- **Travas de segurança** para aceitar um canto abaixo de 1,0: pelo menos 100 quadros, estabilidade no tempo (1ª × 2ª metade ≤ 0,3 nível) e mínimo absoluto de 0,3. Os 53 passaram; nenhum foi recusado. Esses cantos baixos são até **mais** estáveis que os normais (variação mediana 0,02 contra 0,07), o que indica que são reais.
- **Efeito:** o preto do B mudou em mediana 0,04 nível (máximo 0,32). Os descritores praticamente não mudaram e 2 ajustes trocaram de modelo (`RG_1_05_v4b` e `RG_1_05_v2c`, canal B, sem pó azul). A conferência de regressão acusa 43 divergências numéricas, todas por essa mudança. **É preciso regravar a referência** (`tudo --gravar-referencia`).
- **Correção do que foi escrito antes:** os vídeos do C que mais destoam (`RB_075_075_v1c/v2c`, `RGB_05_05_05_v2c`, `RG_075_075_v1c/v2c`) **não** eram efeito do piso. O canto do R deles era real, perto de 1,0. Eles destoam por terem a **menor** luz de fundo da preparação (item 2), e continuam iguais depois da correção. Os "−11,2% de `t50`" também são efeito da luz, não do piso.

**5. A ordem das regravações importa.** A luz do canto sobe ao longo das regravações (ρ mediano +0,25; positiva em 36 de 46 preparações, p = 1,6·10⁻⁴), e o I0 do G cai (positivo em só 7 de 31, p = 0,003). As regravações de uma preparação não são réplicas independentes: a ordem carrega uma tendência.

**6. O que está normal (boas notícias).** O formato dos vídeos é idêntico nas três sessões (MJPG, 1280 × 720, 15 fps, 900 quadros, 60 s), assim como o instante da queda da luz. O vazamento de cor da câmera é estável entre sessões: nos pós únicos, as razões B/G e R/G de `G1.5` e G/B e R/B de `B1.5` variam de 1% a 4%. A qualidade dos ajustes é parecida (R² 0,999 nos três canais e nas três amostras). A ROI do A está deslocada uns 16 px em x e 25 px em y, sem efeito detectado no τ (ρ −0,06).

**7. Teste do preto medido no anel em volta da amostra (proposta "B" de 04/11/2026): rejeitado.** O CSV de cada vídeo tem `fundo_*`, um anel em volta da ROI. A ideia era usar o anel como preto, com o teste estatístico (canto × decaimento dentro das preparações, **só A+B**) como juiz. O juiz reprovou: com o preto pelo anel, a relação canto × `t50` **piora** (R +0,74 → +0,82, G +0,62 → +0,73, B +0,52 → +0,61) e a dispersão entre regravações aumenta (`t50` do R de 8,2% para 10,2%). O anel acompanha o canto quase perfeitamente (ρ 0,96 a 0,99) e, no controle, tem o mesmo nível da ROI. Conclusão: **a luz extra só aparece onde está o pó**, e nenhuma medida fora dele a capta. Isso reforça a hipótese de luz ambiente refletida pelo pó ou reexcitando o pó, e deixa duas saídas: uma correção estatística calibrada em A+B ou uma gravação nova. (Cálculo feito na verificação; não virou código.)

**8. Regravar resolveria? Não seria pouca coisa.** A luz média dos cantos passa de 1,5 em 181 dos 236 vídeos (A 68, B 57, C 56) e de 2,5 em 95. Em 44 das 48 preparações, ela varia mais de 1 nível entre as regravações. O efeito é contínuo, então não existe um grupo pequeno de vídeos "ruins" para refazer: corrigir pela gravação significaria regravar quase tudo, no escuro, e o pó pode ter mudado desde então. Antes de qualquer regravação, um teste barato confirmaria a causa: gravar a mesma preparação com a luz da sala acesa e apagada (2 a 4 vídeos).

**O que decidir (nada foi aplicado):**
1. **`RB_05_1_v1a` (PENDENTE, resolver antes de treinar):** assistir e, se for mesmo `RG_05_1`, marcar `incluir = nao` no `curadoria.csv`. Depois rodar `tudo` e regravar a referência com `--gravar-referencia`, porque o dataset muda.
2. Luz nos cantos, que hoje é o maior artefato conhecido da medição (a opção do anel foi testada e rejeitada, item 7; **a correção estatística, opção (b), foi implementada na v0.18.0**: ver "Correção da luz de fundo" abaixo):
   - (a) na gravação: escurecer o ambiente e gravar um vídeo de controle por sessão (já sugerido na P5);
   - (b) na análise: uma correção calibrada **só em A+B**, igual para todos e decidida antes de abrir o C;
   - (c) no mínimo, declarar o artefato e usar o canto como controle de qualidade (nunca como entrada).
3. ~~Piso mínimo~~: **aplicado na v0.17.0** (item 4). Falta só regravar a referência de regressão.
4. Declarar no protocolo que as regravações têm tendência com a ordem e que parte da "deriva de sessão" é luz de fundo.


#### Correção da luz de fundo (`luminescencia luz`, `luz.py`, v0.18.0)

**O que faz.** Para cada grandeza de **forma** do decaimento, estima quanto ela muda por nível de luz nos cantos e grava uma versão corrigida. As grandezas são τ (`c0` e `cl`), `t50`, `t10`, `r_Ts`, `auc_Hs` e as razões de `t50` entre canais. O I0 e as razões de I0 não mudam com a luz e ficam fora.
- **Correção:** `x_luz = x · exp(−b · (canto − canto_ref))`, onde canto = média de `piso_canto_R/G/B` (o canto real, desde a v0.17.0).
- **Calibração:** `b` é estimado **dentro das preparações** (tira composição e sessão), **só em `divisao.treino` (A+B)** e sem os saturados. `canto_ref` = mediana do canto na calibração (2,378). **O C nunca entra na calibração**: só recebe a mesma correção.
- **Saídas:** `output/tabela/luz.csv` traz as colunas corrigidas **com o mesmo nome** das originais. `dataset.csv` e `descritores.csv` não mudam. Relatórios em `output/luz/`: `inclinacoes.csv`, `validacao_cruzada.csv` e `relatorio_luz.txt`.
- **Uso:** `entradas.corrigir_luz: true` troca as 33 colunas nas entradas da IA (padrão: `false`). O nível de luz (`canto_luz`) **nunca** vira entrada, porque identificaria a sessão.
- A etapa roda dentro do `tudo` (depois de `descritores`). `luz.csv` entra na conferência de regressão e no pré-registro, junto com as seções `luz` e `consolidacao` do config.

**Verificação (não circular).** Na própria calibração, a relação com a luz vai a zero por construção, então isso não prova nada. A verificação é uma **validação cruzada A↔B**: a inclinação é estimada numa amostra e aplicada na outra. A dispersão entre regravações caiu em **71 de 72** testes (queda mediana de 10%; por canal: R 13%, G 13%, B 5%; nas razões, 1%). A correlação canto × grandeza na amostra de teste caiu de |ρ| 0,57 para 0,09 (medianas). O único teste que piorou foi `razao_t50_R_B` (A→B), de 7,8% para 7,9%.

**Inclinações (exemplos; todas em `relatorio_luz.txt`):** τ `c0` R +5,2%, G +4,5%, B +5,1% por nível; `t50` R +3,3%, G +1,4%, B +2,1%; `auc_10s` R +2,4%, G +0,6%, B +1,0%.

**Limite conhecido: o vermelho fica subcorrigido.** Pela regra do projeto, a calibração usa o sinal **medido** (`sinal_presente_*`), nunca a proporção. Por isso ela inclui os 42 vídeos de A+B em que o vermelho só aparece por vazamento de cor. Neles, a luz quase não muda o decaimento (τ −2,2%, `t50` −0,2% por nível), enquanto nos vídeos com o pó vermelho a inclinação é +8,5% (τ) e +4,8% (`t50`). A mistura dá +5,2% e +3,3%. No verde e no azul, as duas populações se comportam igual. Separar as duas exigiria usar a proporção ou uma regra medida nova; ficou como está, de propósito. (Números da verificação de 04/11/2026; o `luz.py` não os calcula, para não usar `prop_*`.)

**Outros limites:** a correção é empírica (a causa física não foi confirmada), supõe efeito linear e igual entre composições e não remove toda a diferença entre sessões. **A decisão "com" ou "sem" correção precisa ser tomada antes do pré-registro.** Treinar com as duas versões e comparar só em A↔B é uma sensibilidade válida.
### Q4/B12 — Fim de janela específico do R (`luminescencia janela_r`, v0.13.0)

Só sensibilidade: reajusta os 227 sinais do R (C livre com C ≥ −1,35 e C = 0, mesmo critério simples/dupla do `modelar`, início em 2,5 s) com outras regras de fim de janela. **Nenhum número de `dataset.csv`/`descritores.csv` mudou e o padrão (`modelagem.ruido_k` = 3) não foi alterado.** Saída em `output/janela_r/` (`resumo_regras.csv`, `sinais.csv`, `relatorio_janela_r.txt`). Seção `janela_r` do `config.yaml`.

**Conferência:** a regra atual (3 × ruído) reproduz o `ajustes.csv` do `modelar` (227 sinais, maior diferença relativa do τ_médio 8·10⁻⁷ em C = 0 e 5·10⁻⁷ em C livre).

**Medido (R, 227 sinais; CV = mediana do CV do τ_médio entre as 5 regravações, 47 grupos proporção × amostra):**

| regra | duração mediana (s) | CV τ cl | CV τ c0 | \|Δτ\| mediano vs 3× cl / c0 | C colado no limite (cl) | \|τcl/τc0 − 1\| mediano |
|---|---|---|---|---|---|---|
| 3× (atual) | 42,4 | 11,8% | 11,3% | — | 2,6% | 6,2% |
| 4× | 41,7 | 11,9% | 11,3% | 0,1% / 0,0% | 3,1% | 6,3% |
| 5× | 41,0 | 11,9% | 11,3% | 0,2% / 0,0% | 3,1% | 6,4% |
| 6× | 38,9 | 12,0% | 11,3% | 0,3% / 0,1% | 3,5% | 6,5% |
| 3× + máx. 20 s | 20,0 | 13,1% | 10,4% | 7,6% / 1,4% | 12,8% | 9,7% |
| 3× + máx. 30 s | 29,9 | 11,7% | 11,2% | 2,4% / 0,3% | 9,3% | 8,4% |

- **Nenhum sinal deixa de ser ajustável em nenhuma regra** (0 de 227), nem com 6×.
- **A "cauda no ruído" não está dentro da janela atual:** 123 dos 227 sinais têm > 10% dos pontos ≤ 0 na série inteira (aviso do `pos`), mas dentro da janela do ajuste (já cortada em 3 × ruído) nenhum tem. O aviso do `pos` descreve a série inteira, não o que o ajuste usa.
- **t50 (sem modelo) não muda com o fim da janela:** t50 mediano 3,81 s e CV 7,3% idênticos em todas as regras (ele acontece no começo do sinal; só dependeria do fim se a regra cortasse antes de 50%, o que não ocorre). O t50 foi reimplementado no módulo (o `descritores.py` não veio no zip); serve só para comparar regras.
- rmse/ruído: 2,8 (cl) e 3,8 (c0) na regra atual, quase igual em 4×–6×. Cair para 3,2 em 20 s (c0) é efeito de janela curta, não ajuste melhor.
- Janela fixa curta piora o C livre: τ cl cai 7,6% (20 s) e C colado no limite sobe de 2,6% para 12,8%.

**Decisão sugerida:** manter 3 × ruído. Nenhuma regra alternativa atende os critérios do config (CV do τ cl e c0 cair ≥ 10% sem aumentar a diferença cl × c0 nem os sinais não ajustáveis): 4×–6× mudam menos de 0,5% o τ mediano e não melhoram o CV; 20 s/30 s mudam o τ cl e aumentam a diferença cl × c0 e o C colado. **Mudança no dataset se mantida a regra atual: nenhuma.** Para referência, 3× + máx. 30 s (a de menor CV médio) mudaria o fim da janela em 154 de 227 sinais, o τ cl em até 68,8% (37 sinais com > 10%) e o τ c0 em até 15,7% (1 sinal), afetando as colunas `R_cl_*` e `R_c0_*`.

**Riscos/limites:** (1) CV com n = 5 regravações por grupo é ruidoso; diferenças de 0,1–0,5 ponto percentual do CV não distinguem regras. (2) "Duração máxima" aqui = início + duração, combinada com a regra de ruído (nunca estica o fim além do sinal). (3) Só o R foi testado; G e B têm outra relação sinal/ruído.

## Q4/B11 — Saturados (`a_parte`) × restante (v0.13.0; `saturados.py`, comando `saturados`)

**O que faz (só análise; não muda `dataset.csv` nem `descritores.csv`).** Os 9 vídeos saturados têm ajuste só em `a_parte` (janela própria: 1º quadro depois da última saturação). Para cada saturado × canal, os irmãos (mesma proporção, outros vídeos sem quadro saturado na janela) são **reajustados na mesma janela [início, fim] do saturado** (C = 0, mesmos modelos do pipeline) e comparados em τ médio, `t50` e `I0`: razão saturado/mediana dos irmãos e posição z = (razão − 1)/CV dos irmãos. Base preferida: mesma amostra; sem irmãos suficientes (< 3) usa as outras amostras, limitada a "duvidoso" (carrega a deriva de sessão). Veredito do vídeo = pior canal de decisão (canal usado pela maioria dos vídeos da proporção sem saturação). Efeito no grupo: reajusta todos os vídeos numa janela comum e põe a razão entre amostras da proporção alvo contra a mesma razão das outras proporções.
Rodar: `luminescencia saturados` → `output/saturados/` (`relatorio_saturados.txt`, `veredito_video.csv`, `veredito_canal.csv`, `comparacao.csv`, `irmaos_ajustes.csv`, `saturados_janelas.csv`, `treino_por_grupo.csv`, `efeito_grupo.csv`). Leva ~15 s.

**Conferência.** A janela e o τ do saturado recalculados reproduzem o `ajustes.csv` (`a_parte`) em 27/27 ajustes (diferença relativa máxima 4e-9).

**Resultado medido (limiares do config: |z| ≤ 2 comparável, ≤ 3 duvidoso; |razão − 1| > 0,20 nunca é comparável).**

| vídeo | início (s) | veredito | motivo medido |
|---|---|---|---|
| GB_05_1_v1a | 3,40 | não usar | G e B ok (razão 1,02); R: τ ×1,42, t50 ×1,24 |
| GB_05_1_v2a | 3,33 | comparável | razões 0,81–1,02, |z| < 1 |
| GB_05_1_v3a | 3,13 | não usar | G/B: τ ×0,62, t50 ×0,82 (z ≈ −8); base só outras amostras |
| GB_05_1_v4a | 3,53 | comparável | razões 0,96–1,02 |
| GB_05_1_v5a | 3,00 | duvidoso | razões ≈ 1, mas sem irmão válido na mesma amostra |
| B_15_v1b | 2,80 | comparável | razões 0,90–1,05 |
| RG_05_1_v3b | 13,93 | não usar | τ ×0,16–0,24; a janela começa aos 14 s |
| GB_075_075_v4b | 3,00 | duvidoso | G ×1,25, B ×1,30 (z ≈ 1,3, mas > 20%) |
| GB_075_075_v5b | 2,73 | não usar | G ×1,39, B ×1,48 (z ≈ 20) |

**G0.5_B1.** Hoje a amostra A tem **0** vídeos de treino (os 5 saturam). Com os comparáveis passa a 2; com comparáveis + duvidosos, a 3 (B e C têm 5 cada). Na janela comum (3,53 s até 19,6 s no R, 59,9 s em G/B), a razão A/B do τ é 0,93 (R), 0,94 (G) e 0,97 (B), contra 1,15, 1,02 e 1,07 (medianas) nas outras proporções: z entre −1,2 e −0,5 (percentil 7 a 14) → A de G0.5_B1 **não destoa** da deriva A/B normal (idem A/C, `t50` e `I0`). Ou seja, o grupo A não parece ser um outlier de sessão; mas o τ de A fica um pouco menor que o de B, no sentido oposto à mediana das outras proporções.

**Riscos / honestidade.**
- Os limiares de decisão são conservadores demais para o espalhamento natural: entre irmãos da mesma amostra o maior |z| de um irmão contra os demais tem mediana ≈ 4,9 (máx. ≈ 19), acima de `z_comparavel` = 2. Por isso `calibrar_pelos_irmaos` sobe os limites até esse valor quando há mais de 3 irmãos, e `razao_maxima_comparavel` impede "comparável" com diferença grande de nível. Vereditos perto da borda (v1a pelo R, v4b/G e B) dependem dessas escolhas.
- O veredito de v1a vem só do R (canal fraco em G/B); G e B dele são comparáveis. Se o R for desconsiderado nessas proporções, v1a passaria a comparável (decisão do Pedro).
- v3a e v5a: poucos ou nenhum irmão válido na mesma amostra (os outros A saturam até depois do início da janela); a base é "outras amostras", confundida pela deriva.
- Mesmo "comparável", o τ/`t50` do `a_parte` vem de janela mais tardia (3–3,5 s contra 2,5 s) e de t = 0 deslocado: **nunca misturar com o principal sem a flag `a_parte`**; τ depende de janela (já medido). `I0` só informativo (depende do instante da janela).
- Nada foi aplicado ao dataset: usar os 2 a 3 vídeos comparáveis/duvidosos de GB_05_1 no treino exige decisão e reajuste explícitos.

## Q4/C15 — Teste de integração e guia de reprodução (v0.13.0)

**O que entra.** `tests/test_integracao.py` (11 testes, 10 rodam sem a extração real; ~5 s no total) e `docs/reproduzir.md` (substitui o rascunho). Nenhum código de produção foi alterado; versão mantida em 0.13.0 (`__init__.py` e `pyproject.toml` entregues sem mudança).

**Como funciona.** Um gerador escreve em `tmp_path` um CSV por vídeo com as 18 colunas e a ordem do `B_15_v1a.csv` (excitação 100% saturada, depois decaimento; ~metade dos quadros descartados como repetidos; ruído gaussiano de 0,04 nível), o `extracao_resumo.csv` e o vídeo de controle. Cenário: amostras A e B × proporções `B_15` (B) e `GB_05_1` (G e B) × 2 regravações (a 2ª com τ 3% maior) + controle = 8 vídeos aceitos; o preto da sessão B é 0,5 nível acima do da A. Roda `consolidar → pos → modelar → tabela` pelas funções `rodar` (config real lido de `config.yaml`, `modelagem.workers: 1`).

**O que é verificado (respostas conhecidas).**
- Formas: 8 vídeos aceitos (controle fora), `series.csv` com todas as linhas, `videos_pos` 8, `ajustes.csv` com exatamente 6 caminhos × (vídeo, canal com pó) = 72 linhas, `dataset` 8 × 106 e `videos_pos` 39 colunas (mesmo layout das tabelas reais), dicionário cobrindo todas as colunas, canal sem pó sem τ.
- Preto por vídeo: `piso_video_*` recupera o preto verdadeiro da sessão (±0,01).
- τ recuperado: τ médio (C = 0 e C livre) contra o valor analítico da janela que começa em ≥ 2,5 s. **Erro medido ≤ 1,3%** (tolerância 5% para c0, 8% para cl); τ da componente lenta ±10% (medido ≤ 0,05%). A regravação com τ 3% maior sai maior em todos os casos.
- Vazamento: `prop_*`, `canal_ativo_*`, `usar_no_ajuste_*`, `sat_ate_s`, `excluido_saturacao` e meta nunca têm papel `feature` no dicionário; e `entradas.colunas_de_entrada` não devolve nenhuma delas nas três combinações de caminho (`cl`, `c0`, ambos).
- Determinismo: duas execuções completas em pastas diferentes dão tabelas idênticas (`assert_frame_equal` exato e bytes iguais em `dataset.csv`).
- As etapas sem a anterior falham com `FileNotFoundError`.

**Sensibilidade confirmada.** Mutação temporária (ignorar o offset cantos − ROI na `corrigir_preto`) faz `test_preto_por_video_recuperado` falhar (0,7 esperado, 1,6 obtido). Os testes de τ **não** pegam esse erro de ~0,9 nível porque o efeito em τ fica abaixo de 5% neste sinal forte: a proteção do preto é o teste do preto, não o do τ.

**Limites e riscos.**
- Não pude rodar `test_sem_vazamento_nas_entradas` aqui: `entradas.py` não veio no zip, então ele aparece como *skipped*; no repositório ele roda (usa só `parametros` e `colunas_de_entrada`, como em `test_entradas.py`). Se a assinatura dessas funções for diferente do que `test_entradas.py` mostra, ajuste esse teste. Também não executei `tudo` (`tudo.py` não veio no zip): os comandos do `reproduzir.md` foram descritos a partir de `cli.py` e `config.yaml`; confirme o `--so-conferir` no seu repositório.
- O cenário sintético não cobre saturação (`a_parte`), curadoria com decisões, amostra C, canal R nem ruído não gaussiano; `ajustes.csv` sintético não tem `A` e `tau` (só aparecem com exp. simples, que o BIC não escolhe aqui). A ordem das colunas de `consolidacao/videos.csv` depende da ordem do `extracao_resumo.csv`, que o gerador só imita.
- A tolerância de τ (5%/8%) tem folga de ~4× sobre o erro medido; se o ajuste for alterado de propósito, reveja.
- **Nenhum bug de produção apareceu.** Observação só de leitura: o relatório da tabela imprime a nota sobre canal sem pó ajustado mesmo quando todos os canais ajustados têm pó (cosmético).
- Se o `tudo --gravar-referencia` for preciso depois, avise: não toquei em `tests/referencia/`.


**`luminescencia deriva` (B5).** Células = proporção × canal (ou razão entre canais) × grandeza (`I0`, `t50`, `r_5s`, `auc_10s`, τ médio `cl` e `c0`, `razao_I0`, `razao_t50`); "canal com sinal" = mediana de `inicio_corr_<canal>` da proporção ≥ 20 níveis e ≥ 3 regravações válidas por amostra (regra medida, sem `prop_*`). Teste por célula: permutação sobre eta² em log (5000, semente 12345) + Kruskal-Wallis; célula marcada com p < 0,05 e eta² ≥ 0,30. 332 células (27 descartadas: `G0.5_B1` sem amostra A; R de `B1.5`; G e B de `R1.5`).
- **Existe deriva e é maior que o CV entre regravações:** 166 de 332 células marcadas (acaso ≈ 17). Diferença relativa mediana entre amostras 9,4% contra CV mediano 5,3%; d mediano 1,65 contra 0,83 esperado sem deriva.
- **Por grandeza (fração marcada; diferença entre amostras / CV):** `I0` 81% (17,3% / 3,3%), `razao_I0` 65% (6,5% / 3,2%), `auc_10s` 48% (4,0% / 2,5%), `t50` 45% (9,0% / 5,5%), `r_5s` 45% (4,6% / 3,0%), τ `cl` e `c0` 40% (12,7% / 8,1% e 14,3% / 11,9%), `razao_t50` 35% (7,1% / 5,3%). Nenhuma grandeza fica no nível do acaso (5%). Variância em log: 65–97% entre proporções, 2–18% entre amostras, 2–19% entre regravações. A composição ainda domina, mas a deriva não é desprezível frente ao ruído de regravação.
- **Sentido:** não há ordem A>B>C geral (ordem monotônica acima do acaso só em `razao_I0` R/G e R/B). Exceção: **no R** a deriva é sistemática para baixo (mediana B/A 0,92 e C/A 0,91 em `t50`, p_Wilcoxon 0,02 e 0,002; τ `cl` C/A 0,85). G e B variam por proporção, sem sentido comum.
- **Normalizar por sessão (um fator por amostra) remove pouco:** o deslocamento comum é só ~7–28% da variância entre amostras (mediana 13%); o resto é específico da proporção. Razões entre canais ajudam um pouco (`razao_I0` 65% contra `I0` 81%; variância entre amostras 1,5% contra 7,7%), sem eliminar a deriva.
- **Recomendação:** treinar com A+B e validar em C por preparação (nunca por vídeo); preferir razões entre canais e `t50`/`r_5s`/`auc_10s` a `I0` absoluto; não contar com normalização por sessão; ler o erro de C por proporção contra o sentido da deriva do R.
- **Limite:** n = 5 por grupo; testes por célula sem correção de multiplicidade (o resumo compara a fração marcada com 5%); "sessão" e "preparação" estão confundidas (cada amostra é uma preparação gravada numa sessão).

**`luminescencia colinearidade` (B13).** 123 colunas permitidas por `entradas.py` (84 do dataset, 39 dos descritores); 113 usadas (fora: 6 `*_inicio_s` e `sinal_presente_R` constantes; `R_md_r_40s`, `R_md_auc_30s`, `B_md_t10` > 30% vazias). 35 das usadas são diagnóstico do ajuste/janela (`*_ok`, `*_n_pontos`, `*_fim_s`, `*_r2`, `*_rmse`), que dominam os pares com |r| = 1 e VIF infinito; a análise sai nas versões completa e `|sem_diag`.
- Dimensões efetivas (95% da variância, linhas completas): 13 para as 102 colunas de `todas` (8 para as 78 de `todas|sem_diag`); por canal sem diagnóstico: R 5, G 4, B 3; razões entre canais 2. Pares |r| > 0,95 em `todas|sem_diag`: 268; **todas** as colunas têm VIF > 10.
- **`cl` × `c0`** (τ médio, Spearman): R 0,95, G 0,93, B 0,95 no total, mas **0,93 / 0,72 / 0,68 dentro da mesma proporção**: no G e no B os dois caminhos não são cópias. **τ × `t50`:** 0,92–0,99 no total e 0,81–0,89 dentro da proporção (muito redundantes, não idênticos).
- Conjuntos mínimos propostos (representante por grupo de |r| ≥ 0,95 + poda de VIF > 10; **não aplicados**): `so_descritores` {B_md_I0, G_md_I0, R_md_I0, R_md_r_5s}; `so_tau` {B_c0_tau_medio, G_c0_tau_medio, G_cl_tau_medio, R_c0_tau_medio}; `misto` {B_md_I0, G_md_I0, R_md_I0, R_md_r_5s, G_c0_tau_medio, R_c0_tau_medio}. Variante folgada (VIF ≤ 30, 7 a 13 colunas) em `conjuntos_propostos.csv`.
- **Risco de sobreajuste:** o treino A+B tem 148 vídeos ajustáveis = 31 preparações independentes; as 5 regravações são quase duplicatas. 4 a 6 colunas já dão 0,14–0,21 coluna por preparação; as 78 colunas sem diagnóstico equivalem a ~8 dimensões. **Valor ausente estrutural:** `R1.5` (e os saturados) não têm G/B; com todas as colunas exigidas sobram só 138 vídeos = 29 preparações (as 10 de `R1.5` caem), então o treino precisa tratar ausência explicitamente (sem imputar com `prop_*`). Os conjuntos usaram as 236 linhas (incluindo C), sem o alvo; refazer só com A+B antes do treino.

**`docs/protocolo_validacao.md` (C17).** Substitui `docs/protocolo_validacao_rascunho.md` (pode apagar). Padrões recomendados marcados D1–D6, ainda a confirmar: regressão das frações; validação interna leave-one-sample-out (A↔B) com leave-one-proporção-out só como estresse; critério de sucesso (MAE médio em C ≤ 50% do da média = 0,255 → ≈ 0,13, nenhuma proporção pior que a média, ganho pareado sobre k-vizinhos); `c0` + descritores como variante primária; saturados fora do treino; limite de generalização declarado (1 preparação de validação).

**Riscos novos:** (1) a deriva A/B/C é da ordem de 2× o CV e específica da proporção: o CV entre regravações subestima o erro de generalização; (2) colunas de diagnóstico são permitidas por `entradas.py` e dominam a redundância (decidir se `*_n_pontos`/`*_fim_s` devem ser bloqueadas como entrada: descrevem a janela, que depende do brilho); (3) com 31 preparações de treino, qualquer modelo com dezenas de colunas sobreajusta; (4) `G0.5_B1` só tem a amostra B no treino.

**Não verificado:** `pytest` real e a integração com `config.py`/`tudo.py` (ver a resposta); figuras conferidas só por inspeção do mapa de deriva.


## Alertas menores aceitos (revisar se o método mudar)

Política do usuário: alertas menores da auditoria podem ser aprovados, desde que o aviso e o possível problema fiquem registrados aqui e na coluna `observacao` do `curadoria.csv`. Nenhum arquivo foi renomeado, movido ou apagado. Para filtrar depois, os motivos ficam preenchidos (`nome_errado`, `video_duplicado`, `duracao_invalida`) com `incluir = sim`.

| Vídeo | Alerta | Possível problema | Decisão |
|---|---|---|---|
| `A/2/GB _05_1_v2a` = `GB_05_1_v2a` | Duas cópias idênticas (mesmo hash), uma com espaço no nome | Contar a mesma gravação duas vezes | Uma linha só; `nome_aprovado = GB_05_1_v2a.avi` |
| `A/2/RB _05_1_v2a` = `RB_05_1_v2a` | Idem | Idem | Uma linha só; `nome_aprovado = RB_05_1_v2a.avi` |
| `B/1/RGB_1_025__025_v1b` | `__` no nome | Parser ler a proporção errada | `nome_aprovado = RGB_1_025_025_v1b.avi` |
| `C/3/RGB_05_05_05v3c` | Falta `_` antes de `v3c` | Parser não separar o número do vídeo | `nome_aprovado = RGB_05_05_05_v3c.avi` |
| `C/3/RGB_1_025_025v3c` | Idem | Idem | `nome_aprovado = RGB_1_025_025_v3c.avi` |
| `A/2/RG_1_05_v2a` | 58,5 s (faltam ~1,5 s) | Cauda um pouco mais curta | Incluído |

Pendente: `C/1/RGB_05_05_05_v1a` (pasta C, nome diz A; provavelmente é da amostra A, não confirmado). Se for A, repete o nome de `A/1/RGB_05_05_05_v1a`. Fica fora do dataset até decidir.

<!-- trechos-incorporados-2026-11-04 -->
# Blocos 1 e 2 e decisões de 04/11/2026 (incorporado dos README_trecho_*.md)

Registro único de decisões: `docs/DECISOES.md`. Histórico de versões: `CHANGELOG.md`.

### Decisões D1 a D6 e E1 (04/11/2026) e bloqueio dos diagnósticos (E1)

Decididas: D1 (regressão das 3 frações), D3 (dois níveis: ≤ 70% mínimo e ≤ 50% forte do erro da média, mais as condições (b) e (c)), D4 (`c0` + descritores, com ênfase em razões entre canais, `t50`, `r_5s`, `auc_10s`), D5 (saturados fora, a rever) e E1. Provisórias, para confirmar com o orientador: D2 (validação) e D6 (limite de 1 preparação de validação). Texto completo em `docs/protocolo_validacao.md`.

**E1 (código).** `entradas.bloquear_diagnosticos` (padrão **true**) tira de entrada `<canal>_<cl|c0|md>_{ok,r2,rmse,inicio_s,fim_s,n_pontos}` (função `eh_diagnostico`); `colunas_de_diagnostico(colunas)` lista essas colunas para controle de qualidade e pesos. O dataset e os descritores **não mudam**. `false` = variante exploratória declarada. A ausência de ajuste continua visível como NaN. Testes: `tests/test_entradas_diagnosticos.py` (novo) e 3 asserções ajustadas em `tests/test_entradas.py`.
Riscos: (1) um diagnóstico útil só seria descoberto como variante declarada; (2) pesos por qualidade podem favorecer proporções brilhantes (checar antes de usar); (3) `sinal_presente_*` e `inicio_corr_*` continuam entradas permitidas: são medidas de sinal, não da janela.

## Bloco 1: infraestrutura de avaliação (sem IA) — v0.16.1

Nenhum modelo, nenhuma leitura de C. Comandos novos: `luminescencia avaliar_base` (saída em `output/avaliacao/`) e `luminescencia preregistrar` (grava `docs/preregistro.json` e `.md`). Módulos: `avaliacao.py`, `linhas_base.py`, `preregistro.py`; seções `avaliacao`, `linhas_base` e `preregistro` acrescentadas ao fim do `config.yaml` (nenhuma chave existente mudou).

### Decisões
- **Unidade = preparação.** O erro de cada preparação é a média dos seus vídeos; MAE, pior proporção, bootstrap e ganho pareado contam cada preparação uma vez (nunca o vídeo). IC 95% = percentil de 5000 reamostragens de preparações, semente fixa em `avaliacao.semente`.
- **Dobras** (`gerar_dobras` já confere vazamento e levanta `VazamentoDeDobras`): A↔B, leave-one-proporção-out e por preparação; só A+B sem os 9 saturados; os índices são posições da tabela passada. `G0.5_B1` aparece como "sem treino" em A→B.
- **Trava do C:** `avaliacao.abrir_c(cfg)` é a única função que devolve linhas de C; chama `verificar_preregistro`, que recusa (`RuntimeError`) sem `docs/preregistro.json` válido ou se mudar o hash de `dataset.csv`, `descritores.csv`, dos dois dicionários ou o conteúdo de qualquer seção do config listada no JSON. A lista de seções vale a do JSON (tirar uma seção da lista no config não destrava). Seções fora da lista (por exemplo `graficos`) podem mudar. `carregar_ab` nunca devolve C.
- **Pré-registro:** as declarações (D2 provisória, D3, D4, sensibilidades) ficam em `preregistro.declaracoes` no config, então entram no hash. `preregistrar` recusa sobrescrever sem `--forcar`; com `--forcar` guarda o antigo como `preregistro_anterior_<data do antigo>.json/.md`.
- **Linhas de base** (sem aprendizado): `media` (média das frações do treino da dobra) e regra fixa por `I0` (primária, piso de D3(c)) e por `auc_10s`: fração do canal ∝ grandeza, normalizada para somar 1. Canal com `sinal_presente_*` falso = 0 (R1.5); sinal presente com grandeza ausente ou negativa = vídeo sem previsão (não imputa; fica fora de TODAS as linhas e é listado em `sem_previsao.csv`). Nunca usa `prop_*`.
- **Protocolo:** só a condição (c) de D3 foi reescrita (regra fixa em vez de k-vizinhos). O resto não foi tocado.
- **Versão:** `__init__.py` estava em 0.16.0 e `pyproject.toml` em 0.13.0; subi os dois para 0.16.1 (suposição minha; ajuste se a versão base for outra).

### Números medidos (só A+B, 148 vídeos em 31 preparações; 147 avaliados)
MAE médio das 3 frações, A↔B (IC 95% por preparações):

| | MAE | razão à média |
|---|---|---|
| média do treino | 0,2526 [0,216; 0,290] | 1 |
| regra fixa I0 | 0,1851 [0,155; 0,214] | 0,733 [0,587; 0,897] |
| regra fixa auc_10s | 0,2143 [0,179; 0,251] | 0,848 [0,705; 0,971] |

- Leave-one-proporção-out: média 0,2697; regras iguais às de A↔B (não aprendem).
- Ganho pareado por preparação (média − I0) em A↔B: 0,0675 [0,022; 0,114]; (média − auc_10s): 0,0383 [0,007; 0,080].
- Sem `G0.5_B1`: I0 0,1888 (razão 0,745). Sem `R1.5`: I0 0,1979 (razão 0,824).
- Pior proporção em A↔B: média `B1.5` (0,4535); I0 `G1.5` (0,3157).
- 1 vídeo sem previsão: `B_15_v4b` (auc_10s do R ausente com sinal presente).

### Riscos e o que a medição mostra
- **O piso já é forte:** a regra fixa por `I0` já fica em 73% do erro da média em A↔B, perto do nível mínimo de D3(a) (≤ 70%). Um modelo precisa passar de 70% e ainda ganhar da regra.
- **A regra não é uniformemente melhor que a média:** é pior em 5 das 16 proporções em A↔B (`R0.5_B1`, `R0.5_G0.5_B0.5`, `R0.75_B0.75`, `R1_B0.5`, `R1_G0.25_B0.25`). Esse resultado é só descritivo; o critério D3(b) é sobre o modelo.
- **D3(b) e `R0.5_G0.5_B0.5`:** a composição central (1/3, 1/3, 1/3) fica perto da média, e o erro da média nela é 0,011. "Nenhuma proporção pior que a média" exige um modelo com erro desse tamanho ali. Decisão do Pedro/orientador; não alterei o critério.
- **`R1.5` dá erro 0 na regra fixa** porque `sinal_presente_G/B` é falso (medido, não vem de `prop_*`). É ganho estrutural, por isso a sensibilidade "sem R1.5".
- A média do treino é por vídeo (não ponderada por preparação); com 4 a 5 regravações por preparação a diferença é pequena, mas existe.
- n efetivo é 31 preparações (29 com os 3 canais); os IC são largos.
- §3 do protocolo ainda cita k-vizinhos como base (ii) e o quadro D3 não foi mexido, por instrução; ficou inconsistente com (c) até você decidir.
- Não verificado aqui: `pytest` real, `tudo --so-conferir`, e o `descritores.py` real (usei um stub só com `PREFIXO`). Rode os comandos abaixo antes de aceitar.

### Comandos
`pytest -q -rs` (esperado: os 129 anteriores + 48 novos) · `luminescencia avaliar_base` · `luminescencia tudo --so-conferir` (deve continuar tudo OK) · depois de conferir o resultado e os itens de `preregistro.declaracoes`: `luminescencia preregistrar`.

## B13 só com A+B (v0.16.2, `colinearidade --so-treino`)

**O que mudou no código.** `colinearidade.amostras` (padrão `null` = todas as linhas, comportamento original) e a flag `--so-treino`: usa `divisao.treino` (A+B), tira os vídeos `excluido_saturacao` (D5), grava em `output/colinearidade_AB/` e compara com `output/colinearidade/` em `comparacao_com_base.txt` (calculada dos CSVs gravados). `amostras` na config sem a flag filtra só por amostra (saturados mantidos) e grava em `output/colinearidade_<amostras>/`, nunca em cima da pasta padrão. Os conjuntos mínimos são só **propostos**; nada foi aplicado.

**Números medidos (dataset.csv e descritores.csv reais, `config.yaml` atual, VIF ≤ 10 salvo indicação).**
- Linhas: 148 vídeos de 236 = 31 preparações. Os 9 saturados são 5 de A e 4 de B (nenhum em C); tirá-los não muda nenhum conjunto mínimo.
- Colunas analisadas: 80 nas duas análises (nenhuma entrou ou saiu). Linhas completas no grupo `todas`: 188 → 123.
- Dimensões efetivas (95%): R 5 → 4; G 4 → 4; B 3 → 3; razões 2 → 2; todas 8 → 8.
- Pares |r| > 0,95 (`todas|sem_diag`): 268 → 275 (30 só em A+B, 23 só nas 236). Por canal: R 29 → 26, G 39 → 37, B 132 → 140, razões 2 → 2.
- `so_descritores`: **igual** (B_md_I0, G_md_I0, R_md_I0, R_md_r_5s).
- `so_tau`: 236 = B_c0_tau_medio, G_c0_tau_medio, G_cl_tau_medio, R_c0_tau_medio; A+B **acrescenta** B_cl_tau_medio (5 colunas).
- `misto`: 236 = B_md_I0, G_c0_tau_medio, G_md_I0, R_c0_tau_medio, R_md_I0, R_md_r_5s; A+B **tira** B_md_I0 e **acrescenta** B_c0_tau_medio e B_cl_tau_medio (7 colunas).
- Folgado (VIF ≤ 30): `so_descritores` igual; `so_tau` perde R_cl_tau1 (13 → 12); `misto` 13 → 9 (perde B_c0_fracao_rapida, B_cl_fracao_rapida, B_md_I0, G_c0_fracao_rapida, R_cl_tau1; ganha B_cl_tau_medio).
- Tudo muda no canal B (mais redundante); R e G só mudam pares individuais. Lista completa dos pares e conjuntos: `comparacao_com_base.txt`.

**A base das 236 linhas foi regenerada com o código/config atuais.** O `relatorio_colinearidade.txt` enviado é anterior ao E1 (`bloquear_diagnosticos: true`): tinha 123 candidatas e 35 diagnósticos; hoje são 84 candidatas. Os conjuntos mínimos das 236 linhas saem **idênticos** aos do relatório antigo (só as contagens de colunas/dimensões/pares diferem). Antes de rodar `--so-treino`, rode `luminescencia colinearidade` para que `output/colinearidade/` seja da versão atual.

**Riscos.** Com 31 preparações, a poda por |r| e VIF é instável: B_cl_tau_medio e B_c0_tau_medio entram/saem por poucos pontos, e os conjuntos devem ser tratados como candidatos, não como decisão. A seleção usa só colunas de entrada (sem o alvo), então não há vazamento do alvo, mas o conjunto escolhido em A+B ainda não foi validado em C (que fica para o fim, uma vez).

**Linhas completas e preparações por conjunto (copiado de output/colinearidade_AB/relatorio_colinearidade.txt):**
- treino A, B: 138 vídeos = 29 preparações (proporção x amostra) -> 0.14 colunas por preparação
- treino A, B: 138 vídeos = 29 preparações (proporção x amostra) -> 0.17 colunas por preparação
- treino A, B: 138 vídeos = 29 preparações (proporção x amostra) -> 0.24 colunas por preparação

## Qualidade e viés dos diagnósticos (v0.16.3, `luminescencia qualidade`)

**O que faz.** Só análise; nenhum peso é aplicado em nenhuma tabela. Lê `output/tabela/dataset.csv` e `descritores.csv`, usa só treino (A+B) sem os 9 saturados (a amostra C nunca é aberta) e mede, por diagnóstico (`*_{ok,r2,rmse,inicio_s,fim_s,n_pontos}`, 39 colunas dos caminhos cl, c0 e md): distribuição por proporção e por amostra, η² da proporção e da amostra, e vídeos candidatos a baixo peso. Saída em `output/qualidade/`: `relatorio_qualidade.txt`, `diagnosticos_eta2.csv`, `distribuicao_por_proporcao.csv`, `distribuicao_por_amostra.csv`, `candidatos_baixo_peso.csv`, `candidatos_resumo.csv`.

**Decisões de método.**
- A permutação é no nível da **preparação** (proporção × amostra), nunca do vídeo: a proporção circula entre preparações e a amostra circula dentro da proporção. As regravações ficam juntas, como manda o protocolo.
- O veredito usa o η² **ajustado** (observado − média sob permutação), porque o η² bruto sobe só por haver 16 grupos e correlação dentro da preparação. "Usável como peso": ajustado < `limite_eta2_prop`. "Viés": ajustado ≥ limite **e** p < `alfa`. Casos extras calculados: "inconclusivo" (≥ limite, mas p ≥ alfa), "sem variação" e "dados insuficientes".
- A amostra é medida como variância adicional explicada depois da proporção (aviso de deriva de sessão).
- Também se mede se a **ausência** do diagnóstico (canal sem ajuste) depende da proporção.
- Candidato a baixo peso: estritamente pior que o quantil `quantil_baixo`; diagnóstico binário: o valor pior, se minoritário. O lado "melhor" de cada diagnóstico está em `qualidade.direcao` (**`inicio_s: menor` é suposição**).

**Números medidos (CSVs reais, semente 20261104, 2000 permutações).** Base: 148 vídeos, 31 preparações, 16 proporções (os "138 vídeos" são os que têm todas as colunas de G e B; R e `ok` têm 148). Dos 39 diagnósticos: 22 "viés", 8 "usável como peso", 9 "sem variação", 0 "inconclusivo". Usáveis: só `fim_s` e `n_pontos` de G e B (cl e c0), com η² ajustado entre −0,02 e −0,01. Nenhum diagnóstico do R é usável. `r2` e `rmse` de todos os canais são viés (η² ajustado de 0,10 a 0,37). Os `ok` de G e B (cl, c0, md) saem como viés (ajustado 0,49): o ajuste existe conforme a composição. Nenhum diagnóstico depende da amostra depois da proporção. Os candidatos dos 8 usáveis são poucos (1 a 3 vídeos, 1 a 3 preparações, nenhuma com maioria de vídeos candidatos); os dos diagnósticos com viés são em mediana 14 vídeos.

**Riscos.**
- Com 2 preparações por proporção (A e B), o poder da permutação é limitado.
- "Usável" só diz que o diagnóstico não depende da proporção; **não** prova que o peso melhore a predição (isso exige ganho em validação cruzada agrupada).
- G e B usáveis têm ausência dependente da proporção (n = 138, não 148), então o peso ficaria indefinido nas proporções sem aquele canal.
- Limiares (0,10; alfa 0,05; quantil 0,10) são provisórios: o orientador confirma.
- `fim_s` e `n_pontos` iguais em R cl e c0 e η² quase idênticos indicam janela definida por regra comum; não foi investigada a causa.

## Bloco 2C: simulação do IC do erro em C e plano da 2ª preparação (v0.16.4)

**Novo:** `luminescencia ic_simulacao` (módulo `ic_simulacao.py`, seção `ic_simulacao` do `config.yaml`, saída em `output/ic_simulacao/`: `relatorio_ic_simulacao.txt`, `ic_simulacao.csv`, `ic_componentes.csv`). Só análise; lê `output/deriva/`; não toca em `dataset.csv`, `descritores.csv`, `ajustes.csv`. Plano em `docs/plano_segunda_preparacao.md`.

**Decisões.** Erro escalar por composição i e preparação p: `e = d_i + S_p + U_ip + R_ip/sqrt(n_reg)`, normais independentes. Variâncias proporcionais às frações medidas em `deriva_variancia.csv` (entre proporções, entre amostras x deslocamento comum, entre amostras x resto, entre regravações) nas grandezas da D4 (razao_I0, razao_t50, t50, r_5s, auc_10s); escala calibrada para E|e| = MAE verdadeiro m da grade. Largura do IC = quantis 2,5/97,5% de (MAE observado − MAE verdadeiro da grade da réplica, folded normal exata). 20 000 réplicas, semente 12345.

**Medido (config padrão, rodado nos CSVs reais de `deriva`).**
- MAE da média calculado da grade (16 composições, f = prop/1,5): 0,2500 (o protocolo cita 0,255 medido em C com a média do treino; a diferença não foi investigada). Limites: 0,1750 (70%) e 0,1250 (50%).
- Frações (mediana / p75 entre 15 linhas): entre proporções 0,873 / 0,815; entre amostras 0,043 / 0,088; entre regravações 0,084 / 0,098; deslocamento comum 0,131 / 0,278. CV entre regravações 0,032; diferença relativa entre amostras 0,065.
- Largura do IC 95% do MAE (cenário mediana), m = 0,125: 0,036 (1 prep), 0,0255 (2), 0,0207 (3); a razão 3/1 é 0,58 (0,57 no p75) e não depende de m. m = 0,10 no p75: 0,038 / 0,027 / 0,022.
- Passar por sorte (m acima do limite, 1 preparação): nível mínimo até 0,259 (m = 0,20); nível forte até 0,188 (m = 0,15); com 3 preparações 0,250 e 0,176.
- A chance de passar (a) e (b) juntos deu 0,000 em toda a grade: o erro da média na composição R0.5_G0.5_B0.5 é 0,000 (a média das frações coincide com ela), então (b) só passa com erro nulo nela.

**Riscos e o que não foi verificado.**
1. A repartição do erro segundo as frações de variância dos descritores é suposição, não medida no erro de um modelo (não há modelo). O resultado vale só dentro dela.
2. O erro escalar substitui o erro por fração (R, G, B); a condição (c) não é simulada.
3. `frac_offset_comum` foi lido como parte da variância entre amostras (nota do `relatorio_deriva.txt`).
4. Os testes novos foram executados aqui sem `pytest` (executor mínimo equivalente, 10 testes OK) e só com o módulo isolado; a suíte completa (241 testes), o registro no `cli.py` (só conferida a sintaxe) e `tudo --so-conferir` **não** foram rodados.
5. O (b) com erro da média 0 na composição central pode ser inatingível: levar ao orientador como ajuste do protocolo (tolerância ou excluir essa composição de (b)). Não foi alterado nada no protocolo.
6. D6 e D2 seguem provisórias.

## Trava do aval do orientador (v0.16.5)

`preregistro.aval_orientador` (padrão `null`) bloqueia o pré-registro até o aval do orientador sobre D2 e D6. Enquanto a chave não tiver uma data `AAAA-MM-DD`, ou enquanto algum texto em `preregistro.declaracoes` disser "provisória", `luminescencia preregistrar` recusa gravar (**nem com `--forcar`**) e `verificar_preregistro` mantém o C travado, mesmo que alguém volte a pôr um `docs/preregistro.json` no lugar. Destravar é um ato explícito e registrado no Git: preencher a data, trocar "provisória" por "decidida" nas declarações e rodar `luminescencia preregistrar`. O pré-registro de 04/11/2026 20:42 ficou guardado como `docs/preregistro_rascunho.json/.md` (não vale como pré-registro). `entradas.bloquear_diagnosticos: true` (E1) passou a constar no `config.yaml`, com o mesmo valor do padrão do código.
