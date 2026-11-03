# Análise de Luminescência por Vídeo

> **Projeto em desenvolvimento (v0.11.2).** Estão implementadas as etapas **1 (auditoria), 2 (ROI), 3 (extração), 4 (consolidação), 5 (pós-processamento), 6a (modelagem), 6b (tabela final)**, a **comparação de modelos** (6c), os **descritores sem modelo** (6d), os **gráficos** (7, agora com log-log, ln I × t e 1/I × t) e a escolha de **entradas da IA** (`entradas.py`). Ainda não existem: treino da IA e app. O que está marcado como "previsto" abaixo é plano e pode mudar.

## Objetivo

Transformar vídeos de amostras luminescentes em **dados numéricos limpos e comparáveis**: curvas de brilho e de cor ao longo do tempo, durante a excitação e o decaimento da luz.

Os dados serão usados para **treinar um modelo de IA**. Por isso a prioridade do projeto não é só extrair números, e sim garantir que **todas as amostras sejam medidas do mesmo jeito**. Qualquer diferença sistemática entre grupos de vídeos (outro fps, outra exposição, outra distância da câmera) pode ser aprendida pelo modelo no lugar da diferença real entre as amostras.

**O que são as amostras:** pós luminescentes em proporções diferentes de três componentes, identificados por amostra (A/B/C) e número do vídeo (1 a 5). O protocolo grava a fase de excitação e depois o decaimento (luz de excitação apagada logo no início, entre 0,2 e 0,4 s de vídeo), em arquivos de 15 fps (cerca de 8 quadros novos por segundo reais, pois ~47% dos quadros se repetem).

## Visão geral

O pipeline é dividido em seis programas independentes. Cada um lê a saída do anterior e grava seus resultados em arquivos. Entre as etapas há pausas: você lê o que foi gerado e decide se avança ou se volta para refazer só aquela etapa, sem repetir as outras.

```
[1 Auditoria] → [2 ROI] → [3 Extração] → [4 Consolidação] → [5 Pós-processamento] → [6 Gráficos]
```

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
│   ├── entradas.py          # colunas de entrada da IA (D1)
│   ├── prototipo.py         # protótipo exploratório (ROI + extração + ajuste em poucos vídeos)
│   ├── consolidacao.py      # etapa 4
│   └── posprocessamento.py  # etapa 5
├── config.yaml
├── curadoria.csv            # suas decisões por vídeo (versionado)
├── output/                  # relatórios, ROIs, CSVs (fora do Git)
├── tests/                   # test_nomes, test_curadoria, test_consolidacao, test_posprocessamento, test_modelos, test_modelar, test_tabela, test_modelos_extra, test_comparacao_modelos, test_gama, test_entradas, test_graficos, test_c_minimo, test_descritores, test_graficos_transformacoes, test_offset
├── requirements.txt
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
- **Decisões de 02/11/2026 (D1 a D6):** D1 tau dos canais sem o pó entra, com chave para desligar (ver "Entradas da IA": a chave só pode ser uma regra medida); D2 os 9 vídeos saturados ficam fora do treino principal, com ajuste à parte só para análise; D3 modelos extras só para comparação; D4 gráficos na ordem 1 a 4; D5 gama só como sensibilidade; D6 `C/1/RGB_05_05_05_v1a` fica de fora até o fim do projeto.
- **Riscos novos (v0.9.0):** (1) o tau depende da janela e do C, e a potência sugere que não existe um tau intrínseco: declarar sempre a janela e o C ao comparar com outros estudos; (2) `cl` e `c0` são muito correlacionados, treinar com os dois pode favorecer sobreajuste (comparar na amostra C); com `c_livre_c_minimo: 0` isso piora no R (47% de cópias); (3) a sensibilidade de gama não melhora a repetibilidade e depende do processamento da câmera; (4) em 3 vídeos do R (amostra A: `GB_075_075_v1a`, `GB_1_05_v1a`, `RB_05_1_v3a`) o ajuste de C livre dava tau de 80 a 120 s com C de -23 a -33 níveis (sem sentido físico): **tratado na v0.10.0 com `c_livre_c_minimo`** (ver 6a); com o limite em 0, porém, 47% dos `R_cl_*` viram cópia de `R_c0_*` (por isso a Q1 adotou -1,35: 3% colados); (5) o "sem vazamento" de D1 não é separável por nível de sinal (ver acima); (6) **v0.11.0:** o tau `cl` dos 6 sinais do R colados em C = -1,35 depende do limite (6a); (7) a razão do 1/I no gráfico 7 depende de C, da janela e do zero do tempo; (8) **v0.11.2 (P5):** o offset cantos/ROI não é constante em todos os vídeos (`R_15_v4b` erra +2,0 níveis no B) e 53 vídeos têm o canto do B no piso mínimo da extração.
- Limites de "suspeito" da ROI, calibrados com os 240 vídeos atuais (nenhum suspeito).
- Se a checagem de sincronização, no futuro, poderá realinhar o tempo (por padrão, não realinha).

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
