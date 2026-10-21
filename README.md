# Análise de Luminescência por Vídeo

> **Projeto em desenvolvimento.** Este documento descreve o que o projeto pretende ser e pode mudar a qualquer momento. Decisões marcadas como "a definir" ainda estão abertas, e os comandos e nomes de colunas abaixo são o plano, não necessariamente o que já existe no código.

## Objetivo

Transformar vídeos de amostras luminescentes em **dados numéricos limpos e comparáveis**: curvas de brilho e de cor ao longo do tempo, durante a excitação e o decaimento da luz.

Os dados serão usados para **treinar um modelo de IA**. Por isso a prioridade do projeto não é só extrair números, e sim garantir que **todas as amostras sejam medidas do mesmo jeito**. Qualquer diferença sistemática entre grupos de vídeos (outro fps, outra exposição, outra distância da câmera) pode ser aprendida pelo modelo no lugar da diferença real entre as amostras.

**O que são as amostras:** pós luminescentes em proporções diferentes de três componentes, identificados por amostra (A/B/C) e número do vídeo (1 a 5). O protocolo grava a fase de excitação e depois o decaimento (luz apagada perto dos 20 s), a cerca de 11 fps.

## Visão geral

O pipeline é dividido em cinco programas independentes. Cada um lê a saída do anterior e grava seus resultados em arquivos. Entre as etapas há pausas: você lê o que foi gerado e decide se avança ou se volta para refazer só aquela etapa, sem repetir as outras.

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
**Entrada:** pasta de vídeos. **Saída:** relatório em arquivo e resumo no terminal.

Roda antes de tudo e pode ser repetida sempre que chegarem vídeos novos. Mostra primeiro os vídeos que nem puderam ser lidos (corrompidos ou truncados), e no final destaca os vídeos fora do padrão.

- Leitura do nome do arquivo (proporção, amostra A/B/C, número do vídeo).
- Metadados: fps real, número de frames, resolução, duração, codec, frames duplicados ou perdidos (jitter) e fps variável.
- Comparação com o **padrão do protocolo** (por exemplo 11 fps), definido no arquivo de configuração. A **moda** entre os vídeos funciona como alerta secundário, para detectar lotes inteiros fora do padrão.
- **Exposição automática:** quando disponíveis nos metadados, ISO, tempo de exposição e balanço de branco são conferidos. Câmeras de celular podem compensar o brilho quando a luz apaga, o que deforma a curva de decaimento. Se o vídeo não trouxer esses campos, o relatório informa "não disponível".

### 2. Detecção da amostra (ROI)
**Entrada:** vídeos. **Saída:** arquivo de configuração (JSON/CSV) com a ROI de cada vídeo e uma imagem de verificação por vídeo.

- Detecção automática do círculo da amostra em um frame com luz (nunca no decaimento, quando a amostra está escura).
- O usuário escolhe no início como conferir:
  1. **Confiar 100%** no sistema;
  2. **Ver um por um** (Enter confirma, Espaço define manualmente);
  3. **Ver só os suspeitos.**
- Um vídeo é **suspeito** por regras objetivas: círculo não encontrado, raio fora do esperado, detecção instável entre frames, círculo encostado na borda ou contraste baixo. Cada suspeito mostra o motivo.
- **Modo manual:** clicar no centro e definir o raio, para os casos em que a detecção falha.
- **Mosaico:** folhas com as imagens de verificação de vários vídeos de uma vez. A primeira página reúne os suspeitos; as seguintes agrupam por lote, com algo entre 24 e 30 vídeos por folha. Agrupamento, ordenação, tamanho da folha e amostragem são configuráveis, e o mosaico pode ser regerado sem redetectar.
- **Movimento da amostra** (opcional): compara a posição no começo e no fim do vídeo. Só gera aviso.
- O arquivo guarda a `forma` da ROI. Hoje só o círculo está previsto (centro, raio e margem interna), mas outras formas podem ser adicionadas depois sem invalidar arquivos antigos.

### 3. Extração dos dados (o núcleo)
**Entrada:** vídeos, arquivo de ROI e configuração. **Saída:** um CSV por vídeo.

Por frame:

- `frame`, `tempo_s` (do timestamp do container) e `fase` (excitação ou decaimento, pelo tempo do protocolo);
- média de R, G e B, média simples e luminância;
- mediana, desvio padrão, mínimo, máximo e % de pixels saturados;
- média de uma **região de fundo** fora da ROI (3 colunas), que mostra se a câmera está levantando o brilho no escuro.

Cuidados: ordem BGR/RGB, barra de progresso, tratamento de erro por vídeo e reprocessamento fácil.

**Vídeo de controle:** um vídeo de amostra sem luminescência, nas mesmas condições, define o piso real de ruído e reflexo da câmera. É processado do mesmo jeito, marcado como controle, e o valor de base por canal e por sessão vai para o arquivo auxiliar.

### 4. Consolidação
**Entrada:** todos os CSVs individuais. **Saída:** um CSV "longo" único e um arquivo auxiliar.

- Junta tudo e acrescenta colunas lidas do nome do arquivo: proporção dos 3 pós, amostra (A/B/C) e número do vídeo.
- O arquivo auxiliar registra raio, fps, versão do código, data de processamento e valores do vídeo de controle.
- **Modo incremental**, escolhido no início:
  - *substituir existentes:* refaz tudo, inclusive vídeos já presentes;
  - *só adicionar novos:* pula o que já está consolidado.
- A verificação vai além do nome: compara tamanho, data (ou hash), ROI usada e versão do código, e reprocessa se algo mudou.
- **Avisos de duplicata:** mesmo conteúdo com nomes diferentes, e mesmo nome com conteúdo diferente.

### 5. Pós-processamento e análise
**Entrada:** CSV consolidado. **Saída:** CSV com colunas derivadas e tabelas de métricas.

Tudo é opcional, configurável e reversível:

- **Linearização de gama:** vídeos guardam o brilho em curva (gama ~2,2), não proporcional à luz. Como o decaimento físico é descrito em termos de luz, converter para valores lineares evita distorcer a forma da curva. Câmeras de celular aplicam processamento próprio, então o resultado nunca é perfeitamente linear.
- **Métricas de cor** (escolha no arquivo de configuração):
  - **HSV, matiz (H):** a cor como ângulo no círculo cromático. Mostra bem a mudança de cor, mas fica ruidoso no escuro.
  - **Lab:** luminosidade (L) e dois eixos de cor (a: verde↔vermelho, b: azul↔amarelo). A distância entre cores se aproxima da diferença percebida (ΔE).
  - **Razão B/G:** divisão entre canais. Mostra direto qual componente decai mais rápido. Só faz sentido depois de linearizar a gama e subtrair o fundo.
- **Métricas de análise:** desvio padrão, intensidade máxima, tempo para cair a X% e comparações entre amostras.
- **Checagem de sincronização** (versão 1, forma simples): detecta o instante em que a luz apaga e compara com o tempo do protocolo. O desvio fica numa coluna de qualidade (`desvio_queda_s`), e vídeos acima da tolerância são marcados para revisão. **Isso não realinha o tempo.** Realinhar só entra se o usuário escolher, igual para todo o conjunto e registrado no arquivo auxiliar, porque realinhar só alguns vídeos criaria diferenças artificiais entre grupos.

### 6. Gráficos e ajuste de modelos (versão 2)

- **Modelos de decaimento:** exponencial simples e dupla, exponencial esticada e lei de potência, com os parâmetros guardados por vídeo. Os modelos são comparados pelo erro de ajuste, e novos podem ser adicionados em `modelos.py`.
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
- tolerância de sincronia (sugestão inicial: 0,3 s);
- métrica de cor e modelos de decaimento escolhidos.

## Uso previsto

Um projeto único com subcomandos (os nomes podem mudar):

```bash
luminescencia auditar data/      # 1. Auditoria
luminescencia roi                # 2. Detecção da ROI
luminescencia extrair            # 3. Extração
luminescencia consolidar         # 4. Consolidação
luminescencia pos                # 5. Pós-processamento
luminescencia                    # menu interativo, com pausas entre as etapas
```

## Estrutura do repositório

```
analise-luminescencia/
├── src/luminescencia/
│   ├── cli.py               # subcomandos
│   ├── auditoria.py         # etapa 1
│   ├── roi.py               # etapa 2
│   ├── extracao.py          # etapa 3
│   ├── consolidacao.py      # etapa 4
│   ├── posprocessamento.py  # etapa 5
│   └── nomes.py             # leitor de nomes de arquivo
├── config.yaml
├── data/                    # vídeos (fora do Git)
├── output/                  # relatórios, ROIs, CSVs (fora do Git)
├── tests/
├── requirements.txt
└── README.md
```

## Instalação

Requer Python 3.10 ou superior e Git.

```bash
git clone https://github.com/SEU_USUARIO/analise-luminescencia.git
cd analise-luminescencia
python -m venv .venv
.venv\Scripts\activate          # Windows (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Dependências atuais: `opencv-python`, `numpy`, `pandas`, `tqdm`. A versão 2 deve exigir também `scipy`. Os vídeos não ficam no repositório.

## Pontos em aberto

- Formato exato do nome dos vídeos e da proporção dos 3 pós (define o leitor de nomes e as colunas do CSV consolidado).
- Nome do vídeo de controle.
- Limites de "suspeito" da ROI e tolerância de sincronia, a calibrar com os primeiros vídeos.
- Se a checagem de sincronização, no futuro, poderá realinhar o tempo (por padrão, não realinha).
