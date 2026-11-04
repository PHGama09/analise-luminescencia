# Guia de reprodução: projeto `luminescencia` (v0.13.0)

Como refazer os números de referência (`dataset.csv` 236 × 106, `descritores.csv` 236 × 44) e conferir que nada mudou.
Os comandos abaixo vêm do `cli.py` e do `config.yaml`; a cadeia `tudo` e a conferência são as da P6.

## 1. Ambiente
- Windows, Python 3.14 (o ambiente de referência do projeto). O pacote exige Python ≥ 3.10.
- Na raiz do projeto (onde estão `pyproject.toml`, `config.yaml` e `curadoria.csv`):
  ```
  pip install -e ".[dev]"
  ```
  Dependências (do `pyproject.toml`): opencv-python, numpy, pandas, tqdm, pyyaml, scipy, matplotlib; `dev` = pytest.
- Registre as versões usadas: `pip freeze > output/versoes.txt` e `luminescencia --help` (confirma que o comando existe).
  O número da versão do pacote (`luminescencia.__version__`) é carimbado nos relatórios pelo `tudo`.

## 2. O que precisa estar no lugar
| Item | Onde | Observação |
|---|---|---|
| Vídeos | `pastas.videos` do `config.yaml` | Só as etapas 1 a 3 leem vídeos. Nenhum vídeo é renomeado, movido ou gravado. |
| `config.yaml` | raiz | Sem alterar limiares entre execuções que serão comparadas. |
| `curadoria.csv` | raiz (`curadoria.arquivo`) | Suas decisões de inclusão; é a entrada que define quais vídeos entram (236 aceitos). Pendência conhecida: `C/1/RGB_05_05_05_v1a` está fora do dataset. |
| `output/extracao/` | gerado pela etapa 3 | Um CSV por vídeo + `extracao_resumo.csv`. É a entrada de `consolidar`. |

## 2b. Se já existe `output/extracao/`
Comece no passo 4. As etapas 1 a 3 são as únicas que dependem dos vídeos e demoram.

## 3. Da gravação ao CSV (precisa dos vídeos; não está no `tudo`)
```
luminescencia auditar            # 1. auditoria dos vídeos (output/auditoria)
luminescencia roi                # 2. ROI de cada amostra + mosaico de conferência (output/roi/roi.json)
luminescencia extrair            # 3. um CSV por vídeo (output/extracao); --refazer ignora o cache
```
Opções úteis: `--workers N`, `--refazer`; `extrair --videos B_15 RB_075` limita a alguns vídeos.

## 4. Da extração à tabela (a cadeia do `tudo`)
Ordem das etapas (também é a ordem manual):
```
luminescencia consolidar         # 4. vídeos aceitos, preto corrigido por vídeo, formato longo
luminescencia pos                # 5. janela fixa do ajuste (início 2,5 s), saturação, canais usáveis
luminescencia modelar            # 6. ajustes em lote (todos os caminhos); ~13 min com 1 CPU
luminescencia tabela             # 6b. dataset.csv + dicionario.csv
luminescencia comparar_modelos   # 6c. exp simples, dupla, esticada e potência (só análise)
luminescencia descritores        # 6d. descritores.csv + dicionario_descritores.csv
luminescencia offset             # P5 (opcional): teste do offset cantos − ROI
luminescencia graficos           # 7. gráficos
```
Atalho (roda a cadeia e carimba a versão nos relatórios):
```
luminescencia tudo                              # de consolidar até graficos
luminescencia tudo --de pos --ate tabela        # só um trecho (--de e --ate aceitam o nome de qualquer etapa)
luminescencia tudo --com-offset                 # inclui offset depois de descritores
luminescencia tudo --no-parar-no-erro           # continua nas etapas seguintes se uma falhar (o padrão é parar)
```
Cada etapa também aceita pastas de entrada/saída (`--extracao`, `--consolidacao`, `--pos`, `--modelagem`, `--saida`); veja `luminescencia <etapa> --help`.

## 5. Conferir os números de referência
A referência fica em `tests/referencia/referencia.json` (perfil estrutural e estatístico de cada tabela e relatório; as
tolerâncias vêm de `tudo.rtol` = 1e-3 e `tudo.atol` = 1e-9 no `config.yaml`; contagens de linhas, colunas e vazios são exatas).

```
luminescencia tudo --conferir        # roda a cadeia e confere o output com a referência (código de saída != 0 se divergir)
luminescencia tudo --so-conferir     # NÃO roda etapa nenhuma: só confere o output que já está em disco
```
- Use `--so-conferir` para verificar um `output/` existente (por exemplo, depois de copiar o `output/` de outra máquina).
- `luminescencia tudo --gravar-referencia` **substitui** a referência pelo output atual. Só use quando a mudança de números for
  intencional e revisada; registre o motivo e a versão. Nunca para "fazer o teste passar".
- Divergência pequena (≤ 0,1%) passa; 1% falha. Uma divergência real aponta a coluna/estatística diferente.

## 6. Análises do lote Q4 (só leitura; não estão no `tudo` e não alteram as tabelas)
```
luminescencia deriva             # B5: deriva de sessão A/B/C           (output/deriva)
luminescencia colinearidade      # B13: redundância entre colunas       (output/colinearidade)
luminescencia janela_r           # B12: fim de janela do R              (output/janela_r)
luminescencia saturados          # B11: saturados × irmãos, na mesma janela (output/saturados)
```
Todas leem `output/` e gravam em `output/<nome>/relatorio_<nome>.txt`.

## 7. Teste sem dados reais
```
pytest                               # toda a suíte
pytest tests/test_integracao.py      # integração: ~5 s, vídeos sintéticos em tmp_path
```
`tests/test_integracao.py` gera poucos vídeos sintéticos (formato do `B_15_v1a.csv`; decaimento duplo exponencial + ruído) e roda
`consolidar → pos → modelar → tabela` com `workers: 1`. Confere: formas das tabelas, ausência de vazamento do alvo, preto por
vídeo, τ recuperado (erro ≤ 1,3% medido; tolerância de 5%/8%) e determinismo (duas execuções iguais). Ele **não** substitui a
conferência do §5, que usa os dados reais.

## 8. Checklist de uma reprodução
1. Mesma versão do pacote (`__version__`), mesmo `config.yaml` e mesmo `curadoria.csv`.
2. `output/extracao/` igual (ou refazer as etapas 1 a 3 com os mesmos vídeos).
3. `luminescencia tudo --conferir` termina com código 0 (ou `--so-conferir` sobre um `output/` já gerado).
4. `pytest` passa.
5. Guarde `output/versoes.txt` (`pip freeze`) junto dos resultados.

## 9. Problemas comuns
- `FileNotFoundError ... rode 'luminescencia consolidar'`: falta a etapa anterior; rode na ordem do §4.
- `modelar` lento: com 1 CPU leva ~13 min; ajuste `modelagem.workers` (o teste de integração fixa 1).
- Arquivo aberto no Excel impede gravar o CSV: feche e rode de novo.
- Mudou algum limiar do `config.yaml`: a conferência vai divergir; isso é esperado e deve ser decidido, não mascarado.
