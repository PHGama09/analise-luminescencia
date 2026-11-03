## 6e. Comando `tudo`

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
- `--conferir`, `--gravar-referencia`, `--so-conferir`: reservados à parte B (regressão); por ora imprimem "regressão: implementada na parte B" e saem com 0 sem fazer nada.

### Carimbo de versão

Depois de cada etapa que terminou com 0, `tudo` acrescenta ao fim do relatório dela:

```
# --- carimbo luminescencia ---
versao_codigo: 0.11.3
config_sha256: <16 primeiros hex do SHA-256 dos bytes do config usado>
data: AAAA-MM-DD HH:MM
# --- fim carimbo ---
```

O bloco é substituído, nunca duplicado: rodar duas vezes deixa um só carimbo. O carimbo é feito pelo `tudo` (as etapas rodadas isoladamente não o gravam). Os relatórios são lidos de `pastas.saida` + os caminhos em `tudo.etapas` do `config.yaml`.
