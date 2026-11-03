### 6e. Comando `tudo`: conferência de regressão (P6b, v0.12.0)

**P6 feita na v0.12.0.**

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
