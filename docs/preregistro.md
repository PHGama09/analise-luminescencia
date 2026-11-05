# Pré-registro do teste na amostra C

Data: 2026-11-04T20:42:40  
Versão do código: 0.16.1

**Mudar qualquer item depois de abrir C invalida o teste.**

## Hashes (SHA-256)

- `dataset.csv`: `fc0a5a6c9efa54c75dbbf085bc0f347018bf8c46a0780ebb482b92fca830858f`
- `descritores.csv`: `1a341294c281e565f41e2ac7af703e82139b8d16bdf25a09e3a6dc3b7271f944`
- `dicionario.csv`: `8048a1c9a365e52ac1bc02a4940806e49b03b56ad035dfa017d65d8d72fc6781`
- `dicionario_descritores.csv`: `cf87ac79b4d6066b2755726a132fe44b7362bb4608438f3ef0ae7e59e7ab0e8a`
- `config.yaml` (arquivo inteiro): `236bb966b26284e93de665b3750f48642a9628e2dcaa2f106488e921c5a922d1` (16 primeiros: `236bb966b26284e9`)

Seções do config conferidas pela trava (SHA-256 do conteúdo lido):

- `protocolo`: `a41d49458e65a41ce916a050ca7da1e8aac763d941ebe5eadf7d6bc1a30f6852`
- `ajuste`: `5b6b5c2f9bb5731816bc9a93c4534317d668b16c32853ae3385be50aae283b06`
- `posprocessamento`: `880cd27337ada61fab056252470966586e173df96b38a4f14a2187358b077d10`
- `modelagem`: `96083811aefdbb18b0332eb6c60e06973fce2ff1a609aad7341aaad4765bf17d`
- `entradas`: `6eb81a5a4390f5e5741d1639451815eb87027b04c1d89cf3ffe1229cc123835a`
- `descritores`: `27d534b93dfcb6977512a8d27ce42caf386428e941d8832f6741ffcbd052a6b4`
- `avaliacao`: `382f3644058a88eab2593d191511ebbf10d65d78e69dee016c408e35caba590e`
- `linhas_base`: `dadadb1c3e542ba915de87f2dd5fb1e6c7a3768b3f671c9dcb1936c89a9e2f14`
- `preregistro`: `93ee1546541c06eecec2de8d3b11ed29b0b01c1fcd46a677300023786ccbc94f`

## Variante primária (D4)

```json
{
  "id": "c0_descritores",
  "decisao": "D4",
  "caminhos": [
    "c0"
  ],
  "descritores": true,
  "descritores_prioritarios": [
    "razoes entre canais",
    "t50",
    "r_5s",
    "auc_10s"
  ],
  "conjuntos_de_colunas": [
    "so_descritores",
    "so_tau",
    "misto"
  ],
  "canais_sem_po": true,
  "bloquear_diagnosticos": true,
  "dependencia_declarada": "o resultado depende do início da janela de ajuste (2,5 s no protocolo atual)"
}
```

## Sensibilidades

```json
[
  "caminho cl e caminhos cl+c0",
  "sem descritores",
  "canais_sem_po false (sinal forte medido; não equivale a \"sem vazamento\")",
  "métricas com e sem a proporção G0.5_B1"
]
```

## Critério de sucesso (D3)

```json
{
  "decisao": "D3",
  "nivel_minimo_razao_max": 0.7,
  "nivel_forte_razao_max": 0.5,
  "condicao_b": "nenhuma proporção com MAE pior que o da média",
  "condicao_c": "ganho pareado por proporção (16 pares) sobre a regra fixa física (fração do canal proporcional ao seu I0, normalizada para somar 1; sem parâmetros ajustados), IC 95% por reamostragem que não inclui 0; k-vizinhos adiado, nenhum modelo por enquanto",
  "falha": "falhou em (b) ou (c), ou ficou acima de 70% = não demonstrado"
}
```

## Esquema de validação (D2, PROVISÓRIO)

```json
{
  "estado": "provisória (levar ao orientador)",
  "interna": "A<->B para escolher entradas e hiperparâmetros; leave-one-proporção-out só como estresse; leave-one-preparação-out como apoio",
  "unidade": "preparação (proporção x amostra); regravações sempre do mesmo lado",
  "final": "C aberta uma vez, no fim, via abrir_c",
  "limite_d6": "uma preparação de validação por composição (D6, provisória)"
}
```

## Treino

Amostras de treino: A, B; amostra final (não aberta): C. 148 vídeos em 31 preparações (sem os excluídos por saturação).

| proporção | amostra | vídeos |
|---|---|---|
| B1.5 | A | 5 |
| B1.5 | B | 4 |
| G0.5_B1 | B | 5 |
| G0.75_B0.75 | A | 5 |
| G0.75_B0.75 | B | 3 |
| G1.5 | A | 5 |
| G1.5 | B | 5 |
| G1_B0.5 | A | 5 |
| G1_B0.5 | B | 5 |
| R0.25_G0.25_B1 | A | 5 |
| R0.25_G0.25_B1 | B | 4 |
| R0.25_G1_B0.25 | A | 5 |
| R0.25_G1_B0.25 | B | 5 |
| R0.5_B1 | A | 5 |
| R0.5_B1 | B | 5 |
| R0.5_G0.5_B0.5 | A | 5 |
| R0.5_G0.5_B0.5 | B | 5 |
| R0.5_G1 | A | 5 |
| R0.5_G1 | B | 4 |
| R0.75_B0.75 | A | 5 |
| R0.75_B0.75 | B | 5 |
| R0.75_G0.75 | A | 5 |
| R0.75_G0.75 | B | 4 |
| R1.5 | A | 5 |
| R1.5 | B | 5 |
| R1_B0.5 | A | 5 |
| R1_B0.5 | B | 5 |
| R1_G0.25_B0.25 | A | 5 |
| R1_G0.25_B0.25 | B | 5 |
| R1_G0.5 | A | 4 |
| R1_G0.5 | B | 5 |

**Mudar qualquer item depois de abrir C invalida o teste.**
