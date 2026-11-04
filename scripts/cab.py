"""Gera cabecalhos.txt (formato e colunas dos CSVs grandes) para enviar à IA no lugar dos CSVs."""
import pandas as pd

arquivos = [
    "output/modelagem/ajustes.csv",
    "output/modelagem/comparacao_modelos.csv",
    "output/tabela/dataset.csv",
    "output/tabela/descritores.csv",
    "output/posprocessamento/videos_pos.csv",
    "output/consolidacao/videos.csv",
]
with open("cabecalhos.txt", "w", encoding="utf-8") as f:
    for a in arquivos:
        try:
            d = pd.read_csv(a, nrows=100000)
            f.write(f"{a}  linhas={len(d)}\n{','.join(d.columns)}\n\n")
        except Exception as e:
            f.write(f"{a}  ERRO: {e}\n\n")
print("ok: cabecalhos.txt")
