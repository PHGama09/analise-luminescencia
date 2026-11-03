"""Leitor de nomes de arquivo.

Formato dos vídeos de amostra:   <CORES>_<qtd>[_<qtd>[_<qtd>]]_v<N><letra>.avi

  CORES   letras dos pós presentes, na ordem em que aparecem as quantidades (R, G, B)
  qtd     quantidade de cada pó, escrita sem o ponto decimal: o PRIMEIRO dígito é a
          parte inteira e o resto são as casas decimais
            1 -> 1     05 -> 0,5     075 -> 0,75     15 -> 1,5
  N       número do vídeo (1 a 5)
  letra   amostra (a, b ou c)

  Exemplos:  R_15_v1a      -> só R, 1,5       | vídeo 1 | amostra A
             GB_1_05_v3b   -> G=1, B=0,5      | vídeo 3 | amostra B
             RG_075_075_v2c -> R=0,75, G=0,75 | vídeo 2 | amostra C

O vídeo de controle (amostra sem luminescência) é reconhecido por uma palavra no
nome, definida em config.yaml (nomes.controle), por exemplo "referencia".

Erros de digitação simples (espaço sobrando, "__" no lugar de "_", falta do "_" antes do "v",
maiúsculas/minúsculas fora do padrão) são tolerados: o nome é lido normalmente, mas o resultado
traz "correcoes" (o que foi ajustado) e "nome_sugerido" (como o arquivo deveria se chamar).
Quem chama decide o que fazer com isso; este módulo nunca renomeia nada.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

PADRAO_AMOSTRA = re.compile(
    r"(?P<cores>[RGB]{1,3})(?P<qtds>(?:_\d+)+)_v(?P<video>\d+)(?P<amostra>[a-c])", re.IGNORECASE
)
VIDEOS_VALIDOS = range(1, 6)


def _sem_acento(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn").lower()


def _normalizar(stem: str) -> tuple[str, list[str]]:
    """Corrige erros de digitação simples. Retorna (texto corrigido, lista do que foi corrigido)."""
    correcoes, t = [], stem
    if re.search(r"\s", t):
        t = re.sub(r"\s+", "", t)
        correcoes.append("espaço(s) removido(s)")
    if "__" in t:
        t = re.sub(r"_{2,}", "_", t)
        correcoes.append("'_' duplicado reduzido a um")
    t2 = re.sub(r"(?<=\d)(?=v\d+[a-c]$)", "_", t, flags=re.IGNORECASE)
    if t2 != t:
        t = t2
        correcoes.append("faltava '_' antes do número do vídeo")
    return t, correcoes


def nome_canonico(cores: str, qtds: list[str], video: int, amostra: str) -> str:
    return f"{cores.upper()}_{'_'.join(qtds)}_v{video}{amostra.lower()}"


def valor_da_quantidade(token: str) -> float:
    """'1' -> 1.0, '05' -> 0.5, '075' -> 0.75, '15' -> 1.5"""
    return float(token) if len(token) == 1 else float(f"{token[0]}.{token[1:]}")


def ler_nome(nome_arquivo: str, cfg_nomes: dict | None = None) -> tuple[dict | None, str]:
    """Extrai os campos do nome. Retorna (campos, mensagem_de_erro).

    Em campos, a chave opcional "aviso" traz um alerta que não impede a leitura.
    """
    cfg_nomes = cfg_nomes or {}
    stem = Path(nome_arquivo).stem

    palavra = cfg_nomes.get("controle")
    if palavra and _sem_acento(palavra) in _sem_acento(stem):
        return {"tipo": "controle"}, ""

    normalizado, correcoes = _normalizar(stem)
    m = PADRAO_AMOSTRA.fullmatch(normalizado)
    if not m:
        return None, "nome fora do padrão esperado (ex.: GB_1_05_v1a)"

    cores = m.group("cores").upper()
    qtds = m.group("qtds").strip("_").split("_")
    if len(set(cores)) != len(cores):
        return None, f"cor repetida em '{cores}'"
    if len(qtds) != len(cores):
        return None, f"{len(cores)} cor(es) ({cores}) mas {len(qtds)} quantidade(s) no nome"
    video = int(m.group("video"))
    if video not in VIDEOS_VALIDOS:
        return None, f"número de vídeo {video} fora do intervalo {VIDEOS_VALIDOS.start}-{VIDEOS_VALIDOS.stop - 1}"

    valores = {c: valor_da_quantidade(q) for c, q in zip(cores, qtds)}
    campos = {
        "tipo": "amostra",
        "cores": cores,
        "prop_R": valores.get("R", 0.0),
        "prop_G": valores.get("G", 0.0),
        "prop_B": valores.get("B", 0.0),
        "proporcao": "_".join(f"{c}{v:g}" for c, v in valores.items()),
        "amostra": m.group("amostra").upper(),
        "video": video,
    }
    canonico = nome_canonico(cores, qtds, video, m.group("amostra"))
    if canonico != normalizado and "maiúsculas/minúsculas" not in correcoes:
        correcoes.append("maiúsculas/minúsculas ajustadas")
    if canonico != stem:
        campos["correcoes"] = correcoes or ["formato ajustado"]
        campos["nome_sugerido"] = canonico + Path(nome_arquivo).suffix.lower()
    soma, tol = cfg_nomes.get("soma_esperada"), cfg_nomes.get("soma_tolerancia", 0.01)
    if soma is not None and abs(sum(valores.values()) - soma) > tol:
        campos["aviso"] = f"soma das quantidades ({sum(valores.values()):g}) difere do esperado ({soma})"
    return campos, ""
