"""Curadoria manual dos vídeos (curadoria.csv).

O arquivo guarda as SUAS decisões sobre cada vídeo e nunca é decidido pelo programa:

  chave            identifica o vídeo pelo hash do conteúdo (sha1:...). Renomear ou mover o arquivo
                   não perde a decisão; o caminho é só para exibição.
  caminho          onde o vídeo está agora (o programa atualiza)
  status_auditoria resultado da última auditoria (o programa atualiza)
  incluir          sim / nao / pendente (você preenche; novos vídeos entram como pendente)
  motivo           lista fechada (config: curadoria.motivos), para você filtrar depois
  nome_sugerido    sugestão do programa (o programa atualiza)
  nome_aprovado    o nome que VOCÊ aprova; o programa ainda não renomeia nada com ele
  observacao       texto livre

A cada auditoria o arquivo é atualizado: vídeos novos entram, as colunas suas são preservadas e
vídeos que sumiram ficam como "ausente" (a decisão não é apagada). Antes de gravar, a versão
anterior é copiada para curadoria.csv.bak.
"""
from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path

CAMPOS = ["chave", "caminho", "status_auditoria", "incluir", "motivo", "nome_sugerido", "nome_aprovado", "observacao"]
COLUNAS_DO_USUARIO = ("incluir", "motivo", "nome_aprovado", "observacao")
INCLUIR_VALIDOS = {"sim", "nao", "pendente"}
MOTIVOS_PADRAO = [
    "erro_de_gravacao", "duracao_invalida", "queda_nao_detectada", "nome_errado", "pasta_errada",
    "video_duplicado", "condicao_diferente", "qualidade_ruim", "outro",
]
GRAVIDADE = {"ilegivel": 0, "truncado": 0, "fora_do_padrao": 1, "revisar": 2, "ok": 3}


def _ler(caminho: Path) -> dict:
    if not caminho.exists():
        return {}
    with caminho.open(encoding="utf-8-sig", newline="") as f:
        return {l["chave"]: l for l in csv.DictReader(f) if l.get("chave")}


def _incluir_normalizado(valor: str) -> str:
    v = (valor or "").strip().lower().replace("não", "nao")
    return v or "pendente"


def atualizar_curadoria(resultados: list[dict], caminho, motivos: list[str] | None = None) -> dict:
    """Cria ou atualiza o curadoria.csv. Retorna contagens e avisos (nunca levanta erro por arquivo aberto)."""
    caminho, motivos = Path(caminho), list(motivos or MOTIVOS_PADRAO)
    existentes = _ler(caminho)

    por_chave: dict[str, list[dict]] = {}
    for r in resultados:
        if r.get("hash"):
            por_chave.setdefault("sha1:" + r["hash"], []).append(r)

    linhas, novos = [], 0
    for chave, rs in por_chave.items():
        antigo = existentes.pop(chave, None)
        status = min((r["status"] for r in rs), key=lambda s: GRAVIDADE.get(s, 9))
        linha = {
            "chave": chave, "caminho": " | ".join(sorted(r["caminho"] for r in rs)), "status_auditoria": status,
            "nome_sugerido": next((r["nome_sugerido"] for r in rs if r.get("nome_sugerido")), ""),
            "incluir": "pendente", "motivo": "", "nome_aprovado": "", "observacao": "",
        }
        if antigo:
            linha.update({k: antigo.get(k, "") or "" for k in COLUNAS_DO_USUARIO})
        else:
            novos += 1
        linha["incluir"] = _incluir_normalizado(linha["incluir"])
        linhas.append(linha)

    ausentes = 0
    for antigo in existentes.values():                        # sumiram da pasta: guarda a decisão
        antigo["status_auditoria"] = "ausente"
        antigo["incluir"] = _incluir_normalizado(antigo.get("incluir", ""))
        linhas.append({c: antigo.get(c, "") or "" for c in CAMPOS})
        ausentes += 1
    linhas.sort(key=lambda l: (l["caminho"], l["chave"]))

    avisos = []
    for l in linhas:
        if l["incluir"] not in INCLUIR_VALIDOS:
            avisos.append(f"{l['caminho']}: incluir = '{l['incluir']}' (use sim, nao ou pendente)")
        if l["motivo"] and l["motivo"] not in motivos:
            avisos.append(f"{l['caminho']}: motivo = '{l['motivo']}' não está na lista ({', '.join(motivos)})")

    resumo = {
        "arquivo": caminho, "total": len(linhas), "novos": novos, "ausentes": ausentes,
        "pendentes": sum(l["incluir"] == "pendente" for l in linhas), "avisos": avisos, "erro": "",
    }
    try:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        if caminho.exists():
            shutil.copy2(caminho, caminho.with_suffix(".csv.bak"))
        tmp = caminho.with_suffix(".csv.tmp")
        with tmp.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=CAMPOS, extrasaction="ignore")
            w.writeheader()
            w.writerows(linhas)
        os.replace(tmp, caminho)
    except OSError as erro:
        resumo["erro"] = f"não foi possível gravar {caminho} (aberto em outro programa?): {erro}"
    return resumo
