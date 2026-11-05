"""Pré-registro do teste em C (BLOCO 1) e verificação usada pela trava `avaliacao.abrir_c`.

`luminescencia preregistrar` grava docs/preregistro.json e docs/preregistro.md com: data, versão do código, SHA-256 de dataset.csv,
descritores.csv e dos dicionários, hash do config (arquivo inteiro, como o `tudo` faz, e por seção relevante), a variante primária (D4)
e as sensibilidades, o critério D3, o esquema de validação provisório (D2) e as preparações do treino. As declarações (D2, D3, D4,
sensibilidades) vêm da seção `preregistro` do config.yaml: o código não escreve decisões nem resultados.

Mudar qualquer item depois de abrir C invalida o teste. A verificação compara os hashes guardados no JSON com os arquivos atuais:
arquivos da tabela e SEÇÕES do config listadas no próprio JSON (assim, mexer em seção sem relação com o teste não trava, e tirar
uma seção da lista no config não destrava).

Trava do aval (v0.16.5): enquanto `preregistro.aval_orientador` não tiver a data do aval (AAAA-MM-DD) ou alguma declaração ainda
disser "provisória", `preregistrar` recusa gravar (nem `--forcar` passa) e `verificar_preregistro` mantém C travada. Destravar é um
ato explícito e registrado no Git: preencher a data e trocar o estado das declarações no config.yaml.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import date, datetime
from pathlib import Path

from . import __version__
from .tudo import hash_config

PADROES = {
    "arquivo_json": "docs/preregistro.json",
    "arquivo_md": "docs/preregistro.md",
    "arquivos_tabela": ["dataset.csv", "descritores.csv", "dicionario.csv", "dicionario_descritores.csv"],
    "secoes_config": ["protocolo", "ajuste", "posprocessamento", "modelagem", "entradas", "descritores",
                      "avaliacao", "linhas_base", "preregistro"],
    "declaracoes": {},
    "aval_orientador": None,          # data do aval (AAAA-MM-DD); vazio = pré-registro bloqueado e C travada
}
FORMATO = 1
NOTA = "Mudar qualquer item depois de abrir C invalida o teste."
_RE_SHA = re.compile(r"^[0-9a-f]{64}$")


def parametros(cfg: dict) -> dict:
    p = dict(PADROES)
    p.update((cfg or {}).get("preregistro") or {})
    if not p["arquivos_tabela"] or not p["secoes_config"]:
        raise ValueError("preregistro.arquivos_tabela e preregistro.secoes_config não podem ser vazios")
    return p


def _sem_acento(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn").lower()


def _textos(valor, caminho=""):
    """(caminho, texto) de todo texto dentro das declarações (dict, lista ou valor solto)."""
    if isinstance(valor, dict):
        for k, v in valor.items():
            yield from _textos(v, f"{caminho}.{k}" if caminho else str(k))
    elif isinstance(valor, (list, tuple)):
        for i, v in enumerate(valor):
            yield from _textos(v, f"{caminho}[{i}]")
    elif isinstance(valor, str):
        yield caminho, valor


def pendencias_aval(cfg: dict) -> list[str]:
    """O que ainda impede o pré-registro: falta a data do aval do orientador ou há declaração marcada como provisória."""
    p = parametros(cfg)
    pend = []
    aval = p.get("aval_orientador")
    if not (isinstance(aval, date) or (isinstance(aval, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", aval.strip()))):
        pend.append("preregistro.aval_orientador está vazio (preencha com a data do aval, AAAA-MM-DD)")
    for caminho, texto in _textos(p.get("declaracoes") or {}):
        if "provis" in _sem_acento(texto):
            pend.append(f"preregistro.declaracoes.{caminho} ainda diz provisória: {texto!r}")
    return pend


# ------------------------------------------------------------------------------------------------------------- hashes

def sha256_arquivo(caminho) -> str:
    h = hashlib.sha256()
    with Path(caminho).open("rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def sha256_secao(valor) -> str:
    """SHA-256 da seção do config já lida (independe de comentários, ordem de chaves e fim de linha)."""
    texto = json.dumps(valor, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def _pasta_tabela(cfg: dict, pasta_tabela=None) -> Path:
    from .avaliacao import _pasta_tabela as pt
    return pt(cfg, pasta_tabela)


def _hashes_tabela(cfg: dict, p: dict, pasta_tabela=None) -> dict:
    pasta = _pasta_tabela(cfg, pasta_tabela)
    saida = {}
    for nome in p["arquivos_tabela"]:
        caminho = pasta / nome
        if not caminho.is_file():
            raise FileNotFoundError(f"arquivo do pré-registro não encontrado: {caminho.resolve()}")
        saida[nome] = sha256_arquivo(caminho)
    return saida


def _hashes_secoes(cfg: dict, nomes) -> dict:
    return {n: sha256_secao(cfg.get(n)) for n in nomes}


# --------------------------------------------------------------------------------------------------------- montagem

def _resumo_treino(cfg: dict, pasta_tabela=None) -> dict:
    from .avaliacao import carregar_ab, chave_preparacao, parametros as pav

    df, _ = carregar_ab(cfg, pasta_tabela)
    prep = chave_preparacao(df)
    contagem = df.groupby(prep.to_numpy()).agg(proporcao=("proporcao", "first"), amostra=("amostra", "first"), n_videos=("video", "size"))
    return {"amostras": pav(cfg)["amostras_treino"], "amostra_final_nao_aberta": pav(cfg)["amostra_final"],
            "n_videos": int(len(df)), "n_preparacoes": int(len(contagem)),
            "preparacoes": [{"proporcao": r.proporcao, "amostra": r.amostra, "n_videos": int(r.n_videos)}
                            for r in contagem.sort_values(["proporcao", "amostra"]).itertuples()]}


def montar_preregistro(cfg: dict, config_path="config.yaml", pasta_tabela=None, agora: datetime | None = None) -> dict:
    p = parametros(cfg)
    ausentes = [n for n in p["secoes_config"] if n not in cfg]
    if ausentes:
        raise ValueError(f"seções do config listadas em preregistro.secoes_config mas ausentes do config: {ausentes}")
    agora = agora or datetime.now()
    cfg_path = Path(config_path)
    return {
        "formato": FORMATO,
        "data": agora.isoformat(timespec="seconds"),
        "versao_codigo": __version__,
        "nota": NOTA,
        "hashes_tabela": _hashes_tabela(cfg, p, pasta_tabela),
        "config": {
            "arquivo": cfg_path.name,
            "sha256_16": hash_config(cfg_path),
            "sha256": sha256_arquivo(cfg_path) if cfg_path.is_file() else "indisponivel",
            "secoes_sha256": _hashes_secoes(cfg, p["secoes_config"]),
        },
        "declaracoes": p["declaracoes"],
        "treino": _resumo_treino(cfg, pasta_tabela),
    }


def texto_markdown(reg: dict) -> str:
    d = reg.get("declaracoes") or {}
    L = ["# Pré-registro do teste na amostra C", "",
         f"Data: {reg['data']}  ", f"Versão do código: {reg['versao_codigo']}", "",
         f"**{reg['nota']}**", "", "## Hashes (SHA-256)", ""]
    L += [f"- `{n}`: `{h}`" for n, h in reg["hashes_tabela"].items()]
    c = reg["config"]
    L += [f"- `{c['arquivo']}` (arquivo inteiro): `{c['sha256']}` (16 primeiros: `{c['sha256_16']}`)", "",
          "Seções do config conferidas pela trava (SHA-256 do conteúdo lido):", ""]
    L += [f"- `{n}`: `{h}`" for n, h in c["secoes_sha256"].items()]

    def bloco(titulo, valor):
        L.extend(["", f"## {titulo}", "", "```json", json.dumps(valor, ensure_ascii=False, indent=2), "```"])

    for chave, titulo in (("variante_primaria", "Variante primária (D4)"), ("sensibilidades", "Sensibilidades"),
                          ("criterio_d3", "Critério de sucesso (D3)"), ("validacao_d2", "Esquema de validação (D2, PROVISÓRIO)")):
        if chave in d:
            bloco(titulo, d[chave])
    outras = {k: v for k, v in d.items() if k not in ("variante_primaria", "sensibilidades", "criterio_d3", "validacao_d2")}
    if outras:
        bloco("Outras declarações", outras)
    t = reg["treino"]
    L += ["", "## Treino", "",
          f"Amostras de treino: {', '.join(t['amostras'])}; amostra final (não aberta): {t['amostra_final_nao_aberta']}. "
          f"{t['n_videos']} vídeos em {t['n_preparacoes']} preparações (sem os excluídos por saturação).", "",
          "| proporção | amostra | vídeos |", "|---|---|---|"]
    L += [f"| {r['proporcao']} | {r['amostra']} | {r['n_videos']} |" for r in t["preparacoes"]]
    L += ["", f"**{reg['nota']}**", ""]
    return "\n".join(L)


# -------------------------------------------------------------------------------------------------------- verificação

def _ler_registro(arquivo: Path) -> dict:
    if not arquivo.is_file():
        raise RuntimeError(f"C está travada: não há pré-registro em {arquivo}. Rode 'luminescencia preregistrar' e confira o documento "
                           "ANTES de abrir C.")
    try:
        reg = json.loads(arquivo.read_text(encoding="utf-8"))
    except (OSError, ValueError) as erro:
        raise RuntimeError(f"C está travada: {arquivo} não é um JSON legível ({erro}).") from erro
    try:
        ok = (reg.get("formato") == FORMATO and all(_RE_SHA.match(h or "") for h in reg["hashes_tabela"].values())
              and reg["hashes_tabela"] and reg["config"]["secoes_sha256"]
              and all(_RE_SHA.match(h or "") for h in reg["config"]["secoes_sha256"].values()))
    except (AttributeError, KeyError, TypeError):
        ok = False
    if not ok:
        raise RuntimeError(f"C está travada: {arquivo} não tem o formato de pré-registro esperado (hashes SHA-256 ausentes ou inválidos).")
    return reg


def verificar_preregistro(cfg: dict, pasta_tabela=None, arquivo=None) -> dict:
    """Devolve o pré-registro se válido e se os hashes baterem com os arquivos atuais; senão RuntimeError listando o que diverge."""
    p = parametros(cfg)
    reg = _ler_registro(Path(arquivo or p["arquivo_json"]))
    pend = pendencias_aval(cfg)
    if pend:
        raise RuntimeError("C está travada: falta o aval do orientador no config.yaml:\n  - " + "\n  - ".join(pend))
    pasta = _pasta_tabela(cfg, pasta_tabela)
    problemas = []
    for nome, esperado in reg["hashes_tabela"].items():
        caminho = pasta / nome
        if not caminho.is_file():
            problemas.append(f"{nome}: arquivo ausente em {pasta}")
        elif sha256_arquivo(caminho) != esperado:
            problemas.append(f"{nome}: hash diferente do pré-registro (arquivo alterado depois de preregistrar)")
    for secao, esperado in reg["config"]["secoes_sha256"].items():
        if secao not in cfg:
            problemas.append(f"config, seção '{secao}': ausente")
        elif sha256_secao(cfg.get(secao)) != esperado:
            problemas.append(f"config, seção '{secao}': conteúdo diferente do pré-registro")
    if problemas:
        raise RuntimeError("C está travada: o pré-registro não confere com os arquivos atuais:\n  - " + "\n  - ".join(problemas)
                           + "\nDesfaça a alteração, ou (se for intencional) gere novo pré-registro com --forcar ANTES de abrir C; "
                             "o antigo é guardado com data.")
    return reg


# ---------------------------------------------------------------------------------------------------------- comando

def _guardar_antigo(arquivo: Path, data_antiga: str) -> Path:
    selo = re.sub(r"[^0-9]", "", data_antiga)[:14] or datetime.now().strftime("%Y%m%d%H%M%S")
    selo = f"{selo[:8]}-{selo[8:]}" if len(selo) > 8 else selo
    destino = arquivo.with_name(f"{arquivo.stem}_anterior_{selo}{arquivo.suffix}")
    n = 2
    while destino.exists():
        destino = arquivo.with_name(f"{arquivo.stem}_anterior_{selo}_{n}{arquivo.suffix}")
        n += 1
    arquivo.replace(destino)
    return destino


def rodar(cfg: dict, forcar: bool = False, config_path="config.yaml", pasta_tabela=None, agora: datetime | None = None) -> int:
    p = parametros(cfg)
    pend = pendencias_aval(cfg)
    if pend:                                          # vale também com --forcar: destravar é editar o config, não uma flag
        print("Recusado: o pré-registro está bloqueado até o aval do orientador (nem --forcar passa).\n  - " + "\n  - ".join(pend)
              + "\nDepois do aval: preencha preregistro.aval_orientador, troque 'provisória' por 'decidida' nas declarações e rode de novo.")
        return 1
    caminho_json, caminho_md = Path(p["arquivo_json"]), Path(p["arquivo_md"])
    existentes = [c for c in (caminho_json, caminho_md) if c.exists()]
    if existentes and not forcar:
        print(f"Recusado: já existe pré-registro ({', '.join(str(c) for c in existentes)}). Para substituir, rode com --forcar "
              "(o antigo é guardado com data). Mudar o pré-registro depois de abrir C invalida o teste.")
        return 1
    reg = montar_preregistro(cfg, config_path=config_path, pasta_tabela=pasta_tabela, agora=agora)     # antes de mexer no antigo
    guardados = []
    data_antiga = ""                                  # o mesmo selo de data para o .json e o .md antigos (a data do pré-registro antigo)
    if caminho_json in existentes:
        try:
            data_antiga = json.loads(caminho_json.read_text(encoding="utf-8")).get("data", "")
        except (OSError, ValueError):
            pass
    if existentes and not data_antiga:
        data_antiga = datetime.fromtimestamp(existentes[0].stat().st_mtime).isoformat(timespec="seconds")
    for c in existentes:
        guardados.append(_guardar_antigo(c, data_antiga))
    caminho_json.parent.mkdir(parents=True, exist_ok=True)
    caminho_md.parent.mkdir(parents=True, exist_ok=True)
    caminho_json.write_text(json.dumps(reg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    caminho_md.write_text(texto_markdown(reg), encoding="utf-8", newline="\n")
    print(f"Pré-registro gravado: {caminho_json} e {caminho_md}")
    print(f"  versão {reg['versao_codigo']}; treino: {reg['treino']['n_videos']} vídeos em {reg['treino']['n_preparacoes']} preparações")
    for n, h in reg["hashes_tabela"].items():
        print(f"  {n}: {h[:16]}…")
    print(f"  config (arquivo): {reg['config']['sha256_16']}; seções conferidas: {', '.join(reg['config']['secoes_sha256'])}")
    for g in guardados:
        print(f"  anterior guardado em {g}")
    print(NOTA)
    return 0
