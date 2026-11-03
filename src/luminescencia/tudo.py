"""Comando `tudo`: roda a cadeia de etapas de dados em ordem e carimba a versão em cada relatório.

P6b (v0.12.0): conferência de regressão com tolerância (--gravar-referencia, --conferir, --so-conferir).

Cadeia padrão: consolidar -> pos -> modelar -> tabela -> comparar_modelos -> descritores -> graficos
(`offset` entra depois de `descritores` com --com-offset). Fora da cadeia, por usarem vídeos ou serem lentas:
auditar, roi, extrair. Este módulo não altera os módulos das etapas: só chama o `rodar` de cada um com os
padrões do config.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path

from . import __version__

MARCA_INI = "# --- carimbo luminescencia ---"
MARCA_FIM = "# --- fim carimbo ---"
_RE_CARIMBO = re.compile(re.escape(MARCA_INI) + r".*?" + re.escape(MARCA_FIM) + r"\n?", re.S)

# nome da etapa (igual ao comando do cli) -> módulo e relatório padrão (relativo à pasta de saída)
ETAPAS = {
    "consolidar": ("consolidacao", "consolidacao/relatorio_consolidacao.txt"),
    "pos": ("posprocessamento", "posprocessamento/relatorio_posprocessamento.txt"),
    "modelar": ("modelar", "modelagem/relatorio_modelagem.txt"),
    "tabela": ("tabela", "tabela/relatorio_tabela.txt"),
    "comparar_modelos": ("comparacao_modelos", "modelagem/relatorio_comparacao_modelos.txt"),
    "descritores": ("descritores", "tabela/relatorio_descritores.txt"),
    "offset": ("offset", "offset/relatorio_offset.txt"),
    "graficos": ("graficos", "graficos/graficos_resumo.txt"),
}
CADEIA_PADRAO = ["consolidar", "pos", "modelar", "tabela", "comparar_modelos", "descritores", "graficos"]
NOMES_ETAPAS = CADEIA_PADRAO[:6] + ["offset"] + CADEIA_PADRAO[6:]


def montar_cadeia(com_offset: bool = False, de: str | None = None, ate: str | None = None) -> list[str]:
    """Etapas a rodar, em ordem. `offset` entra após `descritores` se pedido (ou se citado em --de/--ate)."""
    for nome in (de, ate):
        if nome is not None and nome not in ETAPAS:
            raise ValueError(f"etapa desconhecida: {nome!r} (válidas: {', '.join(NOMES_ETAPAS)})")
    cadeia = list(CADEIA_PADRAO)
    if com_offset or "offset" in (de, ate):
        cadeia.insert(cadeia.index("descritores") + 1, "offset")
    i = cadeia.index(de) if de else 0
    j = cadeia.index(ate) if ate else len(cadeia) - 1
    if i > j:
        raise ValueError(f"--de {de} vem depois de --ate {ate} na cadeia ({' -> '.join(cadeia)})")
    return cadeia[i:j + 1]


def _obter_rodar(nome: str):
    modulo = importlib.import_module("." + ETAPAS[nome][0], __package__)
    return modulo.rodar


def _pasta_saida(cfg: dict) -> Path:
    return Path((cfg.get("pastas") or {}).get("saida", "output"))


def _caminho_relatorio(cfg: dict, nome: str) -> Path:
    rel = ((cfg.get("tudo") or {}).get("etapas") or {}).get(nome) or ETAPAS[nome][1]
    return _pasta_saida(cfg) / rel


def hash_config(config_path) -> str:
    """16 primeiros hex do SHA-256 dos bytes do config usado."""
    try:
        return hashlib.sha256(Path(config_path).read_bytes()).hexdigest()[:16]
    except OSError:
        return "indisponivel"


def bloco_carimbo(config_path, agora: datetime | None = None) -> str:
    agora = agora or datetime.now()
    return "\n".join([
        MARCA_INI,
        f"versao_codigo: {__version__}",
        f"config_sha256: {hash_config(config_path)}",
        f"data: {agora:%Y-%m-%d %H:%M}",
        MARCA_FIM,
    ]) + "\n"


def carimbar(relatorio, config_path, agora: datetime | None = None) -> bool:
    """Acrescenta ao fim do relatório o bloco de carimbo, substituindo o anterior (idempotente).
    Devolve False se o relatório não existe."""
    relatorio = Path(relatorio)
    if not relatorio.is_file():
        return False
    bruto = relatorio.read_bytes()
    bom = b"\xef\xbb\xbf" if bruto.startswith(b"\xef\xbb\xbf") else b""
    texto = bruto[len(bom):].decode("utf-8")
    # Normaliza quebras de linha para evitar \r\n no Windows
    texto = texto.replace("\r\n", "\n")
    texto = _RE_CARIMBO.sub("", texto).rstrip("\n")
    texto = (texto + "\n\n" if texto else "") + bloco_carimbo(config_path, agora)
    # Garante que usamos apenas \n para que o teste de bytes sem \r passe no Windows
    relatorio.write_bytes(bom + texto.encode("utf-8"))
    return True


def _rodar_cadeia(cfg, de=None, ate=None, com_offset=False, parar_no_erro=True, config_path=None) -> int:
    """Roda a cadeia. Devolve 0 se todas as etapas rodadas deram 0, senão 1."""
    config_path = config_path or "config.yaml"
    try:
        cadeia = montar_cadeia(com_offset, de, ate)
    except ValueError as erro:
        print(f"Erro: {erro}")
        return 1

    print(f"luminescencia {__version__}: cadeia = {' -> '.join(cadeia)}")
    resultados = []                                   # (etapa, situação, segundos)
    falhou = False
    for k, nome in enumerate(cadeia):
        print(f"\n=== [{k + 1}/{len(cadeia)}] {nome} ===", flush=True)
        t0 = time.perf_counter()
        try:
            codigo = _obter_rodar(nome)(cfg)
            codigo = codigo if isinstance(codigo, int) else 0
        except Exception as erro:                     # uma etapa que quebra conta como falha
            print(f"Erro em '{nome}': {type(erro).__name__}: {erro}")
            codigo = 1
        dt = time.perf_counter() - t0
        if codigo == 0:
            if carimbar(_caminho_relatorio(cfg, nome), config_path):
                situacao = "ok"
            else:
                situacao = "ok (relatório não encontrado: sem carimbo)"
        else:
            situacao = f"FALHOU (código {codigo})"
            falhou = True
        resultados.append((nome, situacao, dt))
        print(f"--- {nome}: {situacao} em {dt:.1f} s")
        if codigo != 0 and parar_no_erro:
            break

    feitas = {n for n, _, _ in resultados}
    print("\nResumo:")
    for nome, situacao, dt in resultados:
        print(f"  {nome:<18} {situacao:<12} {dt:7.1f} s")
    for nome in cadeia:
        if nome not in feitas:
            print(f"  {nome:<18} não executada")
    print(f"  {'total':<18} {'':<12} {sum(r[2] for r in resultados):7.1f} s")
    if falhou:
        print("Resultado: FALHA em " + ", ".join(n for n, s, _ in resultados if s.startswith("FALHOU")))
        return 1
    print("Resultado: tudo ok")
    return 0


# ----------------------------------------------------------------------------------------------------------------
# P6b: conferência de regressão com tolerância
# ----------------------------------------------------------------------------------------------------------------
ARQ_REFERENCIA = Path("tests") / "referencia" / "referencia.json"
REL_REGRESSAO = Path("regressao") / "relatorio_regressao.txt"
CSVS_REFERENCIA = ["tabela/dataset.csv", "tabela/descritores.csv", "modelagem/ajustes.csv",
                   "modelagem/comparacao_modelos.csv"]
STATS = ("mediana", "media", "nan")
RTOL_PADRAO, ATOL_PADRAO = 1e-3, 1e-9
# Números extraídos dos relatórios .txt por regex: etapa -> [(chave, padrão)]. Padrão que não casa só gera aviso.
PADROES_RELATORIO = {
    "modelar": [("ajustes", r"Ajustes:\s*(\d+)"), ("com_modelo", r"com modelo:\s*(\d+)")],
    "comparar_modelos": [("pct_no_limite_superior", r"(\d+(?:[.,]\d+)?)%\s+no limite superior")],
}


def _finito(x):
    """float finito ou None (NaN/inf viram None, para o JSON)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def perfil_csv(caminho) -> dict:
    """Perfil de um CSV: nº de linhas, colunas e, por coluna numérica, mediana, média e contagem de NaN."""
    import pandas as pd

    df = pd.read_csv(caminho)
    numericas = {}
    for col in df.columns:
        serie = df[col]
        if not pd.api.types.is_numeric_dtype(serie):
            continue
        if serie.dtype == bool:
            serie = serie.astype(float)
        numericas[str(col)] = {"mediana": _finito(serie.median()), "media": _finito(serie.mean()),
                               "nan": int(serie.isna().sum())}
    return {"linhas": int(len(df)), "colunas": [str(c) for c in df.columns], "numericas": numericas}


def perfil_relatorio(caminho, padroes) -> tuple[dict, list[str]]:
    """Números extraídos de um relatório .txt. Nunca levanta: padrão que não casa vira aviso."""
    valores, avisos = {}, []
    try:
        texto = Path(caminho).read_text(encoding="utf-8", errors="replace")
    except OSError as erro:
        return valores, [f"relatório ilegível ({Path(caminho).name}): {erro}"]
    for chave, padrao in padroes:
        try:
            m = re.search(padrao, texto)
            valores[chave] = float(m.group(1).replace(",", ".")) if m else None
        except (re.error, ValueError, IndexError):
            valores[chave] = None
        if valores[chave] is None:
            avisos.append(f"padrão '{chave}' não casou em {Path(caminho).name} (ignorado)")
    return {k: v for k, v in valores.items() if v is not None}, avisos


def montar_perfis(cfg) -> tuple[dict, list[str]]:
    """Perfil do output/ atual: {"csv": {rel: perfil}, "relatorios": {etapa: {chave: valor}}} e avisos."""
    saida = _pasta_saida(cfg)
    perfis = {"csv": {}, "relatorios": {}}
    avisos = []
    for rel in CSVS_REFERENCIA:
        arq = saida / rel
        if not arq.is_file():
            avisos.append(f"arquivo não encontrado: {arq}")
            continue
        try:
            perfis["csv"][rel] = perfil_csv(arq)
        except Exception as erro:                       # CSV corrompido não pode quebrar a conferência
            avisos.append(f"não consegui ler {arq}: {type(erro).__name__}: {erro}")
    for nome, padroes in PADROES_RELATORIO.items():
        arq = _caminho_relatorio(cfg, nome)
        if not arq.is_file():
            avisos.append(f"relatório não encontrado: {arq}")
            continue
        valores, av = perfil_relatorio(arq, padroes)
        avisos += av
        perfis["relatorios"][nome] = valores
    return perfis, avisos


def tolerancias(cfg) -> tuple[float, float]:
    sec = cfg.get("tudo") or {}
    return float(sec.get("rtol", RTOL_PADRAO)), float(sec.get("atol", ATOL_PADRAO))


def _compara(ref, atual, rtol, atol):
    """(igual?, diferença relativa). None só é igual a None."""
    if ref is None or atual is None:
        return (ref is None and atual is None), (0.0 if ref is None and atual is None else math.inf)
    dif = abs(atual - ref)
    return dif <= atol + rtol * abs(ref), dif / max(abs(ref), atol, 1e-300)


def comparar_perfis(ref: dict, atual: dict, rtol=RTOL_PADRAO, atol=ATOL_PADRAO) -> dict:
    """Compara dois perfis. Devolve {"n_ok", "numericas": [...], "estruturais": [...], "avisos": [...]}.
    Numéricas: dicts (arquivo, coluna, estat, ref, atual, rel), ordenadas da pior para a melhor."""
    n_ok, numericas, estruturais, avisos = 0, [], [], []
    for rel, p in (ref.get("csv") or {}).items():
        q = (atual.get("csv") or {}).get(rel)
        if q is None:
            estruturais.append(f"{rel}: arquivo ausente na saída atual")
            continue
        if p["linhas"] != q["linhas"]:
            estruturais.append(f"{rel}: nº de linhas {p['linhas']} -> {q['linhas']}")
        faltam = [c for c in p["colunas"] if c not in q["colunas"]]
        novas = [c for c in q["colunas"] if c not in p["colunas"]]
        if faltam:
            estruturais.append(f"{rel}: coluna(s) faltando: {', '.join(faltam)}")
        if novas:
            estruturais.append(f"{rel}: coluna(s) nova(s): {', '.join(novas)}")
        for col, est_ref in p["numericas"].items():
            if col not in q["colunas"]:
                continue
            est_at = q["numericas"].get(col)
            if est_at is None:
                estruturais.append(f"{rel}: coluna '{col}' deixou de ser numérica")
                continue
            for est in STATS:
                v_ref, v_at = est_ref[est], est_at[est]
                if est == "nan":                        # contagem: exata
                    igual, rel_dif = v_ref == v_at, abs(v_at - v_ref) / max(v_ref, 1)
                else:
                    igual, rel_dif = _compara(v_ref, v_at, rtol, atol)
                if igual:
                    n_ok += 1
                else:
                    numericas.append({"arquivo": rel, "coluna": col, "estat": est, "ref": v_ref,
                                      "atual": v_at, "rel": rel_dif})
        for col, est_at in q["numericas"].items():
            if col in p["colunas"] and col not in p["numericas"]:
                estruturais.append(f"{rel}: coluna '{col}' passou a ser numérica")
    for nome, valores in (ref.get("relatorios") or {}).items():
        at = (atual.get("relatorios") or {}).get(nome, {})
        for chave, v_ref in valores.items():
            if chave not in at:
                avisos.append(f"relatório {nome}: '{chave}' não foi encontrado agora (não comparado)")
                continue
            if at[chave] == v_ref:                      # números dos relatórios: igualdade exata
                n_ok += 1
            else:
                numericas.append({"arquivo": f"relatório {nome}", "coluna": chave, "estat": "valor",
                                  "ref": v_ref, "atual": at[chave],
                                  "rel": abs(at[chave] - v_ref) / max(abs(v_ref), 1e-300)})
    numericas.sort(key=lambda d: d["rel"], reverse=True)
    return {"n_ok": n_ok, "numericas": numericas, "estruturais": estruturais, "avisos": avisos}


def texto_relatorio_regressao(res: dict, rtol, atol, ref_meta: dict) -> str:
    def fmt(v):
        return "NaN" if v is None else f"{v:.6g}"
    linhas = ["RELATÓRIO DE REGRESSÃO", "",
              f"Referência: versão {ref_meta.get('versao', '?')}, gravada em {ref_meta.get('data', '?')}",
              f"Tolerância: rtol = {rtol:g}, atol = {atol:g} (nº de linhas, colunas e contagem de NaN: exatos)", "",
              f"OK: {res['n_ok']} comparações dentro da tolerância",
              f"Divergências numéricas: {len(res['numericas'])}",
              f"Divergências estruturais: {len(res['estruturais'])}", ""]
    if res["numericas"]:
        linhas.append(f"Divergências numéricas (as {min(20, len(res['numericas']))} piores):")
        for d in res["numericas"][:20]:
            rel = "inf" if math.isinf(d["rel"]) else f"{d['rel']:.3g}"
            linhas.append(f"  {d['arquivo']} | {d['coluna']} | {d['estat']}: ref {fmt(d['ref'])} -> atual "
                          f"{fmt(d['atual'])} (dif. rel. {rel})")
        linhas.append("")
    if res["estruturais"]:
        linhas.append("Divergências estruturais:")
        linhas += [f"  {e}" for e in res["estruturais"]]
        linhas.append("")
    if res["avisos"]:
        linhas.append("Avisos (não contam como divergência):")
        linhas += [f"  {a}" for a in res["avisos"]]
        linhas.append("")
    ok = not res["numericas"] and not res["estruturais"]
    linhas.append("Resultado: " + ("tudo OK" if ok else "DIVERGÊNCIAS ENCONTRADAS"))
    return "\n".join(linhas) + "\n"


def gravar_ref(cfg, config_path) -> int:
    """Grava tests/referencia/referencia.json a partir do output/ atual."""
    perfis, avisos = montar_perfis(cfg)
    for a in avisos:
        print(f"aviso: {a}")
    if not perfis["csv"]:
        print("Erro: nenhum CSV de referência encontrado no output; nada gravado.")
        return 1
    rtol, atol = tolerancias(cfg)
    ref = {"versao": __version__, "data": f"{datetime.now():%Y-%m-%d %H:%M}",
           "config_sha256": hash_config(config_path), "rtol": rtol, "atol": atol, **perfis}
    ARQ_REFERENCIA.parent.mkdir(parents=True, exist_ok=True)
    with open(ARQ_REFERENCIA, "w", encoding="utf-8", newline="\n") as f:
        json.dump(ref, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"referência gravada: {ARQ_REFERENCIA} ({len(perfis['csv'])} CSV, {len(perfis['relatorios'])} relatórios)")
    print("Atenção: a referência vale o que a saída valia. Confira o output antes de gravá-la.")
    return 0


def conferir_ref(cfg, config_path) -> int:
    """Compara o output/ atual com a referência; grava output/regressao/relatorio_regressao.txt. 0 = tudo OK."""
    try:
        ref = json.loads(ARQ_REFERENCIA.read_text(encoding="utf-8"))
    except (OSError, ValueError) as erro:
        print(f"Erro: não consegui ler a referência {ARQ_REFERENCIA}: {erro}\n"
              "Grave-a antes com: luminescencia tudo --gravar-referencia")
        return 1
    rtol, atol = tolerancias(cfg)
    atual, avisos = montar_perfis(cfg)
    res = comparar_perfis(ref, atual, rtol, atol)
    res["avisos"] = avisos + res["avisos"]
    destino = _pasta_saida(cfg) / REL_REGRESSAO
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "w", encoding="utf-8", newline="\n") as f:
        f.write(texto_relatorio_regressao(res, rtol, atol, ref))
    carimbar(destino, config_path)
    ok = not res["numericas"] and not res["estruturais"]
    print(f"regressão: {res['n_ok']} OK, {len(res['numericas'])} numéricas, {len(res['estruturais'])} estruturais "
          f"-> {destino}")
    print("Resultado da regressão: " + ("tudo OK" if ok else "DIVERGÊNCIAS"))
    return 0 if ok else 1


def rodar(cfg, de=None, ate=None, com_offset=False, parar_no_erro=True, config_path=None,
          conferir=False, gravar_referencia=False, so_conferir=False) -> int:
    """Roda a cadeia e, se pedido, grava a referência ou confere a regressão. 0 = tudo ok."""
    if gravar_referencia and (conferir or so_conferir):
        print("Erro: --gravar-referencia não combina com --conferir/--so-conferir.")
        return 1
    config_path = config_path or "config.yaml"
    if not so_conferir:
        codigo = _rodar_cadeia(cfg, de=de, ate=ate, com_offset=com_offset, parar_no_erro=parar_no_erro,
                               config_path=config_path)
        if codigo != 0:
            return codigo
    if gravar_referencia:
        return gravar_ref(cfg, config_path)
    if conferir or so_conferir:
        return conferir_ref(cfg, config_path)
    return 0
