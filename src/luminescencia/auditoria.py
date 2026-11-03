"""Programa 1 - Auditoria dos vídeos.

Lê a pasta de vídeos e gera um relatório (CSV + texto) e um resumo no terminal.
Não altera nenhum arquivo de vídeo e pode ser rodado sempre que chegarem vídeos novos.

Modos de leitura
  rápido (padrão)  cabeçalho do arquivo + primeiro e último frame. Acha arquivos corrompidos
                   ou truncados e é muito mais rápido.
  completo         lê todos os frames: mede jitter e conta frames idênticos ao anterior.

Velocidade: os vídeos são auditados em paralelo (workers) e o resultado de cada um fica em
um cache. Numa nova rodada, só os vídeos novos ou alterados são lidos de novo.

Detecção da queda: nos primeiros segundos de cada vídeo (config: deteccao_queda) procura o instante
em que a luz de excitação apaga. É só verificação: quando não acha, o vídeo vai para o bloco de revisão.

Revisão manual: erros de digitação no nome, pasta diferente do nome (amostra/número) e queda não
detectada ficam num bloco separado, com a opinião do programa. Nada é renomeado nem movido.

Ordem do resumo: primeiro os vídeos que não puderam ser lidos (ou estão truncados),
depois os números gerais e, por último, em destaque, os vídeos fora do padrão.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import re
import shutil
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from . import __version__
from .curadoria import atualizar_curadoria
from .nomes import ler_nome

CAMPOS_CSV = [
    "arquivo", "caminho", "status", "problemas", "revisar", "nome_sugerido", "avisos", "tipo", "cores",
    "prop_R", "prop_G", "prop_B", "proporcao", "amostra", "video", "grupo", "repetido_com", "repetido_conteudo",
    "fps_container", "fps_real", "n_frames_info", "n_frames_lidos", "largura", "altura",
    "duracao_s", "queda_s", "queda_status", "queda_variacao", "queda_candidato_s", "brilho_inicial", "codec", "hash", "intervalos_longos", "intervalos_curtos", "frames_perdidos_estimados",
    "jitter_std_ms", "frames_duplicados", "frac_duplicados", "fps_efetivo", "maior_repeticao", "exposicao", "tamanho_mb", "modo_leitura", "tempo_s_auditoria",
]
EXTENSOES_PADRAO = [".mp4", ".mov", ".avi", ".mkv", ".m4v"]
PADRAO_EXPOSICAO = re.compile(
    r"(?<![a-z])iso(?![a-z])|exposure|white.?balance|shutter|aperture|f.?number", re.IGNORECASE
)


# ------------------------------------------------------------------ leitura
def listar_videos(pasta: Path, extensoes: list[str], ignorar=(), ignorados: list | None = None) -> list[Path]:
    """Vídeos da pasta (em qualquer nível). Pastas com nome em `ignorar` são puladas;
    os arquivos pulados vão para a lista `ignorados`, se ela for passada."""
    pasta = Path(pasta)
    exts = {e.lower() for e in extensoes}
    nomes_ignorados = {str(n).strip().lower() for n in ignorar if n}
    achados = []
    for p in sorted(pasta.rglob("*")):
        if not (p.is_file() and p.suffix.lower() in exts):
            continue
        partes = {parte.lower() for parte in p.relative_to(pasta).parts[:-1]}
        if partes & nomes_ignorados:
            if ignorados is not None:
                ignorados.append(p)
            continue
        achados.append(p)
    return achados


def _texto_fourcc(valor: float) -> str:
    v = int(valor)
    texto = "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ")
    return texto if texto.isprintable() and texto else "?"


def _ler_exposicao(caminho: Path) -> dict:
    """ISO, exposição e balanço de branco gravados nos metadados (se existirem). Usa ffprobe."""
    exe = shutil.which("ffprobe")
    if not exe:
        return {}
    try:
        saida = subprocess.run(
            [exe, "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(caminho)],
            capture_output=True, text=True, timeout=60,
        )
        dados = json.loads(saida.stdout or "{}")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return {}
    fontes = [dados.get("format", {}).get("tags", {})] + [s.get("tags", {}) for s in dados.get("streams", [])]
    return {k: str(v) for fonte in fontes for k, v in fonte.items() if PADRAO_EXPOSICAO.search(k)}


def _novo_registro(caminho: Path) -> dict:
    r: dict = {c: "" for c in CAMPOS_CSV}
    r.update(arquivo=caminho.name, pasta=caminho.parent.name, status="ok", problemas=[], avisos=[], revisar=[], exposicao_dict={})
    return r


def _marcar_ilegivel(r: dict, motivo: str) -> dict:
    r["status"] = "ilegivel"
    r["problemas"].append(motivo)
    return r


def _comparar_protocolo(r: dict, prot: dict, fps_real: float, fps_container: float, duracao: float):
    fps_esp, fps_tol = prot.get("fps_esperado"), prot.get("fps_tolerancia", 0.5)
    if fps_esp and abs(fps_real - fps_esp) > fps_tol:
        r["problemas"].append(f"FPS {fps_real:.2f} difere do protocolo ({fps_esp} ± {fps_tol})")
    if fps_container and abs(fps_container - fps_real) > 0.5:
        r["avisos"].append(f"FPS do container ({fps_container:.2f}) difere do medido ({fps_real:.2f})")
    dur_esp, dur_tol = prot.get("duracao_esperada_s"), prot.get("duracao_tolerancia_s", 1.0)
    if dur_esp and abs(duracao - dur_esp) > dur_tol:
        r["problemas"].append(f"duração {duracao:.1f} s difere do protocolo ({dur_esp} ± {dur_tol} s)")
    res_esp = prot.get("resolucao_esperada")
    if res_esp and (r["largura"], r["altura"]) != tuple(res_esp):
        r["problemas"].append(f"resolução {r['largura']}x{r['altura']} difere do protocolo ({res_esp[0]}x{res_esp[1]})")


def _anexar_exposicao(r: dict, caminho: Path):
    r["exposicao_dict"] = _ler_exposicao(caminho)
    r["exposicao"] = json.dumps(r["exposicao_dict"], ensure_ascii=False) if r["exposicao_dict"] else ""


def _ler_extremos(cap, n_info: int, primeiro_ok: bool | None = None) -> tuple[bool, bool]:
    """Modo rápido: tenta ler o primeiro frame e um dos 3 últimos informados no cabeçalho.
    Se o primeiro frame já foi lido (pela detecção da queda), passe primeiro_ok."""
    primeiro = cap.read()[0] if primeiro_ok is None else primeiro_ok
    ultimo = False
    for k in (1, 2, 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(n_info - k, 0))
        if cap.read()[0]:
            ultimo = True
            break
    return primeiro, ultimo


NOMES_SERIES = ("B", "G", "R", "T")      # médias dos canais B, G, R e média dos 5% pixels mais claros (cinza)


def _brilhos_quadro(quadro, fracao: float) -> list[float]:
    """Quatro medidas de brilho do frame (ver NOMES_SERIES).

    Cada canal é medido separado porque a luz de excitação e o brilho da amostra têm cores diferentes:
    numa amostra verde o brilho do pó quase não muda quando a luz apaga, mas o reflexo da lâmpada
    some nos canais azul e vermelho. Olhar só o brilho total esconderia essa queda.
    """
    if quadro.ndim == 2:
        quadro = cv2.cvtColor(quadro, cv2.COLOR_GRAY2BGR)
    if min(quadro.shape[:2]) >= 64:
        quadro = cv2.resize(quadro, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    b, g, r = (float(x) for x in cv2.mean(quadro)[:3])
    v = cv2.cvtColor(quadro, cv2.COLOR_BGR2GRAY).ravel()
    k = max(1, int(len(v) * fracao))
    return [b, g, r, float(np.partition(v, len(v) - k)[len(v) - k:].mean())]


def _degrau(serie: np.ndarray, k: int, larg: int = 3) -> tuple[float, float]:
    """Queda no frame k: média dos `larg` frames antes menos a dos `larg` depois (o próprio frame k fica
    de fora, pois pode ser um frame de transição). Retorna (queda absoluta, queda relativa)."""
    antes = serie[max(0, k - larg):k].mean()
    depois = serie[k + 1:k + 1 + larg].mean()
    return float(antes - depois), float((antes - depois) / max(antes, 5.0))


def detectar_queda(cap, fps: float, cfg_q: dict) -> dict:
    """Lê os primeiros frames e procura o instante em que a luz de excitação apaga.

    Para cada canal (B, G, R) e para os pixels mais claros, mede o degrau de brilho em cada frame
    (média dos 3 frames antes contra os 3 depois) e fica com o maior. Vale se a queda for de pelo
    menos `queda_minima` (fração do brilho anterior) e de pelo menos 2 níveis de cinza.
    O instante é o do primeiro frame que já mudou (precisão de ±1 frame).
    Retorna: frames (lidos), status (detectada | fora_da_janela | nao_detectada | nao_avaliada),
    queda_s, candidato_s (instante da maior queda na janela, mesmo sem valer), variacao,
    serie (brilhos lidos, para calibrar) e motivo (texto quando o status não é "detectada").
    """
    ini, fim = float(cfg_q.get("janela_inicio_s", 0)), float(cfg_q.get("janela_fim_s", 1.0))
    ctx, frac, minimo = float(cfg_q.get("contexto_s", 0.5)), float(cfg_q.get("fracao_brilho", 0.05)), float(cfg_q.get("queda_minima", 0.2))
    abs_min = 2.0
    res = {"frames": 0, "status": "nao_avaliada", "queda_s": "", "candidato_s": "", "variacao": "", "serie": "", "motivo": ""}

    medidas = []
    limite = int(np.ceil((fim + ctx) * fps)) + 1 if fps > 0 else 30
    for _ in range(limite):
        ok, quadro = cap.read()
        if not ok:
            break
        medidas.append(_brilhos_quadro(quadro, frac))
    res["frames"] = n = len(medidas)
    if n == 0:
        return res
    S = np.array(medidas)
    res["serie"] = "|".join(f"{nome}:" + ",".join(f"{x:.1f}" for x in S[:, i]) for i, nome in enumerate(NOMES_SERIES))
    if fps <= 0:
        res["motivo"] = "FPS do container indisponível: não foi possível localizar a queda no tempo"
        return res
    if n < 5:
        res["status"] = "nao_detectada"
        res["motivo"] = f"só {n} frame(s) lido(s): poucos para procurar a queda"
        return res

    candidatos = []                                   # (queda relativa, frame k, série)
    for k in range(1, n - 2):
        for i in range(S.shape[1]):
            absoluta, rel = _degrau(S[:, i], k)
            if absoluta >= abs_min:
                candidatos.append((rel, k, i))
    na_janela = [c for c in candidatos if ini <= c[1] / fps <= fim]
    if na_janela:
        res["candidato_s"] = round(max(na_janela)[1] / fps, 3)
        res["variacao"] = round(max(na_janela)[0], 3)

    for grupo, status in ((na_janela, "detectada"), ([c for c in candidatos if c not in na_janela], "fora_da_janela")):
        validos = [c for c in grupo if c[0] >= minimo]
        if validos:
            rel, k, i = max(validos)
            res.update(status=status, queda_s=round(k / fps, 3), variacao=round(rel, 3), candidato_s=round(k / fps, 3))
            if status == "fora_da_janela":
                res["motivo"] = f"queda encontrada em {k / fps:.2f} s, fora da janela de {ini:g} a {fim:g} s"
            return res
    res["status"] = "nao_detectada"
    res["motivo"] = (f"nenhuma queda de pelo menos {minimo:.0%} entre {ini:g} e {fim:g} s "
                     f"(maior: {float(res['variacao'] or 0):.0%}, perto de {res['candidato_s'] or '?'} s). "
                     "A luz pode já estar apagada no primeiro frame, ou a queda é gradual demais")
    return res


def auditar_video(caminho: Path, cfg: dict, completo: bool = False) -> dict:
    prot = cfg.get("protocolo", {})
    aud = cfg.get("auditoria", {})
    jitter_tol = aud.get("jitter_tolerancia", 0.5)
    frac_truncado = aud.get("truncado_fracao", 0.98)

    r = _novo_registro(caminho)
    r["tamanho_mb"] = round(caminho.stat().st_size / 1e6, 2)

    campos, erro = ler_nome(caminho.name, cfg.get("nomes", {}))
    if campos:
        for chave in ("tipo", "cores", "prop_R", "prop_G", "prop_B", "proporcao", "amostra", "video"):
            r[chave] = campos.get(chave, "")
        if campos.get("aviso"):
            r["avisos"].append(campos["aviso"])
        if campos.get("correcoes"):
            r["nome_sugerido"] = campos["nome_sugerido"]
            r["revisar"].append({"tipo": "nome", "texto": (
                f"nome com erro de digitação ({'; '.join(campos['correcoes'])}); lido como "
                f"{campos['proporcao']}, amostra {campos['amostra']}, vídeo {campos['video']}. "
                f"Nome sugerido: {campos['nome_sugerido']}")})
    else:
        r["problemas"].append(erro)

    if caminho.stat().st_size == 0:
        return _marcar_ilegivel(r, "arquivo vazio")

    cap = cv2.VideoCapture(str(caminho))
    if not cap.isOpened():
        return _marcar_ilegivel(r, "não foi possível abrir (arquivo corrompido ou codec não suportado)")

    fps_container = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    n_info = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    r.update(
        fps_container=round(fps_container, 3), n_frames_info=n_info,
        largura=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), altura=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        codec=_texto_fourcc(cap.get(cv2.CAP_PROP_FOURCC)),
    )

    dq = detectar_queda(cap, fps_container, cfg.get("deteccao_queda", {}))
    r.update(queda_s=dq["queda_s"], queda_status=dq["status"] if dq["frames"] else "", queda_variacao=dq["variacao"],
             queda_candidato_s=dq["candidato_s"], brilho_inicial=dq["serie"])
    if dq["frames"] and dq["status"] != "detectada":
        r["revisar"].append({"tipo": "queda", "texto": dq["motivo"]})

    # ------------------------------------------------ modo rápido
    if not completo and n_info > 0:
        r["modo_leitura"] = "rápido"
        primeiro, ultimo = _ler_extremos(cap, n_info, primeiro_ok=dq["frames"] > 0)
        cap.release()
        if not primeiro:
            return _marcar_ilegivel(r, "nenhum frame pôde ser decodificado")
        if not ultimo:
            r["status"] = "truncado"
            r["problemas"].append(
                f"possível arquivo truncado/corrompido: o final do vídeo (frame ~{n_info}, informado no cabeçalho) não pôde ser lido"
            )
        duracao = n_info / fps_container if fps_container > 0 else 0.0
        r.update(fps_real=round(fps_container, 3), duracao_s=round(duracao, 3))
        _comparar_protocolo(r, prot, fps_container, fps_container, duracao)
        _anexar_exposicao(r, caminho)
        return r

    # ------------------------------------------------ modo completo (todos os frames)
    r["modo_leitura"] = "completo"
    cap.release()                                   # a detecção da queda já leu os primeiros frames: reabre
    cap = cv2.VideoCapture(str(caminho))
    marcas, falhas, duplicados, anterior, seguidos, maior_rep = [], 0, 0, None, 0, 0
    while cap.grab():
        ok, quadro = cap.retrieve()
        if not ok:
            falhas += 1
            continue
        if anterior is not None and quadro.shape == anterior.shape and np.array_equal(quadro, anterior):
            duplicados += 1
            seguidos += 1
            maior_rep = max(maior_rep, seguidos)
        else:
            seguidos = 0
        anterior = quadro
        marcas.append(cap.get(cv2.CAP_PROP_POS_MSEC))   # instante do frame segundo o container
    cap.release()

    n = len(marcas)
    r["n_frames_lidos"] = n
    if n == 0:
        return _marcar_ilegivel(r, "nenhum frame pôde ser decodificado")
    if falhas:
        r["problemas"].append(f"{falhas} frame(s) não puderam ser decodificados")
    if n_info > 0 and n < n_info * frac_truncado:
        r["status"] = "truncado"
        r["problemas"].append(f"possível arquivo truncado/corrompido: leu {n} de {n_info} frames")

    # Frames idênticos ao anterior: só medidos aqui. Se são problema ou o normal do lote, quem decide é
    # a comparação entre vídeos (_checar_duplicados), porque nos seus vídeos quase todos têm ~47%.
    frac_dup = duplicados / max(n - 1, 1)
    r.update(frames_duplicados=duplicados, frac_duplicados=round(frac_dup, 4), maior_repeticao=maior_rep)

    # tempo e jitter (a partir dos timestamps do container)
    t = np.array(marcas, dtype=float) / 1000.0
    dt = np.diff(t) if n >= 3 else np.array([])
    dt_med = float(np.median(dt)) if len(dt) else 0.0
    if dt_med > 0 and t[-1] > t[0]:
        fps_real = (n - 1) / (t[-1] - t[0])
        longos = dt > (1 + jitter_tol) * dt_med
        curtos = dt < (1 - jitter_tol) * dt_med
        perdidos = int(np.sum(np.maximum(np.round(dt[longos] / dt_med) - 1, 0)))
        duracao = float(t[-1] - t[0] + dt_med)
        r.update(
            intervalos_longos=int(longos.sum()), intervalos_curtos=int(curtos.sum()),
            frames_perdidos_estimados=perdidos, jitter_std_ms=round(float(np.std(dt)) * 1000, 2),
        )
        anomalos = int(longos.sum() + curtos.sum())
        if anomalos > 0.02 * len(dt):
            r["problemas"].append(
                f"FPS variável ou jitter alto: {int(longos.sum())} intervalos longos "
                f"(~{perdidos} frames perdidos) e {int(curtos.sum())} curtos (possíveis duplicados)"
            )
        elif anomalos:
            r["avisos"].append(f"{anomalos} intervalos irregulares entre frames (~{perdidos} perdidos)")
    else:
        fps_real = fps_container
        duracao = n / fps_container if fps_container > 0 else 0.0
        r["avisos"].append("timestamps do container indisponíveis: FPS real e jitter não medidos")
    r.update(fps_real=round(float(fps_real), 3), duracao_s=round(duracao, 3),
             fps_efetivo=round(float(fps_real) * (1 - frac_dup), 2))

    _comparar_protocolo(r, prot, fps_real, fps_container, duracao)
    _anexar_exposicao(r, caminho)
    return r


# ------------------------------------------------------------------ lote
def _moda(valores):
    return Counter(valores).most_common(1)[0][0] if valores else None


def _hash_arquivo(caminho: Path) -> str:
    h = hashlib.sha1()
    with caminho.open("rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def _comparar_lote(resultados: list[dict], cfg: dict, pasta: Path) -> list[str]:
    """Alertas que dependem do conjunto de vídeos (a moda funciona como alerta secundário)."""
    alertas = []
    prot = cfg.get("protocolo", {})
    lidos = [r for r in resultados if r["status"] != "ilegivel"]
    if not lidos:
        return alertas

    fps_moda = _moda([round(r["fps_real"], 1) for r in lidos])
    fps_esp, fps_tol = prot.get("fps_esperado"), prot.get("fps_tolerancia", 0.5)
    if fps_esp and fps_moda is not None and abs(fps_moda - fps_esp) > fps_tol:
        alertas.append(
            f"A maioria dos vídeos tem ~{fps_moda} fps, mas o protocolo pede {fps_esp}. "
            "Pode ser o lote inteiro gravado com a configuração errada (ou o protocolo no config.yaml está desatualizado)."
        )

    if not prot.get("resolucao_esperada"):
        moda = _moda([(r["largura"], r["altura"]) for r in lidos])
        for r in lidos:
            if moda and (r["largura"], r["altura"]) != moda:
                r["problemas"].append(
                    f"resolução {r['largura']}x{r['altura']} difere da maioria ({moda[0]}x{moda[1]})"
                )
    moda_codec = _moda([r["codec"] for r in lidos])
    for r in lidos:
        if r["codec"] != moda_codec:
            r["avisos"].append(f"codec {r['codec']} difere da maioria ({moda_codec})")

    chaves = sorted({k for r in lidos for k in r["exposicao_dict"]})
    if not chaves:
        motivo = "ffprobe não encontrado" if not shutil.which("ffprobe") else "os vídeos não trazem esses campos"
        alertas.append(
            f"Exposição automática não pôde ser conferida (ISO, exposição, balanço de branco): {motivo}. "
            "Arquivos AVI gravados pelo OpenCV não guardam esses valores: registre-os no software de captura."
        )
    for chave in chaves:
        moda = _moda([r["exposicao_dict"].get(chave) for r in lidos if chave in r["exposicao_dict"]])
        for r in lidos:
            valor = r["exposicao_dict"].get(chave)
            if valor is not None and valor != moda:
                r["problemas"].append(f"{chave} = {valor} difere da maioria do lote ({moda})")

    alertas += _checar_duplicados(lidos, cfg)
    alertas += _checar_queda_do_lote(lidos, cfg)
    _checar_pastas(resultados)
    _atribuir_grupos(resultados, cfg)
    alertas += _checar_combinacoes(resultados, cfg, pasta)

    for r in resultados:
        if r["status"] == "ok" and r["problemas"]:
            r["status"] = "fora_do_padrao"
        elif r["status"] == "ok" and r["revisar"]:
            r["status"] = "revisar"
    return alertas


def _checar_duplicados(lidos: list[dict], cfg: dict) -> list[str]:
    """Frames idênticos ao anterior (só no modo completo).

    Se quase todos os vídeos têm a mesma fração, é uma característica do captador: vira um alerta do lote
    e só os vídeos que fogem da maioria viram problema. Se a fração for pequena em todos, nada a avisar.
    """
    aud = cfg.get("auditoria", {})
    tol_lote, desvio = aud.get("duplicados_tolerancia", 0.05), aud.get("duplicados_desvio", 0.03)
    medidos = [r for r in lidos if r.get("frac_duplicados") not in ("", None)]
    if not medidos:
        return []
    mediana = float(np.median([r["frac_duplicados"] for r in medidos]))
    alertas = []
    if mediana > tol_lote:
        fps = float(np.median([r["fps_real"] for r in medidos]))
        alertas.append(
            f"Em geral {mediana:.0%} dos frames são idênticos ao anterior (mediana de {len(medidos)} vídeos): "
            f"só ~{fps * (1 - mediana):.1f} frames novos por segundo, embora o arquivo tenha {fps:g} fps. "
            "É um padrão do lote (captador), não de vídeos isolados; a extração deve contar isso. "
            "Só os vídeos que fogem da maioria são marcados como problema."
        )
    for r in medidos:
        dif = r["frac_duplicados"] - mediana
        if abs(dif) > desvio:
            r["problemas"].append(
                f"{r['frames_duplicados']} frames ({r['frac_duplicados']:.0%}) idênticos ao anterior, "
                f"contra {mediana:.0%} na maioria dos vídeos"
                + (f" (maior sequência de repetidos: {r['maior_repeticao']})" if r.get("maior_repeticao") else "")
            )
    return alertas


def _checar_queda_do_lote(lidos: list[dict], cfg: dict) -> list[str]:
    """O instante da queda deve ser parecido entre os vídeos. Quem foge da maioria vai para a revisão."""
    desvio = float(cfg.get("deteccao_queda", {}).get("desvio_lote_s", 0.25))
    achados = [r for r in lidos if r.get("queda_status") == "detectada" and r.get("queda_s") != ""]
    if len(achados) < 10:
        return []
    mediana = float(np.median([r["queda_s"] for r in achados]))
    for r in achados:
        if abs(r["queda_s"] - mediana) > desvio:
            r["revisar"].append({"tipo": "queda", "texto": (
                f"queda detectada em {r['queda_s']:.2f} s, longe da maioria dos vídeos ({mediana:.2f} s ± {desvio:g} s)")})
    return [f"Queda da luz detectada em {len(achados)} vídeos; instante mediano {mediana:.2f} s "
            f"(menor {min(r['queda_s'] for r in achados):.2f} s, maior {max(r['queda_s'] for r in achados):.2f} s)."]


def _atribuir_grupos(resultados: list[dict], cfg: dict):
    """Coluna `grupo` (divisão treino/validação pela letra da amostra; só para conferência visual).
    Vídeo com pasta diferente do nome fica 'indefinido' até você decidir qual é a letra certa."""
    divisao = cfg.get("divisao") or {}
    letra_para_grupo = {str(l).upper(): g for g, letras in divisao.items() for l in (letras or [])}
    for r in resultados:
        if r["tipo"] != "amostra":
            continue
        if any(i["tipo"] == "pasta" for i in r["revisar"]):
            r["grupo"] = "indefinido"
        else:
            r["grupo"] = letra_para_grupo.get(r["amostra"], "")


PADRAO_PASTA_AMOSTRA = re.compile(r"^amostras?[\s_-]*([a-c])$", re.IGNORECASE)


def _rotulo_da_pasta(caminho: str) -> tuple[str | None, int | None]:
    """Amostra (A/B/C) e número do vídeo indicados pelas pastas, ex.: 'Amostras C/4/arq.avi' -> ('C', 4)."""
    amostra = video = None
    for parte in caminho.split("/")[:-1]:
        m = PADRAO_PASTA_AMOSTRA.match(parte.strip())
        if m:
            amostra = m.group(1).upper()
        elif parte.strip().isdigit():
            video = int(parte.strip())
    return amostra, video


def _checar_pastas(resultados: list[dict]):
    """Compara a pasta (amostra/número) com o que o nome do arquivo diz.

    Não altera nada: só acrescenta um item de revisão com a opinião do programa sobre o que o
    arquivo provavelmente é, olhando quais combinações (proporção + amostra + vídeo) já existem.
    """
    amostras = [r for r in resultados if r["tipo"] == "amostra"]
    ocupado: dict = {}
    for r in amostras:
        ocupado.setdefault((r["proporcao"], r["amostra"], r["video"]), []).append(r["caminho"])

    for r in amostras:
        p_amostra, p_video = _rotulo_da_pasta(r["caminho"])
        if p_amostra is None and p_video is None:
            continue
        dif = []
        if p_amostra is not None and p_amostra != r["amostra"]:
            dif.append(f"amostra: pasta {p_amostra}, nome {r['amostra']}")
        if p_video is not None and p_video != r["video"]:
            dif.append(f"vídeo: pasta {p_video}, nome {r['video']}")
        if not dif:
            continue

        pelo_nome = (r["proporcao"], r["amostra"], r["video"])
        pela_pasta = (r["proporcao"], p_amostra or r["amostra"], p_video or r["video"])
        outros = lambda chave: [c for c in ocupado.get(chave, []) if c != r["caminho"]]
        livre_nome, livre_pasta = not outros(pelo_nome), not outros(pela_pasta)
        desc = lambda chave: f"{chave[0]}, amostra {chave[1]}, vídeo {chave[2]}"

        if livre_pasta and not livre_nome:
            opiniao = (f"provavelmente o NOME está errado: a combinação da pasta ({desc(pela_pasta)}) está livre, "
                       f"enquanto a do nome já existe em {outros(pelo_nome)[0]}")
            sugerir = True
        elif livre_nome and not livre_pasta:
            opiniao = (f"provavelmente o arquivo está na PASTA errada: a combinação do nome ({desc(pelo_nome)}) está livre, "
                       f"enquanto a da pasta já existe em {outros(pela_pasta)[0]}")
            sugerir = False
        elif livre_nome and livre_pasta:
            opiniao = ("as duas hipóteses são possíveis (nome errado ou pasta errada): nenhuma das duas combinações "
                       "está ocupada. Só conferindo o vídeo para decidir")
            sugerir = True
        else:
            opiniao = (f"as duas combinações já existem ({outros(pelo_nome)[0]} e {outros(pela_pasta)[0]}): "
                       "pode ser um vídeo extra ou uma regravação")
            sugerir = False

        if sugerir and not r["nome_sugerido"]:
            base = Path(r["arquivo"])
            novo = re.sub(r"v\d+[a-c]$", f"v{pela_pasta[2]}{pela_pasta[1].lower()}", base.stem, flags=re.IGNORECASE)
            r["nome_sugerido"] = novo + base.suffix
        r["revisar"].append({"tipo": "pasta", "texto": f"pasta e nome não batem ({'; '.join(dif)}); {opiniao}"})


def _checar_combinacoes(resultados: list[dict], cfg: dict, pasta: Path) -> list[str]:
    """Vídeos repetidos (mesma proporção, amostra e número) e, se pedido, vídeos que faltam."""
    # arquivos com pasta diferente do nome já estão no bloco de revisão: não entram como "repetidos"
    amostras = [r for r in resultados if r["tipo"] == "amostra" and not any(i["tipo"] == "pasta" for i in r["revisar"])]
    grupos: dict = {}
    for r in amostras:
        grupos.setdefault((r["proporcao"], r["amostra"], r["video"]), []).append(r)
    repetidos = {k: v for k, v in grupos.items() if len(v) > 1}

    alertas, identicos, diferentes = [], [], []
    for chave, membros in sorted(repetidos.items()):
        try:
            hashes = {m.get("hash") or _hash_arquivo(Path(pasta) / m["caminho"]) for m in membros}
            veredito = "idêntico" if len(hashes) == 1 else "diferente"
        except OSError:
            veredito = "não comparado"
        for m in membros:
            outros = [o["caminho"] for o in membros if o is not m]
            m["repetido_com"] = " | ".join(outros)
            m["repetido_conteudo"] = veredito
            if veredito == "idêntico":
                m["avisos"].append(f"cópia idêntica de {outros[0]}")
                for item in m["revisar"]:
                    if item["tipo"] == "nome":
                        item["texto"] += f". Existe cópia idêntica ({outros[0]}): provavelmente este arquivo é só uma duplicata"
            else:
                m["problemas"].append(f"mesma combinação de {', '.join(outros)} com conteúdo {veredito}")
        (identicos if veredito == "idêntico" else diferentes).append((chave, [m["caminho"] for m in membros]))

    if repetidos:
        alertas.append(
            f"{len(repetidos)} combinação(ões) (proporção + amostra + vídeo) aparecem mais de uma vez: "
            f"{len(identicos)} com arquivos idênticos (cópias), {len(diferentes)} com conteúdo DIFERENTE/não comparado. "
            "Lista completa na coluna repetido_com do CSV."
        )
        for (prop, am, vid), caminhos in diferentes[:10]:
            alertas.append(f"conteúdo diferente: {prop}, amostra {am}, vídeo {vid}: {', '.join(caminhos)}")
        for (prop, am, vid), caminhos in identicos[:3]:
            alertas.append(f"cópia idêntica (exemplo): {prop}, amostra {am}, vídeo {vid}: {', '.join(caminhos)}")

    if cfg.get("auditoria", {}).get("verificar_completude"):
        prot = cfg.get("protocolo", {})
        letras = prot.get("amostras", ["A", "B", "C"])
        n_videos = prot.get("videos_por_amostra", 5)
        esperados = {(p, a, v) for p in {r["proporcao"] for r in amostras} for a in letras for v in range(1, n_videos + 1)}
        faltam = sorted(esperados - set(grupos))
        if faltam:
            lista = ", ".join(f"{p} {a}{v}" for p, a, v in faltam[:12])
            alertas.append(f"Faltam {len(faltam)} vídeo(s) para completar as combinações encontradas: {lista}"
                           + (" ..." if len(faltam) > 12 else ""))
    controles = [r for r in resultados if r["tipo"] == "controle"]
    if not controles:
        alertas.append("Nenhum vídeo de controle (referência sem luminescência) encontrado na pasta.")
    elif len(controles) > 1:
        alertas.append(f"{len(controles)} vídeos de controle encontrados: {', '.join(r['caminho'] for r in controles)}")
    return alertas


# ------------------------------------------------------------------ cache
def _assinatura(cfg: dict, completo: bool) -> str:
    base = {k: cfg.get(k, {}) for k in ("protocolo", "auditoria", "nomes", "deteccao_queda")}
    base.update(completo=completo, versao=__version__)
    return hashlib.sha1(json.dumps(base, sort_keys=True, default=str).encode()).hexdigest()


def _carregar_cache(caminho: Path, assinatura: str) -> dict:
    try:
        dados = json.loads(caminho.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return dados.get("videos", {}) if dados.get("assinatura") == assinatura else {}


def _gravar_cache(caminho: Path, assinatura: str, videos: dict):
    try:
        caminho.write_text(json.dumps({"assinatura": assinatura, "videos": videos}, default=str), encoding="utf-8")
    except OSError:
        pass


# ------------------------------------------------------------------ relatório
def montar_resumo(resultados: list[dict], alertas: list[str], pasta: Path, cor: bool = False, extra: str = "") -> str:
    def c(texto, codigo):
        return f"\033[{codigo}m{texto}\033[0m" if cor else texto

    graves = [r for r in resultados if r["status"] in ("ilegivel", "truncado")]
    fora = [r for r in resultados if r["status"] == "fora_do_padrao"]
    rever = [r for r in resultados if r["revisar"]]
    ok = [r for r in resultados if r["status"] == "ok"]
    com_aviso = [r for r in resultados if r["avisos"]]
    L = [f"AUDITORIA DE VÍDEOS  |  {datetime.now():%d/%m/%Y %H:%M}  |  código v{__version__}", f"Pasta: {pasta}"]
    if extra:
        L.append(extra)
    L.append("")

    if graves:
        L.append(c(f"!! {len(graves)} VÍDEO(S) ILEGÍVEIS, CORROMPIDOS OU TRUNCADOS", "1;31"))
        for r in graves:
            L.append(c(f"   - {r['caminho'] or r['arquivo']} [{r['status']}]: {'; '.join(r['problemas'])}", "31"))
    else:
        L.append(c("Todos os vídeos puderam ser lidos e nenhum está truncado.", "32"))

    L += ["", f"Total: {len(resultados)} vídeos  |  ok: {len(ok)}  |  para revisar: {sum(r['status'] == 'revisar' for r in resultados)}  |  fora do padrão: {len(fora)}  |  "
              f"ilegíveis/truncados: {len(graves)}  |  com avisos: {len(com_aviso)}"]
    fps = [r["fps_real"] for r in resultados if r["fps_real"] != ""]
    if fps:
        L.append(f"FPS: mín {min(fps):.2f}  |  mediana {np.median(fps):.2f}  |  máx {max(fps):.2f}")

    if alertas:
        L += ["", c("ALERTAS DO LOTE", "1;33")] + [c(f"   * {a}", "33") for a in alertas]
    if com_aviso:
        L += ["", f"Avisos (não críticos), {len(com_aviso)} vídeo(s); detalhes no CSV."]

    if rever:
        L += ["", c(f"PARA REVISÃO MANUAL ({len(rever)} vídeo(s)): nada foi alterado; a decisão é sua", "1;36")]
        for r in rever:
            L.append(c(f"   - {r['caminho'] or r['arquivo']}", "36"))
            L += [c(f"       [{i['tipo']}] {i['texto']}", "36") for i in r["revisar"]]

    if fora:
        L += ["", c(f"VÍDEOS FORA DO PADRÃO ({len(fora)})", "1;31;47" if cor else "")]
        for r in fora:
            L.append(c(f"   - {r['caminho'] or r['arquivo']}", "1;31"))
            L += [c(f"       {p}", "31") for p in r["problemas"]]
    else:
        L += ["", c("Nenhum vídeo fora do padrão.", "32")]
    return "\n".join(L)


def _gravar_csv(resultados: list[dict], caminho: Path):
    with caminho.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS_CSV, extrasaction="ignore")
        w.writeheader()
        for r in resultados:
            w.writerow({**r, "problemas": " | ".join(r["problemas"]), "avisos": " | ".join(r["avisos"]),
                        "revisar": " | ".join(f"[{i['tipo']}] {i['texto']}" for i in r["revisar"])})


def _formatar_tempo(seg: float) -> str:
    return f"{int(seg // 60)}min{int(seg % 60):02d}s" if seg >= 60 else f"{seg:.1f}s"


def auditar_pasta(pasta, cfg: dict, saida=None, cor: bool = False, completo: bool = False,
                  workers: int | None = None, refazer: bool = False) -> dict | None:
    inicio = time.perf_counter()
    pasta = Path(pasta)
    if not pasta.is_dir():
        raise NotADirectoryError(f"Pasta de vídeos não encontrada: {pasta.resolve()}")
    nomes = cfg.get("nomes", {})
    ignorar = cfg.get("auditoria", {}).get("pastas_ignoradas") or []
    ignorados: list[Path] = []
    videos = listar_videos(pasta, nomes.get("extensoes", EXTENSOES_PADRAO), ignorar, ignorados)
    if not videos:
        print(f"Nenhum vídeo encontrado em {pasta.resolve()}")
        return None

    workers = max(1, int(workers or cfg.get("auditoria", {}).get("workers", 6)))
    pasta_saida = Path(saida or Path(cfg.get("pastas", {}).get("saida", "output")) / "auditoria")
    pasta_saida.mkdir(parents=True, exist_ok=True)
    cache_path = pasta_saida / "cache_auditoria.json"
    assinatura = _assinatura(cfg, completo)
    cache = {} if refazer else _carregar_cache(cache_path, assinatura)

    rel = lambda v: v.relative_to(pasta).as_posix()
    por_caminho, pendentes, do_cache = {}, [], 0
    for v in videos:
        st, item = v.stat(), cache.get(rel(v))
        if item and item.get("tamanho") == st.st_size and abs(item.get("mtime", 0) - st.st_mtime) < 1:
            por_caminho[rel(v)] = copy.deepcopy(item["registro"])
            do_cache += 1
        else:
            pendentes.append(v)

    def tarefa(v):
        t0 = time.perf_counter()
        try:
            r = auditar_video(v, cfg, completo)
        except Exception as erro:                                  # um vídeo ruim não derruba o lote
            r = _marcar_ilegivel(_novo_registro(v), f"erro inesperado: {type(erro).__name__}: {erro}")
        r["caminho"] = rel(v)
        try:
            r["hash"] = _hash_arquivo(v) if v.stat().st_size else ""
        except OSError:
            r["hash"] = ""
        r["tempo_s_auditoria"] = round(time.perf_counter() - t0, 2)
        return v, r

    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futuros = [pool.submit(tarefa, v) for v in pendentes]
        for f in tqdm(as_completed(futuros), total=len(futuros), desc="Auditando", unit="vídeo", disable=not pendentes):
            v, r = f.result()
            por_caminho[rel(v)] = r
        pool.shutdown()
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        # guarda o que já foi lido (inclusive se você interromper com Ctrl+C), antes das comparações de lote
        novo_cache = {}
        for v in videos:
            r = por_caminho.get(rel(v))
            if r and r["status"] != "ilegivel":
                st = v.stat()
                novo_cache[rel(v)] = {"tamanho": st.st_size, "mtime": st.st_mtime, "registro": copy.deepcopy(r)}
        _gravar_cache(cache_path, assinatura, novo_cache)

    resultados = [por_caminho[rel(v)] for v in videos]
    alertas = _comparar_lote(resultados, cfg, pasta)

    extra = (f"Leitura {'completa' if completo else 'rápida'}  |  em paralelo: {workers}  |  do cache: {do_cache}  |  "
             f"lidos agora: {len(pendentes)}  |  ignorados (pasta ignorada): {len(ignorados)}  |  "
             f"tempo: {_formatar_tempo(time.perf_counter() - inicio)}")
    csv_path, txt_path = pasta_saida / "relatorio_auditoria.csv", pasta_saida / "relatorio_auditoria.txt"
    _gravar_csv(resultados, csv_path)
    txt_path.write_text(montar_resumo(resultados, alertas, pasta, cor=False, extra=extra), encoding="utf-8")

    cur_cfg = cfg.get("curadoria") or {}
    cur = atualizar_curadoria(resultados, cur_cfg.get("arquivo", "curadoria.csv"), cur_cfg.get("motivos"))

    print()
    print(montar_resumo(resultados, alertas, pasta, cor=cor, extra=extra))
    print(f"\nRelatório salvo em:\n  {csv_path}\n  {txt_path}")
    print(f"\nCuradoria: {cur['arquivo']}  |  {cur['total']} vídeos  |  novos: {cur['novos']}  |  "
          f"ausentes: {cur['ausentes']}  |  pendentes: {cur['pendentes']}")
    if cur["erro"]:
        print(f"  ATENÇÃO: {cur['erro']}")
    for aviso in cur["avisos"][:10]:
        print(f"  valor inválido na curadoria: {aviso}")
    return {"resultados": resultados, "alertas": alertas, "csv": csv_path, "txt": txt_path, "curadoria": cur}
