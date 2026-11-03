"""Programa 4 - Consolidação.

Entrada: output/extracao/ (um CSV por vídeo + extracao_resumo.csv) e o curadoria.csv.
Saída (output/consolidacao/):
  videos.csv                 uma linha por vídeo aceito: identificação, grupo, níveis inicial/final brutos e corrigidos
  series.csv                 formato longo: uma linha por quadro novo de cada vídeo aceito (para o pós-processamento)
  relatorio_consolidacao.txt quem entrou, quem ficou de fora e por quê, e a conferência da correção do preto

Quem entra
  incluir = sim                                   entra
  incluir = pendente e status_auditoria = ok      entra (consolidacao.incluir_pendentes_ok; ainda não foi decidido, mas não há alerta)
  incluir = pendente e status_auditoria != ok     fica de fora até o usuário decidir
  incluir = nao, e o vídeo de controle            ficam de fora
O nome vem de nome_aprovado (curadoria) e, se vazio, do nome do arquivo. Nenhum arquivo é renomeado ou movido.

Correção do preto por vídeo (consolidacao.piso: video)
  O preto da câmera muda de uma sessão de gravação para outra (os cantos do vídeo variam de -1,6 a +4,8 níveis em relação
  ao controle), e o piso fixo do controle (piso_ref_*) não acompanha isso. Então, por vídeo e canal:
      piso_video = piso_canto_do_vídeo - (piso_canto_do_controle - piso_ref_do_controle)
  O termo entre parênteses é a diferença "cantos - dentro da ROI" medida no controle (~0,8 a 1,0). SUPOSIÇÃO: essa diferença
  vale igual para todas as sessões. Os valores brutos continuam nas colunas sem sufixo; os corrigidos vêm como *_corr.
  Com consolidacao.piso: referencia usa o piso fixo do controle (piso_ref_*) e a correção por sessão fica desligada.

Conferência (no relatório): em cada grupo de mesma proporção e mesma amostra (5 regravações), compara a dispersão do nível
final com o piso fixo e com o piso por vídeo, só nos canais cujo pó está presente. Se a correção funciona, a dispersão cai.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .curadoria import _ler as _ler_curadoria
from .nomes import ler_nome

CANAIS = ["R", "G", "B"]


def parametros(cfg: dict) -> dict:
    p = {"piso": "video", "incluir_pendentes_ok": True}
    p.update(cfg.get("consolidacao", {}) or {})
    if p["piso"] not in ("video", "referencia"):
        raise ValueError("consolidacao.piso deve ser 'video' ou 'referencia'")
    return p


def selecionar(resumo: pd.DataFrame, curadoria: dict[str, dict], p: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Separa quem entra de quem fica de fora. Retorna (aceitos, excluidos[{caminho, motivo}])."""
    aceitos, excluidos = [], []
    for _, l in resumo.iterrows():
        cur = curadoria.get(l["chave"], {})
        incluir = (cur.get("incluir") or str(l.get("incluir", "pendente"))).strip().lower()
        status = cur.get("status_auditoria") or l.get("status_auditoria", "")
        if bool(l.get("controle", False)):
            excluidos.append({"caminho": l["caminho"], "motivo": "vídeo de controle"})
        elif incluir == "nao":
            excluidos.append({"caminho": l["caminho"], "motivo": f"incluir = nao ({cur.get('motivo') or 'sem motivo'})"})
        elif incluir == "sim" or (incluir == "pendente" and status == "ok" and p["incluir_pendentes_ok"]):
            aceitos.append({**l.to_dict(), "nome_aprovado": cur.get("nome_aprovado", ""), "incluir_curadoria": incluir,
                            "observacao_curadoria": cur.get("observacao", ""), "motivo_curadoria": cur.get("motivo", "")})
        else:
            excluidos.append({"caminho": l["caminho"], "motivo": f"incluir = {incluir}, auditoria = {status}: decidir na curadoria"})
    return pd.DataFrame(aceitos), excluidos


def _identificar(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """Acrescenta amostra, proporção, etc., lidos do nome aprovado (ou do arquivo)."""
    avisos, linhas = [], []
    treino = set(cfg.get("divisao", {}).get("treino", []))
    valid = set(cfg.get("divisao", {}).get("validacao", []))
    for _, l in df.iterrows():
        nome = l["nome_aprovado"] or Path(str(l["caminho"]).split(" | ")[0]).name
        campos, erro = ler_nome(nome, cfg.get("nomes", {}))
        if not campos or campos.get("tipo") != "amostra":
            avisos.append(f"{l['caminho']}: nome não lido ({erro or 'não é amostra'}); fica de fora")
            linhas.append(None)
            continue
        pasta = Path(str(l["caminho"]).split(" | ")[0]).parts
        letra_pasta = pasta[-3][-1].upper() if len(pasta) >= 3 and pasta[-3].lower().startswith("amostras") else ""
        if letra_pasta and letra_pasta != campos["amostra"]:
            avisos.append(f"{l['caminho']}: pasta diz {letra_pasta} mas o nome diz {campos['amostra']}")
        grupo = "treino" if campos["amostra"] in treino else "validacao" if campos["amostra"] in valid else "outro"
        stem = Path(campos.get("nome_sugerido") or nome).stem
        linhas.append({"video": stem[:-1] + stem[-1].lower(),
                       "amostra": campos["amostra"], "grupo": grupo, "num_video": campos["video"],
                       "proporcao": campos["proporcao"], "cores": campos["cores"],
                       **{f"prop_{c}": campos[f"prop_{c}"] for c in CANAIS}})
    ok = [i for i, x in enumerate(linhas) if x]
    meta = pd.DataFrame([linhas[i] for i in ok], index=df.index[ok])
    return pd.concat([meta, df.loc[df.index[ok]]], axis=1), avisos


def corrigir_preto(df: pd.DataFrame, controle: pd.Series | None, modo: str) -> pd.DataFrame:
    """Acrescenta piso_video_*, inicio_corr_*, final_corr_* e dif_piso_* (piso por vídeo - piso fixo)."""
    df = df.copy()
    for c in CANAIS:
        fixo = df[f"piso_ref_{c}"].astype(float)
        if modo == "video" and controle is not None:
            offset = float(controle[f"piso_canto_{c}"]) - float(controle[f"piso_ref_{c}"])
            piso = df[f"piso_canto_{c}"].astype(float) - offset
            df[f"offset_canto_roi_{c}"] = round(offset, 3)
        else:
            piso = fixo
        df[f"piso_video_{c}"] = piso.round(3)
        df[f"dif_piso_{c}"] = (piso - fixo).round(3)
        df[f"inicio_corr_{c}"] = (df[f"inicio_{c}"] - piso).round(3)
        df[f"final_corr_{c}"] = (df[f"final_{c}"] - piso).round(3)
        df[f"canal_ativo_{c}"] = df[f"prop_{c}"] > 0
    return df


def avaliar_correcao(df: pd.DataFrame) -> dict:
    """Dispersão do nível final entre as regravações (mesma proporção e amostra), piso fixo x piso por vídeo.

    Só entram canais cujo pó está na mistura. Retorna, por canal, o desvio padrão agrupado (pooled) e o nº de grupos.
    """
    saida = {}
    for c in CANAIS:
        ativo = df[df[f"canal_ativo_{c}"]]
        var_fixo, var_video, n = [], [], 0
        for _, g in ativo.groupby(["proporcao", "amostra"]):
            if len(g) < 2:
                continue
            var_fixo.append((g[f"final_{c}"] - g[f"piso_ref_{c}"]).var(ddof=1))
            var_video.append(g[f"final_corr_{c}"].var(ddof=1))
            n += 1
        if n:
            saida[c] = {"grupos": n, "dp_piso_fixo": float(np.sqrt(np.mean(var_fixo))),
                        "dp_piso_video": float(np.sqrt(np.mean(var_video))),
                        "mediana_final": float(ativo[f"final_{c}"].median())}
    # viés entre amostras: diferença média de piso (video - fixo) por amostra
    vies = {c: df.groupby("amostra")[f"dif_piso_{c}"].mean().round(2).to_dict() for c in CANAIS}
    saida["vies_por_amostra"] = vies
    return saida


def _relatorio(aceitos: pd.DataFrame, excluidos: list[dict], avisos: list[str], aval: dict, modo: str, tem_controle: bool) -> str:
    L = ["CONSOLIDAÇÃO", "=" * 60, f"Vídeos aceitos: {len(aceitos)}   fora: {len(excluidos)}   piso: {modo}"]
    if modo == "video" and not tem_controle:
        L.append("AVISO: sem vídeo de controle no resumo; a correção por vídeo usou o piso fixo.")
    L += ["", "Aceitos por amostra e grupo:"]
    if len(aceitos):
        for r in aceitos.groupby(["amostra", "grupo"]).size().reset_index(name="n").itertuples():
            L.append(f"  {r.amostra} ({r.grupo}): {r.n}")
    L += ["", "Fora da consolidação:"] + [f"  {e['caminho']}: {e['motivo']}" for e in excluidos]
    com_alerta = aceitos[aceitos["motivo_curadoria"].astype(str) != ""] if len(aceitos) else aceitos
    if len(com_alerta):
        L += ["", "Aceitos com alerta (ver README, 'Alertas menores aceitos'):"]
        L += [f"  {r['video']}: {r['motivo_curadoria']} - {r['observacao_curadoria']}" for _, r in com_alerta.iterrows()]
    if avisos:
        L += ["", "Avisos:"] + [f"  {a}" for a in avisos]
    L += ["", "Conferência do preto (desvio padrão do nível final entre as regravações; só canais com o pó presente):"]
    for c in CANAIS:
        if c in aval:
            a = aval[c]
            L.append(f"  {c}: piso fixo {a['dp_piso_fixo']:.2f}  piso por vídeo {a['dp_piso_video']:.2f}  "
                     f"({a['grupos']} grupos; nível final mediano {a['mediana_final']:.1f})")
    L.append("  Viés médio do piso por vídeo menos o fixo, por amostra: " + str(aval.get("vies_por_amostra", {})))
    L.append("  Se o desvio com piso por vídeo não for menor que com o piso fixo, a correção não ajuda e deve ser revista.")
    return "\n".join(L) + "\n"


def rodar(cfg: dict, pasta_extracao=None, saida=None, curadoria=None) -> int:
    p = parametros(cfg)
    raiz = Path(cfg.get("pastas", {}).get("saida", "output"))
    pasta_ext = Path(pasta_extracao or raiz / "extracao")
    resumo_csv = pasta_ext / "extracao_resumo.csv"
    if not resumo_csv.exists():
        raise FileNotFoundError(f"{resumo_csv} não existe (rode 'luminescencia extrair')")
    pasta_out = Path(saida or raiz / "consolidacao")
    pasta_out.mkdir(parents=True, exist_ok=True)

    resumo = pd.read_csv(resumo_csv, encoding="utf-8-sig")
    cur = _ler_curadoria(Path(curadoria or cfg.get("curadoria", {}).get("arquivo", "curadoria.csv")))
    controle = resumo[resumo["controle"].astype(bool)]
    controle = controle.iloc[0] if len(controle) else None

    aceitos, excluidos = selecionar(resumo, cur, p)
    avisos: list[str] = []
    if len(aceitos):
        aceitos, avisos = _identificar(aceitos, cfg)
    if not len(aceitos):
        print("Nenhum vídeo aceito.")
        return 1
    aceitos = corrigir_preto(aceitos, controle, p["piso"])
    nomes_dup = aceitos["video"][aceitos["video"].duplicated(keep=False)].unique()
    for n in nomes_dup:
        avisos.append(f"nome repetido entre vídeos aceitos: {n} (o mesmo nome em duas gravações diferentes)")

    # formato longo
    partes, sem_csv = [], []
    for _, v in aceitos.iterrows():
        arq = pasta_ext / v["arquivo_csv"]
        if not arq.exists():
            sem_csv.append(v["arquivo_csv"])
            continue
        d = pd.read_csv(arq, encoding="utf-8-sig")
        for c in CANAIS:
            d[f"corr_{c}"] = (d[f"media_{c}"] - v[f"piso_video_{c}"]).round(3)
        for k in ["video", "amostra", "grupo", "num_video", "proporcao", "prop_R", "prop_G", "prop_B"]:
            d.insert(0, k, v[k]) if k not in d else None
        partes.append(d)
    if sem_csv:
        avisos.append(f"{len(sem_csv)} vídeo(s) sem CSV de extração em {pasta_ext} e portanto sem série (ex.: {', '.join(sem_csv[:3])})")
    ordem = ["video", "amostra", "grupo", "num_video", "proporcao", "prop_R", "prop_G", "prop_B", "frame", "tempo_s", "fase",
             "media_R", "media_G", "media_B", "corr_R", "corr_G", "corr_B", "saturados_pct", "repeticoes"]
    if partes:
        series = pd.concat(partes, ignore_index=True)
        series[[c for c in ordem if c in series.columns]].to_csv(pasta_out / "series.csv", index=False, encoding="utf-8-sig")

    aceitos.drop(columns=["caminho_original"], errors="ignore").to_csv(pasta_out / "videos.csv", index=False, encoding="utf-8-sig")
    aval = avaliar_correcao(aceitos)
    texto = _relatorio(aceitos, excluidos, avisos, aval, p["piso"], controle is not None)
    (pasta_out / "relatorio_consolidacao.txt").write_text(texto, encoding="utf-8")
    print(texto)
    print(f"Arquivos em {pasta_out}: videos.csv, series.csv, relatorio_consolidacao.txt")
    return 0
