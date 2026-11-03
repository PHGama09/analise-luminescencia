"""Testes da curadoria: as decisões do usuário nunca são perdidas. Rode: pytest"""
import csv

from luminescencia.curadoria import atualizar_curadoria


def _r(hash_, caminho, status="ok", sugerido=""):
    return {"hash": hash_, "caminho": caminho, "status": status, "nome_sugerido": sugerido}


def _ler(caminho):
    with open(caminho, encoding="utf-8-sig", newline="") as f:
        return {l["chave"]: l for l in csv.DictReader(f)}


def _decidir(caminho, chave, **campos):
    linhas = list(_ler(caminho).values())
    for l in linhas:
        if l["chave"] == chave:
            l.update(campos)
    with open(caminho, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)


def test_novos_entram_como_pendente(tmp_path):
    arq = tmp_path / "curadoria.csv"
    res = atualizar_curadoria([_r("aa", "A/1.avi"), _r("bb", "A/2.avi", "revisar", "X.avi")], arq)
    assert res["novos"] == 2 and res["pendentes"] == 2
    l = _ler(arq)
    assert l["sha1:bb"]["nome_sugerido"] == "X.avi" and l["sha1:bb"]["status_auditoria"] == "revisar"


def test_decisao_sobrevive_a_mover_e_a_sumir(tmp_path):
    arq = tmp_path / "curadoria.csv"
    atualizar_curadoria([_r("aa", "A/1.avi"), _r("bb", "A/2.avi")], arq)
    _decidir(arq, "sha1:aa", incluir="nao", motivo="erro_de_gravacao", observacao="travou")

    atualizar_curadoria([_r("aa", "B/novo_nome.avi"), _r("bb", "A/2.avi")], arq)      # vídeo movido e renomeado
    l = _ler(arq)["sha1:aa"]
    assert l["caminho"] == "B/novo_nome.avi" and l["incluir"] == "nao" and l["observacao"] == "travou"

    res = atualizar_curadoria([_r("bb", "A/2.avi")], arq)                              # vídeo sumiu da pasta
    l = _ler(arq)["sha1:aa"]
    assert res["ausentes"] == 1 and l["status_auditoria"] == "ausente" and l["incluir"] == "nao"
    assert (tmp_path / "curadoria.csv.bak").exists()


def test_copias_identicas_viram_uma_linha(tmp_path):
    arq = tmp_path / "curadoria.csv"
    res = atualizar_curadoria([_r("aa", "A/2/GB _05_1_v2a.avi"), _r("aa", "A/2/GB_05_1_v2a.avi")], arq)
    assert res["total"] == 1
    assert " | " in _ler(arq)["sha1:aa"]["caminho"]


def test_valor_invalido_gera_aviso_sem_alterar(tmp_path):
    arq = tmp_path / "curadoria.csv"
    atualizar_curadoria([_r("aa", "A/1.avi")], arq)
    _decidir(arq, "sha1:aa", incluir="talvez", motivo="inventado")
    res = atualizar_curadoria([_r("aa", "A/1.avi")], arq)
    assert len(res["avisos"]) == 2 and _ler(arq)["sha1:aa"]["incluir"] == "talvez"
