"""Arrumação do repositório luminescencia (verificação de 2026-11-04).

Uso (na raiz do repositório, venv ativo):
    python arrumar_2026-11-04.py            # só mostra o que faria (padrão = --dry-run)
    python arrumar_2026-11-04.py --aplicar  # executa

Idempotente: rodar de novo não faz nada que já foi feito.
Nunca toca em vídeos, output/, curadoria.csv nem .venv (há uma trava explícita abaixo).

O que faz:
  1. Tira `curadoria.csv.bak` do índice do Git (git rm --cached; o arquivo fica no disco, já coberto por *.bak).
  2. Move os zips soltos da raiz (exceto alteracoes*.zip) para _arquivo/zips_2026-11-04/.
  3. Move `conferencia_v0.11.2/` para _arquivo/ (já está fora do Git).
  4. Move `README_trecho_*.md` para _arquivo/readme_trechos/ SÓ SE o README.md já tiver o marcador
     `<!-- trechos-incorporados-2026-11-04 -->` (ou seja, depois de aplicar o README do alteracoes.zip).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

RAIZ = Path.cwd()
ARQ = RAIZ / "_arquivo"
MARCADOR = "<!-- trechos-incorporados-2026-11-04 -->"
PROTEGIDOS_NOMES = {"output", ".venv", "curadoria.csv", ".git"}
EXT_VIDEO = {".avi", ".mp4", ".mov", ".mkv"}


def protegido(p: Path) -> bool:
    rel = p.resolve().relative_to(RAIZ.resolve())
    return (rel.parts and rel.parts[0] in PROTEGIDOS_NOMES) or p.suffix.lower() in EXT_VIDEO or p.name == "curadoria.csv"


def mover(src: Path, dst_dir: Path, aplicar: bool) -> None:
    if not src.exists():
        return
    if protegido(src):
        print(f"  RECUSADO (protegido): {src.name}")
        return
    dst = dst_dir / src.name
    if dst.exists():
        print(f"  já existe no destino, nada feito: {dst.relative_to(RAIZ)}")
        return
    print(f"  mover {src.relative_to(RAIZ)} -> {dst.relative_to(RAIZ)}")
    if aplicar:
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def git_rastreado(caminho: str) -> bool:
    r = subprocess.run(["git", "ls-files", "--error-unmatch", caminho], cwd=RAIZ, capture_output=True)
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="só mostra (padrão)")
    g.add_argument("--aplicar", action="store_true", help="executa")
    a = ap.parse_args()
    aplicar = bool(a.aplicar)
    if not (RAIZ / "pyproject.toml").exists() or not (RAIZ / "src" / "luminescencia").is_dir():
        print("Rode na raiz do repositório luminescencia.")
        return 2
    print(("APLICANDO" if aplicar else "DRY-RUN (nada é alterado; use --aplicar)") + f" em {RAIZ}")

    print("1. curadoria.csv.bak fora do Git")
    if git_rastreado("curadoria.csv.bak"):
        print("  git rm --cached curadoria.csv.bak")
        if aplicar:
            subprocess.run(["git", "rm", "--cached", "-q", "curadoria.csv.bak"], cwd=RAIZ, check=True)
    else:
        print("  já fora do Git")

    print("2. zips soltos na raiz")
    for z in sorted(RAIZ.glob("*.zip")):
        if z.name.lower().startswith("alteracoes"):
            continue
        mover(z, ARQ / "zips_2026-11-04", aplicar)

    print("3. conferencia_v0.11.2/")
    mover(RAIZ / "conferencia_v0.11.2", ARQ, aplicar)

    print("4. README_trecho_*.md")
    trechos = sorted(RAIZ.glob("README_trecho_*.md"))
    readme = (RAIZ / "README.md").read_text(encoding="utf-8") if (RAIZ / "README.md").exists() else ""
    if not trechos:
        print("  nenhum trecho na raiz")
    elif MARCADOR not in readme:
        print("  README.md ainda não incorpora os trechos (aplique o README do alteracoes.zip antes); nada movido")
    else:
        for t in trechos:
            mover(t, ARQ / "readme_trechos", aplicar)

    gi = (RAIZ / ".gitignore").read_text(encoding="utf-8") if (RAIZ / ".gitignore").exists() else ""
    if "_arquivo/" not in gi:
        print("AVISO: _arquivo/ não está no .gitignore (aplique o .gitignore do alteracoes.zip).")
    print("Fim.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
