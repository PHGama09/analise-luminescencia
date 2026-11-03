"""Testes para o módulo de deriva."""
from luminescencia import deriva

def test_deriva_basico(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    
    # Cria a estrutura de pastas esperada pelo caminho predefinido
    out_dir = tmp_path / "output" / "tabela"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Cria ficheiros mínimos com colunas de exemplo válidas
    conteudo = "proporcao,amostra,grupo,num_video,video,corr_g_I0\nB1,A,G1,1,video1,10.0\n"
    (out_dir / "dataset.csv").write_text(conteudo, encoding="utf-8")
    (out_dir / "descritores.csv").write_text(conteudo, encoding="utf-8")
    
    res = deriva.analisar_deriva()
    assert res["status"] == "concluido"