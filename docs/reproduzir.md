> **RASCUNHO escrito sem dados; não foi validado nem discutido com o Pedro; ver pendência Q4/C17**

# Guia de Reprodução - Projeto Luminescência

Passo a passo para reproduzir as análises e a consolidação:
1. Instale o pacote em modo de desenvolvimento (`pip install -e .`).
2. Execute a consolidação dos dados: `luminescencia consolidar`.
3. Execute o processamento posterior: `luminescencia pos`.
4. Execute a modelagem: `luminescencia modelar`.
5. Verifique com os testes automatizados: `pytest`.
