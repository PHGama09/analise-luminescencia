# Protocolo de Validação (Fase 1)

## 1. Objetivo e Alvo
- **Alvo:** Regressão das frações de proporção R/G/B.
- **Estratégia de Divisão:** Treino nas amostras **A + B** (múltiplas preparações), teste final estrito na amostra **C** (preparação independente única, evitando vazamento de lote de fabricação).

## 2. Validação Cruzada
- Utilização de **Validação Cruzada Agrupada** (Leave-One-Sample-Out por proporção/amostra) para garantir que regravações da mesma preparação não fiquem divididas entre treino e validação.

## 3. Métricas e Linhas de Base
- **Métricas:** Erro Absoluto Médio (MAE) por fração e Coeficiente de Determinação ($R^2$).
- **Linha de Base:** Previsão pela média global e modelo k-NN restrito apenas a razões normalizadas de $I_0$ e $t_{50}$.

## 4. Tratamento de Exceções
- Os 9 vídeos saturados mantêm-se isolados no conjunto `a_parte` e não entram no ajuste principal.