# \# Estratégia de Arbitragem Estatística com PCA e Processos de Ornstein-Uhlenbeck

# 

# Este projeto implementa uma estratégia quantitativa de arbitragem estatística baseada em fatores principais (PCA) e processos de Ornstein–Uhlenbeck aplicados aos resíduos de regressões fatoriais. A estratégia investe em reversão à média, utilizando \*\*s-scores\*\* derivados de modelos AR(1) calibrados sobre janelas móveis dos resíduos.

# 

# \## 💡 Objetivo

# 

# Desenvolver uma estratégia “market-neutral” baseada exclusivamente em preços históricos, sem uso de ETFs, que:

# \- Gera sinais de compra/venda com base em desvios estatísticos (s-score),

# \- Modela resíduos como processos OU para detectar reversão,

# \- Rebalanceia diariamente com controle de alavancagem,

# \- Incorpora custo de fricção de 10 bps por trade.

# 

# \## 📈 Metodologia

# 

# 1\. \*\*Coleta de dados\*\*: Baixar preços ajustados de ações (ex: S\&P 500) com `yfinance`.

# 2\. \*\*Normalização e PCA\*\*:

# &nbsp;  - Calcular retornos normalizados.

# &nbsp;  - Extrair os 15 principais componentes da matriz de correlação.

# 3\. \*\*Modelo fatorial\*\*:

# &nbsp;  - Regressão dos retornos das ações sobre os fatores PCA.

# &nbsp;  - Extração dos resíduos diários.

# 4\. \*\*Modelagem dos resíduos\*\*:

# &nbsp;  - Em janelas móveis (60 dias), modelar resíduos acumulados como processos OU.

# &nbsp;  - Estimar \\( a, b, \\kappa, m, \\sigma\_{eq} \\) via regressão AR(1).

# &nbsp;  - Calcular o \*\*s-score\*\*: \\( s = \\frac{-m}{\\sigma\_{eq}} \\)

# 5\. \*\*Backtest\*\*:

# &nbsp;  - Estratégia bang-bang: abre posição total quando |s| > 1.5, fecha quando |s| < 0.5.

# &nbsp;  - PnL diário calculado com rebalanceamento, fricção, alavancagem.

