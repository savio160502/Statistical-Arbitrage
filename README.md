# Estratégia de Arbitragem Estatística com PCA e Processos de Ornstein-Uhlenbeck

Este projeto implementa uma estratégia quantitativa de arbitragem estatística baseada em fatores principais (PCA) e processos de Ornstein–Uhlenbeck aplicados aos resíduos de regressões fatoriais. A estratégia investe em reversão à média, utilizando **s-scores** derivados de modelos AR(1) calibrados sobre janelas móveis dos resíduos.

## 💡 Objetivo

Desenvolver uma estratégia “market-neutral” baseada exclusivamente em preços históricos, sem uso de ETFs, que:
- Gera sinais de compra/venda com base em desvios estatísticos (s-score),
- Modela resíduos como processos OU para detectar reversão,
- Rebalanceia diariamente com controle de alavancagem,
- Incorpora custo de fricção de 10 bps por trade.

## 📈 Metodologia

1. **Coleta de dados**: Baixar preços ajustados de ações (ex: S&P 500) com `yfinance`.
2. **Normalização e PCA**:
   - Calcular retornos normalizados.
   - Extrair os 15 principais componentes da matriz de correlação.
3. **Modelo fatorial**:
   - Regressão dos retornos das ações sobre os fatores PCA.
   - Extração dos resíduos diários.
4. **Modelagem dos resíduos**:
   - Em janelas móveis (60 dias), modelar resíduos acumulados como processos OU.
   - Estimar a, b, $\kappa$, m, $\sigma_{eq}$ via regressão AR(1).
   - Calcular o **s-score**:  $$s = \frac{-m}{\sigma_{eq}} $$
5. **Backtest**:
   - Estratégia bang-bang: abre posição total quando |s| > 1.5, fecha quando |s| < 0.5.
   - PnL diário calculado com rebalanceamento, fricção, alavancagem.

## ⚙️ Como rodar
