# 📈 Statistical Arbitrage with PCA and Mean Reversion

Projeto de dissertação desenvolvido na FGV EMAp com foco na construção, implementação e avaliação empírica de uma estratégia de **arbitragem estatística neutra ao mercado**, baseada em fatores extraídos via PCA e modelagem de reversão à média.

---

## 🧠 Overview

Este projeto propõe uma abordagem quantitativa para explorar ineficiências no mercado acionário por meio de:

- decomposição dos retornos em componentes **sistemáticos e idiossincráticos** via **PCA**;
- modelagem dos resíduos como processos de **Ornstein–Uhlenbeck**;
- construção de sinais de negociação com base em medidas de desalinhamento relativo;
- implementação de uma estratégia **long-short com neutralização fatorial**;
- avaliação **out-of-sample** com custos de transação realistas.

---

## ⚙️ Metodologia

### 1. Extração de fatores (PCA)

- aplicação de PCA em janelas móveis;
- construção de fatores principais a partir da estrutura de covariância/correlação dos retornos;
- separação entre componentes comuns e componentes idiossincráticos.

### 2. Regressão e resíduos

- regressão dos retornos de cada ativo sobre os fatores principais;
- obtenção dos resíduos idiossincráticos;
- uso desses resíduos como base para a modelagem de reversão à média.

### 3. Modelagem OU

O processo de Ornstein–Uhlenbeck é dado por:

$$
dX_t = \kappa (m - X_t)\,dt + \sigma\,dW_t
$$

onde:

- $\kappa$: velocidade de reversão à média;
- $m$: nível de equilíbrio de longo prazo;
- $\sigma$: volatilidade do processo.

### 4. Geração de sinais

- construção do **s-score**;
- definição de regras de entrada e saída baseadas em limiares:

$$
|s| > s_{\text{open}}
$$

para abertura de posição, e

$$
|s| < s_{\text{close}}
$$

para encerramento.

### 5. Construção de portfólio

- estratégia **long-short**;
- neutralização explícita das exposições fatoriais obtidas via PCA;
- rebalanceamento e avaliação com custos de transação.

---

## 📊 Resultados

### 🇧🇷 Mercado brasileiro

- desempenho consistente no curto e no longo prazo;
- evidência de reversão à média nos componentes idiossincráticos.

### 🇺🇸 Mercado norte-americano

- resultados positivos no curto prazo;
- deterioração significativa do desempenho no longo prazo.

---

## 🔍 Extensões analisadas

- número adaptativo de fatores;
- limiares dinâmicos baseados em quantis;
- otimização de hiperparâmetros com **Optuna**;
- procedimentos de recalibração ao longo do tempo.

De forma geral, essas extensões produziram melhorias pontuais, mas não restauraram de forma consistente o desempenho de longo prazo no mercado americano.

---

## 🧩 Código do projeto

- `funcoes.py`  
  Reúne as principais funções do projeto, incluindo rotinas de PCA, estimação do processo OU, cálculo dos sinais, construção de portfólio e backtesting.

- `run_backtest.py`  
  Executa os experimentos e backtests para o **mercado norte-americano**.

- `run_backtest_br.py`  
  Executa os experimentos e backtests para o **mercado brasileiro**.

---

## 📦 Dependências principais

- `numpy`
- `pandas`
- `statsmodels`
- `matplotlib`
- `optuna`

---

## 📈 Métricas avaliadas

- Sharpe Ratio
- CAGR
- Drawdown
- Turnover
- Volatilidade

---

## 📚 Referências

- AVELLANEDA, M.; LEE, J.-H. *Statistical arbitrage in the U.S. equities market*. SSRN Electronic Journal, 2008. Disponível em: <https://ssrn.com/abstract=1153505>.

- GATEV, E.; GOETZMANN, W. N.; ROUWENHORST, K. G. *Pairs trading: Performance of a relative-value arbitrage rule*. **Review of Financial Studies**, v. 19, n. 3, p. 797–827, 2006.

- VIDYAMURTHY, G. *Pairs Trading: Quantitative Methods and Analysis*. Hoboken: Wiley, 2004.

- KUMAR, N. *Advantages and Disadvantages of Principal Component Analysis in Machine Learning*. 2019. Disponível em: <http://theprofessionalspoint.blogspot.com/2019/03/advantages-anddisadvantages-of-4.html>. Acesso em: 15 abr. 2021.

- WANG, J. et al. *A novel combination of PCA and machine learning techniques to select the most important factors for predicting tunnel construction performance*. **Buildings**, v. 12, n. 7, 2022. ISSN 2075-5309. Disponível em: <https://www.mdpi.com/2075-5309/12/7/919>.

---

## 👨‍💻 Autor

**Sávio Amaral**  
FGV EMAp — Mestrado em Matemática Aplicada e Ciência de Dados

---

## ⚖️ Licença

Este projeto é destinado exclusivamente para fins acadêmicos e educacionais.
