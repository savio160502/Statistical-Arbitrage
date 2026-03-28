# 📈 Statistical Arbitrage with PCA and Mean Reversion

Projeto de dissertação desenvolvido na FGV EMAp com foco na construção, implementação e avaliação empírica de uma estratégia de **arbitragem estatística neutra ao mercado**, baseada em fatores extraídos via PCA e modelagem de reversão à média.

---

## 🧠 Overview

Este projeto propõe uma abordagem quantitativa para explorar ineficiências no mercado acionário através de:

- Decomposição dos retornos em componentes **sistemáticos e idiossincráticos** via **PCA**
- Modelagem dos resíduos como processos de **Ornstein–Uhlenbeck**
- Construção de sinais de trading baseados em **desvios de equilíbrio (s-score)**
- Implementação de uma estratégia **long–short com neutralização fatorial**
- Avaliação **out-of-sample** com custos de transação realistas

---

## ⚙️ Metodologia

### 1. Extração de fatores (PCA)
- Aplicação de PCA em janelas móveis
- Construção de fatores principais (eigenportfólios)

### 2. Regressão e resíduos
- Regressão dos retornos de cada ativo nos fatores
- Extração dos resíduos idiossincráticos

### 3. Modelagem OU

O processo de Ornstein–Uhlenbeck é dado por:

\[
dX_t = \kappa (m - X_t)\,dt + \sigma\,dW_t
\]

Onde:
- \(\kappa\): velocidade de reversão à média  
- \(m\): nível de equilíbrio  
- \(\sigma\): volatilidade  

### 4. Geração de sinais
- Construção do **s-score**
- Estratégia baseada em thresholds:
  - Entrada: \(|s| > s_{open}\)
  - Saída: \(|s| < s_{close}\)

### 5. Construção de portfólio
- Estratégia long–short
- Neutralização das exposições fatoriais (PCA)
- Alocação balanceada entre posições

---

## 📊 Resultados

### 🇧🇷 Mercado Brasileiro
- Performance consistente no curto e longo prazo
- Evidência de reversão à média nos resíduos idiossincráticos

### 🇺🇸 Mercado Norte-Americano
- Bons resultados no curto prazo
- Deterioração significativa no longo prazo

---

## 🔍 Extensões analisadas

- Número adaptativo de fatores (variance explained)
- Thresholds dinâmicos baseados em quantis
- Otimização de hiperparâmetros com **Optuna (TPE)**
- Walk-forward optimization

**Conclusão:** melhorias pontuais, mas sem ganhos robustos no longo prazo no mercado americano.

---

## 🧩 Código do Projeto

O projeto está organizado de forma simples e direta:

- `funcoes.py`  
  Contém todas as funções principais utilizadas ao longo do projeto, incluindo:
  - PCA e construção de fatores
  - Estimação do processo OU
  - Cálculo de s-score
  - Construção de portfólio
  - Backtesting

- `run_backtest.py`  
  Script responsável pela execução da estratégia no **mercado norte-americano**

- `run_backtest_br.py`  
  Script responsável pela execução da estratégia no **mercado brasileiro**

---

## 📦 Dependências principais

- numpy  
- pandas  
- scikit-learn  
- statsmodels  
- matplotlib  
- optuna  

---

## 📈 Métricas avaliadas

- Sharpe Ratio  
- CAGR  
- Drawdown  
- Turnover  
- Volatilidade  

---

## 📚 Referências

- Avellaneda, M., & Lee, J. (2008). *Statistical Arbitrage in the U.S. Equities Market*  
- Gatev, Goetzmann & Rouwenhorst (2006)  
- Vidyamurthy (2004)  

- KUMAR, N. *Advantages and Disadvantages of Principal Component Analysis in Machine Learning*. 2019.  
  ⟨http://theprofessionalspoint.blogspot.com/2019/03/advantages-anddisadvantages-of-4.html⟩  

- WANG, J. et al. *A novel combination of PCA and machine learning techniques to select the most important factors for predicting tunnel construction performance*.  
  Buildings, v. 12, n. 7, 2022.  
  ⟨https://www.mdpi.com/2075-5309/12/7/919⟩  

---

## 👨‍💻 Autor

**Sávio Amaral**  
FGV EMAp — Mestrado em Matemática Aplicada e Ciência de Dados  

---

## 📬 Contato

Se quiser discutir o projeto ou estratégias quantitativas, fique à vontade para entrar em contato.

---

## ⚖️ Licença

Este projeto é destinado exclusivamente para fins acadêmicos e educacionais.
