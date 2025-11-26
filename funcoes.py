# Arquivo com todas as funções que serão usadas para a execução da estratégia

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import bs4 as bs
import requests
import seaborn as sns

# =============================================================================
# Utilidades gerais
# =============================================================================

def padronizar_janela(df):
    """
    Padroniza cada coluna por média e desvio-padrão (ddof=1) na própria janela.
    Mantém NaN se std==0 ou NaN.
    """
    mu = df.mean()
    sd = df.std(ddof=1).replace(0, np.nan)
    return (df - mu) / sd

def decompor_corr(Z):
    """
    Decompõe a matriz de correlação de Z (colunas padronizadas) via np.linalg.eigh
    e retorna (autovalores decrescentes, autovetores colunarmente ordenados).
    """
    C = Z.corr()
    evals, evecs = np.linalg.eigh(C.values)
    idx = np.argsort(evals)[::-1]
    return evals[idx], evecs[:, idx]

def normalizar_pesos(w):
    """
    Normaliza pesos para soma das magnitudes = 1 (controle de gross).
    Se tudo for zero, devolve zeros.
    """
    s = np.abs(w).sum()
    if s is None or s == 0 or not np.isfinite(s):
        return w * 0.0
    return w / s


# =============================================================================
# Fatores PCA
# =============================================================================

def compute_pca_factor_returns(returns: pd.DataFrame, window_pca: int = 60, n_factors: int = 15):
    """
    PCA: para cada t (a partir de window_pca),
    - padroniza janela [t-window_pca, t)
    - decompõe a correlação
    - monta pesos w_j = v_j / sigma_j (sigma da janela nos ativos válidos)
    - calcula F_j(t) = Rt · w_j usando retornos do próprio dia t

    Retorna
    -------
    DataFrame com colunas eig1..eig_m
    """
    dates = returns.index
    F = pd.DataFrame(index=dates, columns=[f"eig{i+1}" for i in range(n_factors)], dtype=float)

    for i in range(window_pca, len(dates)):
        t_hist = dates[i - window_pca : i]   # janela [t-60, t)
        t = dates[i]                         # data-alvo

        Rw = returns.loc[t_hist]
        Zw = padronizar_janela(Rw)
        Zw = Zw.dropna(axis=1, how='any')    # exclui colunas com NaN na janela
        Zw = Zw.dropna(axis=0, how='all')   # exclui linhas com NaN na janela
        
        if Zw.shape[1] < n_factors + 1:
            continue

        # PCA na correlação da janela
        _, evecs = decompor_corr(Zw)

        # std dos ativos válidos na janela, para construir w_j
        sigma_w = Rw.std(ddof=1)

        # retornos do dia t
        Rt = returns.loc[t, Zw.columns]
        if Rt.isnull().any():
            continue

        for j in range(n_factors):
            vj = pd.Series(evecs[:, j], index=Zw.columns)
            wj = normalizar_pesos(vj / sigma_w)
            F.loc[t, f"eig{j+1}"] = float(Rt.dot(wj))

    return F.dropna(how="all")

# =============================================================================
# OU e s-score (com centralização cross-sectional do m)
# =============================================================================

def regress_action_on_pcs(X, y):
    """
    1) Ajusta regressão linear: y ~ X  (retorno da ação y com os retornos do PCs X)
    2) Retorna (betas, resíduos epsilon)
    """
    model = LinearRegression().fit(X, y)
    beta = model.coef_
    eps = y - model.predict(X)
    return beta, eps

def estimate_ou_from_cumsum(epsilon):
    """
    Estima um OU a partir do passeio acumulado X_k = sum_{j<=k} epsilon_j

    Modelo (discreto, AR(1)):
        X_{n+1} = a + b * X_n + ζ_{n+1}

    Retorna:
        a, b, var_zeta, kappa_ann, sigma_eq, m, X_T

    Onde:
      - kappa_ann = -log(b) * 252
      - sigma_eq  = sqrt( Var(ζ) / (1 - b^2) )
      - m         = a / (1 - b)                 (média de longo prazo)
      - X_T       = X_{ultimo} = sum(epsilon)   (estado atual do processo)
    
    """
    # checagens básicas
    eps = np.asarray(epsilon, dtype=float)
    if eps.ndim != 1:
        eps = eps.ravel()
    if len(eps) < 2 or not np.isfinite(eps).all():
        return None

    # passeio acumulado
    Xk = np.cumsum(eps)
    if len(Xk) < 2:
        return None

    # regressão 1-lag: X_{n+1} = a + b X_n + ζ_{n+1}
    X_ou = Xk[:-1].reshape(-1, 1)
    y_ou = Xk[1:]

    model = LinearRegression().fit(X_ou, y_ou)
    a = float(model.intercept_)
    b = float(model.coef_[0])

    # checando os parâmetros
    if not np.isfinite(a) or not np.isfinite(b) or not (1e-6 < b < 0.999):
        return None

    # ruído e variância do choque
    zeta = y_ou - model.predict(X_ou)
    var_zeta = float(np.var(zeta, ddof=1))
    if not np.isfinite(var_zeta) or var_zeta <= 0:
        return None

    # m (nível de equilíbrio), kappa anualizado e sigma no equilíbrio
    m = a / (1.0 - b)
    kappa_ann = -np.log(b) * float(252)
    sigma_eq = np.sqrt(var_zeta / (1.0 - b**2))

    # estado atual do processo: X_T
    X_T = float(Xk[-1])

    return m, X_T, var_zeta, kappa_ann, sigma_eq

def compute_s_scores_cross_sectional(returns: pd.DataFrame,factors: pd.DataFrame, kappa_min: float = 252.0/30.0): # filtro do paper
    """
    - Para cada ação: regressão ação~PCs → resíduos → OU → (a,b,var, kappa, sigma_eq)
    - Calcula m_i = a/(1-b) por ação válida
    - Centraliza m_i: m_i* = m_i - mean_j(m_j)
    - s_i = Xt - m_i* / sigma_eq_i
    Retorna:
      s_scores_t (Series por ação), betas_t (dict ação->vetor de betas) de todas as ações naquele dia
    """
    # alinhar datas e colunas
    common_idx = returns.index.intersection(factors.index)
    returns = returns.reindex(index=common_idx)
    factors = factors.reindex(index=common_idx)
    
    stocks = list(returns.columns)
    s_t = pd.Series(index=stocks, dtype=float)
    betas_t, m_map, sigma_eq_map, X_T_map = {}, {}, {},{}

    X_mat = factors.values                   # tabela dos retornos dos PCs

    for stock in stocks:
        y = returns[stock].values                 # tabela dos retornos da ação específica
        if np.isnan(y).any() or np.isnan(X_mat).any():
            continue

        beta, eps = regress_action_on_pcs(X_mat, y)
        ou = estimate_ou_from_cumsum(eps)          #  return m, X_T, var_zeta, kappa_ann, sigma_eq
        if ou is None:
            continue

        m, X_T, var_zeta, kappa_ann, sigma_eq = ou
        if kappa_ann <= kappa_min:
            # reverte devagar demais segundo o critério do paper
            continue

        m_map[stock] = m
        sigma_eq_map[stock] = sigma_eq
        betas_t[stock] = beta
        X_T_map[stock] = X_T

    if not m_map:
        return s_t, betas_t  # vazio

    # centralização cross-sectional
    m_bar = np.mean(list(m_map.values()))
    for stock, m in m_map.items():
        m_centered = m - m_bar
        s_val = (X_T_map[stock] - m) / sigma_eq_map[stock]  # s_score do paper
        s_val2 = (X_T_map[stock] - m_centered) / sigma_eq_map[stock] # s_score do paper centralizado (melhor)

        if np.isfinite(s_val2):
            s_t.loc[stock] = s_val2

    return s_t, betas_t

# =============================================================================
# Regras de posição, hedge e PnL
# =============================================================================

def position_from_s(
    s: float,            # s-score do dia
    pos_prev: float,     # {-1, 0, +1}  (posições do dia anterior)
    sbo: float = 1.25,   # buy-to-open se s < -sbo
    sso: float = 1.25,   # sell-to-open se s > +sso
    sbc: float = 0.75,   # close short se s < +sbc
    ssc: float = 0.50,   # close long  se s > -ssc
):

    if pd.isna(s):
        return float(pos_prev)

    pos = pos_prev

    # Aberturas (prioridade)
    if s > +sso:
        pos = -1.0
    elif s < -sbo:
        pos = +1.0

    # Fechamentos (aplicados se não houve nova abertura)
    elif (s < +sbc) and (pos_prev == -1.0):
        pos = 0.0
    elif (s > -ssc) and (pos_prev == +1.0):
        pos = 0.0

    return float(pos)

def equal_weight_by_side(row_pos: pd.Series):
    """
    Converte posições discretas {-1,0,+1} em pesos por lado (long/short)
    com igual peso dentro de cada lado; neutros ficam 0.
    """
    tmp = row_pos.astype(float).copy()
    longs = tmp > 0
    shorts = tmp < 0
    nL = longs.sum()
    nS = shorts.sum()

    if nL > 0:
        tmp.loc[longs] = +1.0 / float(nL)
    else:
        tmp.loc[longs] = 0.0

    if nS > 0:
        tmp.loc[shorts] = -1.0 / float(nS)
    else:
        tmp.loc[shorts] = 0.0

    tmp.loc[~longs & ~shorts] = 0.0
    
    return tmp

# anular a exposição fator de toda a carteira de ações naquele dia
def hedge_from_betas(
    algo_weights: pd.DataFrame,
    betas: pd.DataFrame,
    stocks,
    pcs):
    """
    Calcula hedge nos PCs neutralizando a exposição agregada:
      hedge_t = - sum_s( w_s(t) * beta_s(t) )
    Saída é DataFrame [date x pcs] com pesos de hedge.
    """
    
    # alinhar datas e colunas
    common_idx = algo_weights.index.intersection(betas.index)
    beta_bar = betas.reindex(index=common_idx, columns=stocks)
    W = algo_weights.reindex(index=common_idx, columns=stocks).fillna(0.0)
    
    # limpar células: garantir vetor v´alido por a¸c~ao
    def clean_cell(v):
        if isinstance(v, np.ndarray) and v.shape == (len(pcs),) and np.isfinite(v).all():
            return v
        return np.zeros(len(pcs))
    beta_bar = beta_bar.apply(lambda col: col.map(clean_cell))

    hedge_rows = []
    for dt, w in W.iterrows():
        b_row = beta_bar.loc[dt, stocks].to_list()
        if not b_row:
            hedge_rows.append(np.zeros(len(pcs)))
            continue
        b_dt = np.vstack(b_row)                         # shape: (n_stocks, m)
        expo = w.values @ b_dt                          # Σ_i w_i β_i  (shape: (m,)) exposição média por PC
        hedge_rows.append(-expo)                        # neutraliza com sinal oposto
        
    hedge = pd.DataFrame(hedge_rows, index=common_idx, columns=pcs)
    return hedge

def normalize_gross(w_all: pd.DataFrame, gross_target: float = 1.0):
    """
    Normaliza linha-a-linha para que ∑|w_i| = gross_target (por padrão =1).
    """
    gross = w_all.abs().sum(axis=1).replace(0, np.nan)
    W = w_all.div(gross, axis=0).fillna(0.0)
     
    return W * gross_target

def compute_pnl_with_costs(
    w_all: pd.DataFrame,
    returns_mod: pd.DataFrame,
    eps_per_turnover: float = 0.0005
):
    """
    PnL líquido com:
      - execução com lag-1 (w_{t-1})
      - custo linear sobre turnover diário
    """
    # Garante que só usamos colunas comuns entre pesos e retornos
    common_cols = w_all.columns.intersection(returns_mod.columns)

    # Alinha índices e preenche eventuais faltas com zero
    w = w_all[common_cols].reindex(index=returns_mod.index).fillna(0.0)
    rets = returns_mod[common_cols].reindex(index=w.index).fillna(0.0)

    # Pesos com lag-1: evita look-ahead (usa pesos de ontem com retornos de hoje)
    w_shift = w.shift(1).fillna(0.0)

    # PnL bruto: soma dos produtos peso * retorno em todas as colunas
    ret_gross = (rets * w_shift).sum(axis=1)
    
    # Turnover diário/custo de rebalanceamento: quanto o peso mudou de um dia para o outro em todas as colunas
    delta_w = w - w.shift(1)
    turnover = delta_w.abs().sum(axis=1).fillna(0.0)

    # PnL líquido e curva acumulada
    ret_net = ret_gross - eps_per_turnover * turnover
    cumret = (1.0 + ret_net).cumprod()

    return ret_net, cumret, turnover


    # =============================================================================
# Função principal (backtest) sem comentário - mais rápida
# =============================================================================

def pca_portfolio_spy(
    returns: pd.DataFrame,
    returns_spy: pd.DataFrame,
    num_pc: int = 15,
    s_win: int = 60,
    # thresholds do paper:
    sbo: float = 1.25,
    sso: float = 1.25,
    sbc: float = 0.75,
    ssc: float = 0.50,
    eps_cost: float = 0.0005,
    plot: bool = True,
):
    
    # Fatores PCA (rolling) com janela de 60 dias
    Factor_PCA = compute_pca_factor_returns(
    returns, window_pca=60, n_factors=num_pc)
    
    pcs = [f"eig{i+1}" for i in range(num_pc)]
    stocks = [c for c in returns.columns]
    usable_index = returns.iloc[252:].index

    # tabelas
    s_scores = pd.DataFrame(index=usable_index, columns=stocks, dtype=float)
    betas = pd.DataFrame(index=usable_index, columns=stocks, dtype=object)
    algo_pos = pd.DataFrame(index=usable_index, columns=stocks, dtype=float)
    
    # ------------- loop temporal -------------
    for t in usable_index:
        print(f"Tempo : {t}")
        # janela [t-s_win, t] para estimação OU
        ret = returns.loc[:t].iloc[-s_win:].copy()
        ret = padronizar_janela(ret)
        factor = Factor_PCA.loc[:t].iloc[-s_win:].copy()
        factor = padronizar_janela(factor)
        
        # checagem: PCs não podem ter NaN nessa janela padronizada
        if factor[pcs].isnull().any().any():
            continue

        # s-scores para o dia t (com centralização) + betas para hedge
        s_t, betas_t = compute_s_scores_cross_sectional(
            returns=ret,
            factors=factor,
            kappa_min=252.0/30.0,
        )

        # guarda s-scores e betas válidos
        s_scores.loc[t, s_t.index] = s_t
        for k, v in betas_t.items():
            betas.loc[t, k] = v

        # atualiza posições discretas com base no s-score de cada ação
        prev = algo_pos.shift(1).loc[t]
        if prev.isna().all():
            prev = pd.Series(0.0, index=stocks) #caso inicial

        new_pos = []
        for stock in stocks:
            s_val = s_t.get(stock, np.nan)
            new_pos.append(position_from_s(s=s_val,pos_prev=prev.get(stock, 0.0),sbo=sbo, sso=sso, sbc=sbc, ssc=ssc))
        
        algo_pos.loc[t] = new_pos

    # remove linhas sem s-score algum
    null_idx = s_scores.index[s_scores.isnull().all(axis=1)]
    s_scores = s_scores.drop(index=null_idx)
    betas = betas.drop(index=null_idx)
    algo_pos = algo_pos.drop(index=null_idx)
    
    # pesos iguais por lado, não ter viés direcional do mercado (soma zero)
    algo_weights = algo_pos.apply(equal_weight_by_side, axis=1, result_type="broadcast")
    
    # hedge por PCs, zerar a exposição agregada a cada fator PCA
    hedge = hedge_from_betas(algo_weights, betas, stocks, pcs)
    
    # junta pesos (ações + PCs) e normalização da exposição bruta
    w_all = pd.concat([algo_weights, hedge], axis=1)
    w_all = normalize_gross(w_all)
    
    # retornos utilizados no trade (ações + PCs)
    returns_all = pd.concat([returns, Factor_PCA], axis=1).dropna()
    ret_net, cumret_algo, turnover = compute_pnl_with_costs(
        w_all=w_all,
        returns_mod=returns_all,
        eps_per_turnover=eps_cost,
    )

    # comparação com SPY (buy&hold em retorno simples)
    spy = returns_spy.iloc[252:].copy()
    cumret_spy = (1.0 + spy).cumprod()

    if plot:
        plt.figure(figsize=(18, 6))
        plt.grid(True)
        plt.plot(cumret_algo.index, cumret_algo, label='Algo (PCA-OU)')
        plt.plot(cumret_spy.index,  cumret_spy,  label='SPY')
        plt.legend()
        plt.title(f'Estratégia PCA/OU vs SPY | PCs={num_pc}, s_win={s_win}')
        plt.show()

    return cumret_algo, s_scores


# estatisticas de desempenho
def stats_from_returns(ret):
    ann = 252
    cagr = (1+ret).prod()**(ann/len(ret)) - 1
    vol  = ret.std(ddof=1) * np.sqrt(ann)
    sharpe = cagr/vol if vol>0 else np.nan
    # max drawdown
    cum = (1+ret).cumprod()
    peak = cum.cummax()
    dd = (cum/peak - 1).min()
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MaxDD": dd}