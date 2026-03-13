# Arquivo com todas as funções que serão usadas para a execução da estratégia

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os
import time
import threading
import optuna

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
        sigma_w = Rw[Zw.columns].std(ddof=1).replace(0, np.nan)

        # retornos do dia t
        Rt = returns.loc[t, Zw.columns]
        if Rt.isnull().any():
            continue

        for j in range(n_factors):
            vj = pd.Series(evecs[:, j], index=Zw.columns)
            raw = vj / sigma_w
            raw = raw.replace([np.inf, -np.inf], np.nan).dropna()
            if raw.empty:
                continue

            # reindex para manter o tamanho certo (ativos fora viram 0)
            wj = raw.reindex(Zw.columns).fillna(0.0)
            wj = normalizar_pesos(wj)

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
    if not np.isfinite(a) or not np.isfinite(b) or not (1e-6 < b < 1):
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

# drift
def estimate_ou_from_cumsum_with_drift(
    epsilon,
    trading_days: int = 252,
    ma_window: int = 60,
    eps_kappa_floor: float = 1e-8,
    min_obs: int = 30,
):
    """
    Estima OU no mispricing X_t = cumsum(epsilon_t) e calcula drift alpha (paper).

    Mispricing (nível):
        X_{t+1} = a + b X_t + zeta_{t+1}

    Mapeamento AR(1) -> OU (dt = 1 dia):
        b = exp(-kappa_daily)
        kappa_daily = -log(b)
        m = a/(1-b)
        sigma_eq = sqrt( Var(zeta) / (1 - b^2) )

    Drift (paper):
        alpha_daily = slope( MA_{ma_window}(X_t) )  # inclinação da média móvel do nível X
        (unidade: X por dia)

    Modified s-score (paper):
        s_mod = s - alpha/(kappa * sigma_eq)

    Retorna:
        (m, X_T, kappa_ann, sigma_eq, alpha_daily, adj)
    onde:
        adj = alpha_daily / (kappa_daily * sigma_eq)
    """
    eps = np.asarray(epsilon, dtype=float).ravel()
    if len(eps) < max(min_obs, ma_window + 2) or not np.isfinite(eps).all():
        return None

    # 1) constrói X = cumsum(eps)
    Xk = np.cumsum(eps)
    if len(Xk) < 2 or not np.isfinite(Xk).all():
        return None

    # 2) drift alpha = slope da média móvel de X (interpretação do paper)
    if len(Xk) < ma_window + 2:
        return None

    kernel = np.ones(ma_window, dtype=float) / float(ma_window)
    maX = np.convolve(Xk, kernel, mode="valid")  # comprimento = len(Xk)-ma_window+1

    t = np.arange(len(maX), dtype=float)  # passo de 1 dia
    # slope (X por dia)
    alpha_daily = float(np.polyfit(t, maX, 1)[0])
    if not np.isfinite(alpha_daily):
        return None

    # 3) regressão AR(1): X_{t+1} = a + b X_t + zeta
    X_lag = Xk[:-1].reshape(-1, 1)
    y = Xk[1:]

    model = LinearRegression().fit(X_lag, y)
    a = float(model.intercept_)
    b = float(model.coef_[0])

    # valida b: mean-reverting exige 0<b<1
    if not (np.isfinite(a) and np.isfinite(b) and (1e-6 < b < 1.0 - 1e-9)):
        return None

    # 4) resíduos e variância
    zeta = y - model.predict(X_lag)
    var_zeta = float(np.var(zeta, ddof=1))
    if not np.isfinite(var_zeta) or var_zeta <= 0:
        return None

    # 5) parâmetros OU (dt = 1 dia)
    kappa_daily = float(-np.log(b))               # por dia
    if not np.isfinite(kappa_daily) or kappa_daily <= eps_kappa_floor:
        return None
    kappa_ann = float(kappa_daily * trading_days)

    m = float(a / (1.0 - b))
    sigma_eq = float(np.sqrt(var_zeta / (1.0 - b**2)))
    if not np.isfinite(sigma_eq) or sigma_eq <= 0:
        return None

    X_T = float(Xk[-1])

    # 6) ajuste do modified s-score
    adj = float(alpha_daily / (kappa_daily * sigma_eq))
    if not np.isfinite(adj):
        return None

    return m, X_T, kappa_ann, sigma_eq, alpha_daily, adj

def compute_s_scores_cross_sectional(returns: pd.DataFrame,factors: pd.DataFrame, kappa_min: float = 252.0/30.0, use_drift: bool = True, ma_window: int = 60): 
    """
    - Para cada ação: regressão ação~PCs → resíduos → OU → (a,b,var, kappa, sigma_eq)
    - Calcula m_i = a/(1-b) por ação válida
    - Centraliza m_i: m_i* = m_i - mean_j(m_j)
    - s_i = Xt - m_i* / sigma_eq_i - drift (opcional)
    Retorna:
      s_scores_t (Series por ação), betas_t (dict ação->vetor de betas) de todas as ações naquele dia
    """
    # alinhar datas e colunas
    common_idx = returns.index.intersection(factors.index)
    returns = returns.reindex(index=common_idx)
    factors = factors.reindex(index=common_idx)
    
    stocks = list(returns.columns)
    s_t = pd.Series(index=stocks, dtype=float)
    betas_t, sigma_eq_map, u_map, adj_map = {}, {}, {},{}

    X_df = factors.copy()  # tabela dos retornos dos PCs

    for stock in stocks:
        y_ser = returns[stock]
        df = pd.concat([y_ser.rename("y"), X_df], axis=1).dropna()
        if len(df) < 30: 
            continue

        X = df.drop(columns=["y"]).values
        y = df["y"].values

        beta, eps = regress_action_on_pcs(X, y)

        if use_drift:
            ou = estimate_ou_from_cumsum_with_drift(eps,ma_window=ma_window)
            # retorna: m, X_T, kappa_ann, sigma_eq, alpha_daily, adj
            if ou is None:
                continue
            m, X_T, kappa_ann, sigma_eq, alpha_daily, adj = ou
        else:
            ou = estimate_ou_from_cumsum(eps)  # sua função antiga
            if ou is None:
                continue
            m, X_T, var_zeta, kappa_ann, sigma_eq = ou
            adj = 0.0
        
        # filtro do paper para evitar ações com OU muito lento (kappa baixo)
        if kappa_ann <= kappa_min:
            # reverte devagar demais segundo o critério do paper
            continue

        sigma_eq_map[stock] = sigma_eq
        betas_t[stock] = beta
        adj_map[stock] = adj

        u = X_T - m
        if np.isfinite(u) and np.isfinite(sigma_eq) and sigma_eq > 0:
            u_map[stock] = u

    if not u_map:
        return s_t, betas_t, adj_map  # vazio

    # centralização cross-sectional em U = (X_t - m) 
    u_bar = np.mean(list(u_map.values()))

    for stock, u in u_map.items():
        u_centered = u - u_bar
        #s_val = u / sigma_eq_map[stock]  # s_score do paper
        s_val = u_centered / sigma_eq_map[stock] # s_score do paper centralizado (melhor)
        # drift correction (paper): s_mod = s - alpha/(kappa*sigma_eq)
        s_mod = s_val - adj_map.get(stock, 0.0)

        if np.isfinite(s_mod):
            s_t.loc[stock] = s_mod

    return s_t, betas_t, adj_map

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
    
    m = len(pcs)

    # alinhar datas/colunas
    common_idx = algo_weights.index.intersection(betas.index)
    W = algo_weights.reindex(index=common_idx, columns=stocks).fillna(0.0)
    B = betas.reindex(index=common_idx, columns=stocks)

    # limpar células (aceita list/tuple/np.array; garante shape (m,))
    def clean_cell(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            a = np.asarray(v, dtype=float).ravel()
            if a.shape[0] == m and np.isfinite(a).all():
                return a
        return np.zeros(m, dtype=float)

    # aplica limpeza (coluna a coluna)
    B = B.apply(lambda col: col.map(clean_cell))

    hedge_rows = []
    for dt in common_idx:
        w = W.loc[dt].values  # (n_stocks,)
        b_list = B.loc[dt, stocks].to_list()  # lista de arrays (m,)

        # stack -> (n_stocks, m)
        b_dt = np.vstack(b_list)
        expo = w @ b_dt  # (m,)
        hedge_rows.append(-expo)

    hedge = pd.DataFrame(hedge_rows, index=common_idx, columns=pcs, dtype=float)
    return hedge

def hedge_from_betas_adaptive(
    algo_weights: pd.DataFrame,
    betas: pd.DataFrame,
    stocks,
    max_pcs: int,
    pcs_prefix: str = "eig",
):
    """
    Hedge em PCs com dimensão adaptativa por dia:
      - Em cada dia t, usa o tamanho m_t do beta (quando disponível).
      - Computa hedge_t = - Σ_i w_i(t) * beta_i(t)  (em R^{m_t})
      - Armazena em vetor de tamanho max_pcs, preenchendo [0:m_t] e zerando o resto.

    Retorna:
      hedge: DataFrame [date x eig1..eig(max_pcs)]
      num_pcs_used_in_hedge: Series [date] com m_t usado (diagnóstico)
    """
    pcs_cols = [f"{pcs_prefix}{i+1}" for i in range(max_pcs)]

    common_idx = algo_weights.index.intersection(betas.index)
    W = algo_weights.reindex(index=common_idx, columns=stocks).fillna(0.0)
    B = betas.reindex(index=common_idx, columns=stocks)

    hedge = pd.DataFrame(index=common_idx, columns=pcs_cols, dtype=float)
    m_used = pd.Series(index=common_idx, dtype=float)

    for dt in common_idx:
        w = W.loc[dt]  # Series

        # Descobrir m_t a partir do primeiro beta válido do dia
        m_t = None
        for stock in stocks:
            v = B.loc[dt, stock]
            if isinstance(v, (list, tuple, np.ndarray)):
                a = np.asarray(v, dtype=float).ravel()
                if a.size > 0 and np.isfinite(a).all():
                    m_t = int(a.size)
                    break

        if m_t is None:
            hedge.loc[dt] = np.zeros(max_pcs, dtype=float)
            m_used.loc[dt] = np.nan
            continue

        m_t = min(m_t, max_pcs)
        expo = np.zeros(m_t, dtype=float)

        # soma exposições
        for stock in stocks:
            wi = float(w.get(stock, 0.0))
            v = B.loc[dt, stock]

            if wi == 0.0:
                continue

            if isinstance(v, (list, tuple, np.ndarray)):
                a = np.asarray(v, dtype=float).ravel()
                if a.size >= m_t and np.isfinite(a[:m_t]).all():
                    expo += wi * a[:m_t]

        hedge_row = np.zeros(max_pcs, dtype=float)
        hedge_row[:m_t] = -expo
        hedge.loc[dt] = hedge_row
        m_used.loc[dt] = m_t

    return hedge, m_used

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

def compute_pca_factor_returns_adaptive(
    returns: pd.DataFrame,
    window_pca: int = 60,
    variance_target: float = 0.60,
    min_factors: int = 5,
    max_factors: int = 35,
):
    """
    PCA com número ADAPTATIVO de fatores baseado em variância explicada.
    
    Retorna
    -------
    Factor_PCA : DataFrame [date x eig1, eig2, ..., eig_max]
        Colunas além de num_factors_t ficam NaN
    num_factors_used : Series [date]
        Número de fatores usados em cada dia
    """
    dates = returns.index
    F = pd.DataFrame(index=dates, columns=[f"eig{i+1}" for i in range(max_factors)], dtype=float)
    num_factors_series = pd.Series(index=dates, dtype=int)
    
    for i in range(window_pca, len(dates)):
        t_hist = dates[i - window_pca : i]
        t = dates[i]
        
        Rw = returns.loc[t_hist]
        Zw = padronizar_janela(Rw)
        Zw = Zw.dropna(axis=1, how='any').dropna(axis=0, how='all')
        
        if Zw.shape[1] < min_factors + 1:
            continue
        
        # PCA
        evals, evecs = decompor_corr(Zw)
        
        # Determinar número de fatores para este dia
        var_cumsum = np.cumsum(evals) / np.sum(evals)
        n_factors = np.argmax(var_cumsum >= variance_target) + 1
        n_factors = np.clip(n_factors, min_factors, max_factors)
        
        num_factors_series.loc[t] = n_factors
        
        # std dos ativos
        sigma_w = Rw[Zw.columns].std(ddof=1).replace(0, np.nan)
        Rt = returns.loc[t, Zw.columns]
        
        if Rt.isnull().any():
            continue
        
        # Calcular retornos apenas dos n_factors primeiros
        for j in range(n_factors):
            vj = pd.Series(evecs[:, j], index=Zw.columns)
            raw = vj / sigma_w
            raw = raw.replace([np.inf, -np.inf], np.nan).dropna()
            
            if raw.empty:
                continue
            
            wj = raw.reindex(Zw.columns).fillna(0.0)
            wj = normalizar_pesos(wj)
            
            F.loc[t, f"eig{j+1}"] = float(Rt.dot(wj))
    
    return F.dropna(how="all"), num_factors_series.dropna()

def compute_stock_specific_thresholds(
    s_scores_hist: pd.DataFrame,
    window: int = 252,
    percentile_open: float = 0.15,
    percentile_close_short: float = 0.35,
    percentile_close_long: float = 0.45,
    min_sbo: float = 1.0,
    min_sso: float = 1.0,
    min_sbc: float = 0.6,
    max_ssc: float = -0.4,
):
    """
    Calcula thresholds adaptativos específicos para cada ação.
    """
    # Percentis rolling por ação (coluna a coluna)
    sbo = s_scores_hist.rolling(window, min_periods=60).quantile(percentile_open).abs()
    sso = s_scores_hist.rolling(window, min_periods=60).quantile(1 - percentile_open)
    sbc = s_scores_hist.rolling(window, min_periods=60).quantile(1 - percentile_close_short)
    ssc = -s_scores_hist.rolling(window, min_periods=60).quantile(percentile_close_long)
    
    # Aplicar limites mínimos/máximos
    sbo = sbo.clip(lower=min_sbo).fillna(1.25)
    sso = sso.clip(lower=min_sso).fillna(1.25)
    sbc = sbc.clip(lower=min_sbc).fillna(0.75)
    ssc = ssc.clip(upper=max_ssc).fillna(-0.50)
    
    return {'sbo': sbo, 'sso': sso, 'sbc': sbc, 'ssc': ssc}

# =============================================================================
# Função principal (backtest) sem comentário - mais rápida
# =============================================================================
def pca_portfolio_hedge(
    returns: pd.DataFrame,
    returns_bench: pd.DataFrame,
    benchmark: str = "SPY",
    num_pc: int = 15,
    s_win: int = 60,
    # thresholds do paper:
    sbo: float = 1.25,
    sso: float = 1.25,
    sbc: float = 0.75,
    ssc: float = 0.50,
    eps_cost: float = 0.0005,
    rebalanceamento_dias: int = 1,
    kappa_min: float = 252.0/30.0,
    plot: bool = True,
    use_drift: bool = True,
    ma_window: int = 60,
    verbose: bool = True
):
    
    # Fatores PCA (rolling) com janela de 60 dias
    Factor_PCA = compute_pca_factor_returns(
    returns, window_pca=60, n_factors=num_pc)
    
    pcs = [f"eig{i+1}" for i in range(num_pc)]
    stocks = [c for c in returns.columns]
    usable_index = returns.iloc[s_win:].index

    # tabelas
    s_scores = pd.DataFrame(index=usable_index, columns=stocks, dtype=float)
    betas = pd.DataFrame(index=usable_index, columns=stocks, dtype=object)
    algo_pos = pd.DataFrame(index=usable_index, columns=stocks, dtype=float)
    
    # ------------- loop temporal -------------
    for t in usable_index:
        if verbose:
            print(f"Tempo : {t}")
            
        # janela [t-s_win, t] para estimação OU
        ret = returns.loc[:t].iloc[-s_win:].copy()
        factor = Factor_PCA.loc[:t].iloc[-s_win:].copy()
        
        # checagem: PCs não podem ter NaN nessa janela padronizada
        if factor[pcs].isnull().any().any():
            continue

        # s-scores para o dia t (com centralização) + betas para hedge
        s_t, betas_t, adj_map = compute_s_scores_cross_sectional(
            returns=ret,
            factors=factor,
            kappa_min=kappa_min,
            use_drift=use_drift,
            ma_window=ma_window
        )

        # guarda s-scores e betas válidos
        s_scores.loc[t, s_t.index] = s_t
        for k, v in betas_t.items():
            betas.loc[t, k] = v

        # atualiza posições discretas com base no s-score de cada ação
        prev = algo_pos.shift(1).loc[t]
        if prev.isna().all():
            prev = pd.Series(0.0, index=stocks) #caso inicial

        # --- REBALANCEAMENTO A CADA x DIAS ÚTEIS ---
        day_idx = algo_pos.index.get_loc(t)

        if day_idx % rebalanceamento_dias == 0:
            # recalcula posições
            new_pos = []
            for stock in stocks:
                s_val = s_t.get(stock, np.nan)
                new_pos.append(position_from_s(
                    s=s_val,
                    pos_prev=prev.get(stock, 0.0),
                    sbo=sbo, sso=sso, sbc=sbc, ssc=ssc
                ))
            algo_pos.loc[t] = new_pos

        else:
            # mantém a posição anterior (sem trades)
            algo_pos.loc[t] = prev

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
    returns_all = pd.concat([returns, Factor_PCA], axis=1).fillna(0.0)
    ret_net, cumret_algo, turnover = compute_pnl_with_costs(
        w_all=w_all,
        returns_mod=returns_all,
        eps_per_turnover=eps_cost,
    )

    # comparação com SPY (buy&hold em retorno simples)
    bench = returns_bench.iloc[s_win:].copy()
    cumret_bench = (1.0 + bench).cumprod()

    if plot:
        plt.figure(figsize=(18, 6))
        plt.grid(True)
        plt.plot(cumret_algo.index, cumret_algo, label='Algo (PCA-OU)')
        plt.plot(cumret_bench.index,  cumret_bench,  label=benchmark)
        plt.legend()
        plt.title(f'Estratégia PCA/OU vs {benchmark}')
        plt.show()

    return {
        'cumret_algo': cumret_algo,
        's_scores': s_scores,
        'algo_weights': algo_weights,
        "w_all": w_all,  
        'betas': betas,                 
        'ret_net': ret_net,             
        'Factor_PCA': Factor_PCA,       
        'pcs': pcs,                     
        'turnover': turnover,
        'adj_map': adj_map
    }

def pca_portfolio_quantil(
    returns: pd.DataFrame,
    returns_bench: pd.DataFrame,
    benchmark: str = "SPY",
    num_pc: int = 15,
    s_win: int = 60,
    # parâmetros para thresholds adaptativos
    adaptive_window: int = 60,
    percentile_open: float = 0.15,
    percentile_close_short: float = 0.35,
    percentile_close_long: float = 0.45,
    eps_cost: float = 0.0005,
    rebalanceamento_dias: int = 1,
    kappa_min: float = 252.0/30.0,
    plot: bool = True,
    use_drift: bool = True,
    ma_window: int = 60
):
    
    # Fatores PCA (rolling) com janela de 60 dias
    Factor_PCA = compute_pca_factor_returns(
    returns, window_pca=60, n_factors=num_pc)
    
    pcs = [f"eig{i+1}" for i in range(num_pc)]
    stocks = [c for c in returns.columns]
    usable_index = returns.iloc[s_win:].index

    # tabelas
    s_scores = pd.DataFrame(index=usable_index, columns=stocks, dtype=float)
    betas = pd.DataFrame(index=usable_index, columns=stocks, dtype=object)
    algo_pos = pd.DataFrame(index=usable_index, columns=stocks, dtype=float)
    
    # ------------- loop temporal -------------
    for t in usable_index:
        print(f"Tempo : {t}")
        # janela [t-s_win, t] para estimação OU
        ret = returns.loc[:t].iloc[-s_win:].copy()
        factor = Factor_PCA.loc[:t].iloc[-s_win:].copy()
        
        # checagem: PCs não podem ter NaN nessa janela padronizada
        if factor[pcs].isnull().any().any():
            continue

        # s-scores para o dia t (com centralização) + betas para hedge
        s_t, betas_t, adj_map = compute_s_scores_cross_sectional(
            returns=ret,
            factors=factor,
            kappa_min=kappa_min,
            use_drift=use_drift,
            ma_window=ma_window
        )

        # guarda s-scores e betas válidos
        s_scores.loc[t, s_t.index] = s_t
        for k, v in betas_t.items():
            betas.loc[t, k] = v

        # atualiza posições discretas com base no s-score de cada ação
        prev = algo_pos.shift(1).loc[t]
        if prev.isna().all():
            prev = pd.Series(0.0, index=stocks) #caso inicial

        # --- REBALANCEAMENTO A CADA x DIAS ÚTEIS ---
        day_idx = algo_pos.index.get_loc(t)

        if day_idx % rebalanceamento_dias == 0:
            #  CALCULAR THRESHOLDS (adaptativo ou fixo)
            
            # Usar histórico de s-scores até t (inclusive)
            s_hist = s_scores.loc[:t]
            
            # Calcular thresholds adaptativos
            thresh_dict = compute_stock_specific_thresholds(
                s_scores_hist=s_hist,
                window=adaptive_window,
                percentile_open=percentile_open,
                percentile_close_short=percentile_close_short,
                percentile_close_long=percentile_close_long,
            )
            
            # Pegar thresholds do dia t (última linha)
            sbo_t = thresh_dict['sbo'].loc[t]
            sso_t = thresh_dict['sso'].loc[t]
            sbc_t = thresh_dict['sbc'].loc[t]
            ssc_t = thresh_dict['ssc'].loc[t]
                
            # recalcula posições
            new_pos = []
            for stock in stocks:
                s_val = s_t.get(stock, np.nan)
                # Thresholds
                sbo_stock = sbo_t.get(stock, 1.25)
                sso_stock = sso_t.get(stock, 1.25)
                sbc_stock = sbc_t.get(stock, 0.50)
                ssc_stock = ssc_t.get(stock, -0.50)

                new_pos.append(position_from_s(
                    s=s_val,
                    pos_prev=prev.get(stock, 0.0),
                    sbo=sbo_stock,
                    sso=sso_stock,
                    sbc=sbc_stock,
                    ssc=abs(ssc_stock),  
                ))

            algo_pos.loc[t] = new_pos

        else:
            # mantém a posição anterior (sem trades)
            algo_pos.loc[t] = prev

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
    returns_all = pd.concat([returns, Factor_PCA], axis=1).fillna(0.0)
    
    ret_net, cumret_algo, turnover = compute_pnl_with_costs(
        w_all=w_all,
        returns_mod=returns_all,
        eps_per_turnover=eps_cost,
    )

    # comparação com SPY (buy&hold em retorno simples)
    bench = returns_bench.iloc[s_win:].copy()
    cumret_bench = (1.0 + bench).cumprod()

    if plot:
        plt.figure(figsize=(18, 6))
        plt.grid(True)
        plt.plot(cumret_algo.index, cumret_algo, label='Algo (PCA-OU)')
        plt.plot(cumret_bench.index,  cumret_bench,  label=benchmark)
        plt.legend()
        plt.title(f'Estratégia PCA/OU vs {benchmark}')
        plt.show()

    return {
        'cumret_algo': cumret_algo,
        's_scores': s_scores,
        'algo_weights': algo_weights,
        "w_all": w_all,
        'betas': betas,                 
        'ret_net': ret_net,             
        'Factor_PCA': Factor_PCA,       
        'pcs': pcs,                     
        'turnover': turnover,
        'adj_map': adj_map
    }

def pca_portfolio_adaptive_pcs(
    returns: pd.DataFrame,
    returns_bench: pd.DataFrame,
    benchmark: str = "SPY",
    variance_target: float = 0.60,  
    min_pcs: int = 5,
    max_pcs: int = 35,
    s_win: int = 70,
    # thresholds adaptativos
    adaptive_thresholds: bool = False,
    adaptive_window: int = 60,
    percentile_open: float = 0.15,
    percentile_close_short: float = 0.35,
    percentile_close_long: float = 0.45,
    # thresholds fixos
    sbo: float = 1.25,
    sso: float = 1.25,
    sbc: float = 0.50,
    ssc: float = 0.50,
    eps_cost: float = 0.0005,
    rebalanceamento_dias: int = 1,
    kappa_min: float = 252.0/30.0,
    plot: bool = True,
    use_drift: bool = True,
    ma_window: int = 60
):
    """
    Backtest com número de PCs ADAPTATIVO (varia ao longo do tempo).
    """
    
    # Fatores PCA ADAPTATIVOS
    Factor_PCA, num_pcs_used = compute_pca_factor_returns_adaptive(
        returns,
        window_pca=60,
        variance_target=variance_target,
        min_factors=min_pcs,
        max_factors=max_pcs,
    )
    
    stocks = [c for c in returns.columns]
    pcs_all = [f"eig{i+1}" for i in range(max_pcs)]
    usable_index = returns.iloc[s_win:].index
    
    # tabelas
    s_scores = pd.DataFrame(index=usable_index, columns=stocks, dtype=float)
    betas = pd.DataFrame(index=usable_index, columns=stocks, dtype=object)
    algo_pos = pd.DataFrame(index=usable_index, columns=stocks, dtype=float)
    
    # hedge em dimensão fixa max_pcs (preenche só [0:m_t] por dia)
    hedge_pcs = pd.DataFrame(index=usable_index, columns=pcs_all, dtype=float)

    # ------------- loop temporal -------------
    for t in usable_index:
        print(f"Tempo : {t}")
        
        # Número de PCs usado neste dia
        num_pc_t = num_pcs_used.get(t, np.nan)
        if pd.isna(num_pc_t):
            num_pc_t = max_pcs
        
        num_pc_t = int(np.clip(int(num_pc_t), min_pcs, max_pcs))
        pcs_t = [f"eig{i+1}" for i in range(num_pc_t)]
        
        # janela para OU/regressões até t [t-s_win, t] 
        ret = returns.loc[:t].iloc[-s_win:].copy()
        factor = Factor_PCA.loc[:t, pcs_t].iloc[-s_win:].copy()
        # checagem: PCs não podem ter NaN nessa janela
        factor = factor.dropna(axis=1, how="any")
        
        # segurança extra: padronização pode gerar NaN se std=0
        #factor = factor.dropna(axis=1, how="any")
        if factor.shape[1] < min_pcs:
            algo_pos.loc[t] = prev
            hedge_pcs.loc[t] = hedge_pcs.shift(1).loc[t] if t != usable_index[0] else np.zeros(max_pcs, dtype=float)
            continue
        
        # s-scores
        s_t, betas_t, adj_map = compute_s_scores_cross_sectional(
            returns=ret,
            factors=factor,
            kappa_min=kappa_min,
            use_drift=use_drift,
            ma_window=ma_window
        )
        
        # guarda s-scores e betas válidos
        s_scores.loc[t, s_t.index] = s_t
        for k, v in betas_t.items():
            betas.loc[t, k] = v
        
        # atualiza posições
        prev = algo_pos.shift(1).loc[t]
        if prev.isna().all():
            prev = pd.Series(0.0, index=stocks)
        
        # REBALANCEAMENTO
        day_idx = algo_pos.index.get_loc(t)
        
        if day_idx % rebalanceamento_dias == 0:
            # Thresholds (adaptativo ou fixo)
            if adaptive_thresholds:
                s_hist = s_scores.loc[:t]
                thresh_dict = compute_stock_specific_thresholds(
                    s_scores_hist=s_hist,
                    window=adaptive_window,
                    percentile_open=percentile_open,
                    percentile_close_short=percentile_close_short,
                    percentile_close_long=percentile_close_long,
                )
                sbo_t = thresh_dict['sbo'].loc[t]
                sso_t = thresh_dict['sso'].loc[t]
                sbc_t = thresh_dict['sbc'].loc[t]
                ssc_t = thresh_dict['ssc'].loc[t]
            else:
                sbo_t = pd.Series(sbo, index=stocks)
                sso_t = pd.Series(sso, index=stocks)
                sbc_t = pd.Series(sbc, index=stocks)
                ssc_t = pd.Series(-ssc, index=stocks)
            
            # Recalcula posições
            new_pos = []
            for stock in stocks:
                s_val = s_t.get(stock, np.nan)
                sbo_stock = sbo_t.get(stock, 1.25)
                sso_stock = sso_t.get(stock, 1.25)
                sbc_stock = sbc_t.get(stock, 0.50)
                ssc_stock = ssc_t.get(stock, -0.50)
                
                new_pos.append(position_from_s(
                    s=s_val,
                    pos_prev=prev.get(stock, 0.0),
                    sbo=sbo_stock,
                    sso=sso_stock,
                    sbc=sbc_stock,
                    ssc=abs(ssc_stock),
                ))
            
            algo_pos.loc[t] = new_pos
        else:
            algo_pos.loc[t] = prev
        
        # HEDGE DO DIA (com PCs adaptativos)
        # =========================
        # pesos das ações no dia t (igual-weight por lado)
        w_stocks_t = equal_weight_by_side(algo_pos.loc[t])

        # exposição agregada aos PCs do dia: expo = Σ_i w_i * beta_i
        expo = np.zeros(num_pc_t, dtype=float)

        for stock, beta_vec in betas_t.items():
            wi = float(w_stocks_t.get(stock, 0.0))
            if wi == 0.0:
                continue

            b = np.asarray(beta_vec, dtype=float).ravel()
            if b.shape[0] != num_pc_t:
                continue
            if not np.isfinite(b).all():
                continue

            expo += wi * b

        hedge_row = np.zeros(max_pcs, dtype=float)
        hedge_row[:num_pc_t] = -expo
        hedge_pcs.loc[t, pcs_all] = hedge_row

    
    # remove linhas sem s-score
    null_idx = s_scores.index[s_scores.isnull().all(axis=1)]
    s_scores = s_scores.drop(index=null_idx)
    betas = betas.drop(index=null_idx)
    algo_pos = algo_pos.drop(index=null_idx)
    hedge_pcs = hedge_pcs.drop(index=null_idx)

    # pesos e PnL
    algo_weights = algo_pos.apply(equal_weight_by_side, axis=1, result_type="broadcast")
    w_all = pd.concat([algo_weights, hedge_pcs], axis=1)
    w_all = normalize_gross(w_all)
    
    returns_all = pd.concat([returns, Factor_PCA], axis=1).fillna(0.0)

    ret_net, cumret_algo, turnover = compute_pnl_with_costs(
        w_all=w_all,
        returns_mod=returns_all,
        eps_per_turnover=eps_cost,
    )
    
    # comparação com SPY
    bench = returns_bench.reindex(cumret_algo.index).fillna(0.0)
    cumret_bench = (1.0 + bench).cumprod()
    
    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(18, 10))
        
        # Plot 1: Performance
        axes[0].plot(cumret_algo.index, cumret_algo, label='Algo (PCA Adaptativo)', linewidth=2)
        axes[0].plot(cumret_bench.index, cumret_bench, label=benchmark, linewidth=1.5, alpha=0.7)
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_title(f'Estratégia PCA/OU ADAPTATIVO vs {benchmark} | Target Var={variance_target*100:.0f}%')
        
        # Plot 2: Evolução do número de PCs
        num_pcs_used.reindex(cumret_algo.index).plot(ax=axes[1], linewidth=2, color='darkgreen')
        axes[1].set_title('Número de PCs Usado ao Longo do Tempo', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('# PCs')
        axes[1].axhline(num_pcs_used.mean(), color='red', linestyle='--', 
                        label=f'Média: {num_pcs_used.mean():.1f}')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    return {
        "cumret_algo": cumret_algo,
        "cumret_bench": cumret_bench,
        "s_scores": s_scores,
        "algo_weights": algo_weights,
        "hedge_pcs": hedge_pcs,
        "w_all": w_all,
        "algo_pos": algo_pos,
        "betas": betas,
        "ret_net": ret_net,
        "Factor_PCA": Factor_PCA,
        "num_pcs_used": num_pcs_used,
        "turnover": turnover,
        'adj_map': adj_map
    }

# =============================================================================
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



# =============================================================================
# Otimização de parâmetros (OPTUNA)
# --------------------------
# Métrica
# --------------------------
def sharpe_ann(r: pd.Series, trading_days: int = 252) -> float:
    r = r.dropna()
    if len(r) < 60:
        return np.nan
    vol = r.std(ddof=1)
    if vol == 0 or not np.isfinite(vol):
        return np.nan
    return float(np.sqrt(trading_days) * r.mean() / vol)


def score_strategy(ret_net: pd.Series, trading_days: int = 252) -> float:
    r = ret_net.dropna()
    if len(r) < 60:
        return -np.inf
    sh = sharpe_ann(r, trading_days=trading_days)
    if np.isnan(sh):
        return -np.inf
    return sh

def splits(
    index: pd.Index,
    *,
    n_folds: int = 4,
    window_size: int = 252 * 2,          # tamanho da janela (ex.: 2 anos)
    step_size: int | None = None,        # avanço entre janelas (ex.: 1 ano)
) -> list[tuple[int, int]]:
    """
    Retorna lista de (start, end) como posições inteiras.
    """
    n = len(index)
    if n_folds <= 0 or window_size <= 0:
        return []

    if step_size is None:
        step_size = window_size // 2  # default razoável: meia janela
        if step_size <= 0:
            step_size = 1

    split: list[tuple[int, int]] = []
    start = 0

    for _ in range(n_folds):
        end = start + window_size
        if end > n:
            break
        split.append((start, end))
        start += step_size

    return split


# --------------------------
# Avaliação por folds
# --------------------------
def eval_params(
    returns: pd.DataFrame,
    returns_bench: pd.Series | pd.DataFrame,
    params: dict,
    trial: optuna.Trial | None = None,
    *,
    n_folds: int = 4,
    window_size: int = 252 * 2,
    step_size: int | None = None,
    min_points: int = 60,   # mínimo para considerar fold válido
    apply_purge_in_score: bool = True,
) -> float:
    """
    - Roda o backtest em cada janela (fold) e faz score na própria janela.
    """
    # --------------------------
    # alinhamento global
    # --------------------------
    if isinstance(returns_bench, pd.Series):
        common = returns.index.intersection(returns_bench.index)
        returns = returns.loc[common]
        returns_bench = returns_bench.loc[common]
    else:
        common = returns.index.intersection(returns_bench.index)
        returns = returns.loc[common]
        returns_bench = returns_bench.loc[common]

    if len(returns) < min_points:
        return -np.inf

    # purge depende das janelas do método
    s_win = int(params["s_win"])
    ma_window = int(params["ma_window"])
    purge = max(s_win, ma_window) + 5  

    # gerar folds 
    folds = splits(
        returns.index,
        n_folds=n_folds,
        window_size=window_size,
        step_size=step_size,
    )
    if not folds:
        return -np.inf

    fold_scores: list[float] = []

    for k, (start, end) in enumerate(folds):
        sub_returns = returns.iloc[start:end]
        sub_bench = returns_bench.iloc[start:end]

        # alinhamento dentro do fold
        if isinstance(sub_bench, pd.Series):
            aligned = sub_returns.join(sub_bench.rename("bench"), how="inner")
            sub_returns = aligned[sub_returns.columns]
            sub_bench = aligned["bench"]
        else:
            aligned = sub_returns.join(sub_bench, how="inner")
            sub_returns = aligned[sub_returns.columns]
            sub_bench = aligned[sub_bench.columns]

        if len(aligned) < max(min_points, purge + 1):
            return -np.inf

        # roda o backtest no fold (somente treino)
        try:
            res = pca_portfolio_hedge(
                returns=sub_returns,
                returns_bench=sub_bench,
                benchmark=params.get("benchmark", "BENCH"),
                num_pc=int(params["num_pc"]),
                s_win=s_win,
                sbo=float(params["sbo"]),
                sso=float(params["sso"]),
                sbc=float(params["sbc"]),
                ssc=float(params["ssc"]),
                eps_cost=float(params["eps_cost"]),
                rebalanceamento_dias=int(params["rebalanceamento_dias"]),
                kappa_min=float(params["kappa_min"]),
                plot=False,
                use_drift=bool(params["use_drift"]),
                ma_window=ma_window,
                verbose=False,
            )
        except Exception:
            return -np.inf

        # score no próprio treino do fold
        ret_net_fold = res["ret_net"].reindex(aligned.index).dropna()
        if apply_purge_in_score and purge > 0:
            ret_net_fold = ret_net_fold.iloc[purge:]  # queima começo do fold na pontuação

        if len(ret_net_fold) < min_points:
            return -np.inf

        fold_score = score_strategy(ret_net_fold)
        fold_scores.append(float(fold_score))

        # pruning com Optuna
        if trial is not None:
            trial.report(float(np.mean(fold_scores)), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return float(np.mean(fold_scores))


# --------------------------
# Optuna
# --------------------------
def optimize_optuna(
    returns: pd.DataFrame,
    returns_bench: pd.Series | pd.DataFrame,
    *,
    n_trials: int = 60,
    n_folds: int = 4,
    train_size: int = 252 * 2,
    test_size: int = 252,
    step_size: int | None = None,
    seed: int = 42,
    study_name: str = "pca_hedge_long_v1",
    storage: str = "sqlite:///optuna_br.db",
):
    """
    Otimiza parâmetros maximizando Sharpe médio OOS em rolling walk-forward.
    """

    def objective(trial: optuna.Trial) -> float:
        now = time.strftime("%H:%M:%S")
        pid = os.getpid()
        tname = threading.current_thread().name
        print(f"[{now}] START trial={trial.number} pid={pid} thread={tname}")

        # --------------------------
        # Hiperparâmetros
        # --------------------------
        s_win = trial.suggest_int("s_win", 60, 250, step=5)
        ma_window = trial.suggest_int("ma_window", 20, s_win - 2, step=2)

        sbo = trial.suggest_float("sbo", 0.8, 2.5)
        sso = trial.suggest_float("sso", 0.8, 2.5)

        sbc = trial.suggest_float("sbc", 0.1, sso - 0.05)  # close short < open short
        ssc = trial.suggest_float("ssc", 0.1, sbo - 0.05)  # close long  < open long

        kappa_min = trial.suggest_float("kappa_min", 1.0, 20.0)
        num_pc = trial.suggest_int("num_pc", 1, 12)
        rebalanceamento_dias = trial.suggest_int("rebalanceamento_dias", 1, 30)

        params = dict(
            benchmark="IBOV",
            num_pc=num_pc,
            s_win=s_win,
            ma_window=ma_window,
            sbo=sbo,
            sso=sso,
            sbc=sbc,
            ssc=ssc,
            eps_cost=0.0005,
            rebalanceamento_dias=rebalanceamento_dias,
            kappa_min=kappa_min,
            plot=False,
            use_drift=True,
            verbose=False,
        )

        return eval_params(
            returns,
            returns_bench,
            params,
            trial=trial,
            n_folds=n_folds,
            train_size=train_size,
            test_size=test_size,
            step_size=step_size,
        )

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=1,
        max_resource=n_folds,
        reduction_factor=3,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    return study