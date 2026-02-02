# analise_estrategia.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# =============================================================================
# TESTE 1: Exposição Beta Agregada aos Fatores PCA
# =============================================================================

def compute_beta_exposure(algo_weights, betas, pcs):
    """
    Calcula exposição agregada da carteira aos fatores PCA ao longo do tempo.
    
    Parâmetros
    ----------
    algo_weights : pd.DataFrame
        Pesos da carteira [date x stocks]
    betas : pd.DataFrame
        Betas de cada ação aos fatores PCA [date x stocks], cada célula é array
    pcs : list
        Lista de nomes dos fatores ['eig1', 'eig2', ...]
    
    Retorna
    -------
    expo_df : pd.DataFrame
        Exposição agregada [date x pcs]
    """
    expo_list = []
    
    for dt in algo_weights.index:
        w = algo_weights.loc[dt].dropna()
        
        if w.empty:
            expo_list.append(np.zeros(len(pcs)))
            continue
        
        beta_dt = betas.loc[dt, w.index]
        
        # Limpar cells que não são arrays válidos
        valid_betas = []
        valid_weights = []
        for stock in w.index:
            b = beta_dt[stock]
            if isinstance(b, np.ndarray) and b.shape == (len(pcs),) and np.isfinite(b).all():
                valid_betas.append(b)
                valid_weights.append(w[stock])
        
        if valid_betas:
            B = np.vstack(valid_betas)  # shape: (n_stocks, m)
            W = np.array(valid_weights)  # shape: (n_stocks,)
            expo = W @ B  # shape: (m,) → exposição agregada
        else:
            expo = np.zeros(len(pcs))
        
        expo_list.append(expo)
    
    expo_df = pd.DataFrame(expo_list, index=algo_weights.index, columns=pcs)
    return expo_df


def plot_beta_exposure(expo_df, title_prefix=""):
    """
    Plota a evolução temporal da exposição aos fatores PCA.
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    
    # Plot 1: Fator dominante (eigen1)
    expo_df['eig1'].plot(ax=axes[0], color='navy', linewidth=1.5)
    axes[0].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[0].set_title(f'{title_prefix}Exposição ao Eigen1 (Fator Dominante)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Beta Agregado')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Fatores secundários (2-4)
    expo_df[['eig2', 'eig3', 'eig4']].plot(ax=axes[1], linewidth=1.2)
    axes[1].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_title(f'{title_prefix}Exposição aos Eigen2-4', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Beta Agregado')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Exposição total (soma dos valores absolutos)
    total_expo = expo_df.abs().sum(axis=1)
    total_expo.plot(ax=axes[2], color='darkred', linewidth=1.5)
    axes[2].set_title(f'{title_prefix}Exposição Total (Σ|β|)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Soma |Betas|')
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Heatmap dos primeiros 10 fatores
    expo_sample = expo_df[expo_df.columns[:10]].T
    im = axes[3].imshow(expo_sample.values, aspect='auto', cmap='RdBu_r', 
                         vmin=-0.3, vmax=0.3, interpolation='nearest')
    axes[3].set_yticks(range(10))
    axes[3].set_yticklabels(expo_sample.index)
    axes[3].set_title(f'{title_prefix}Heatmap de Exposição (Top 10 Fatores)', fontsize=12, fontweight='bold')
    axes[3].set_xlabel('Tempo')
    plt.colorbar(im, ax=axes[3], label='Beta')
    
    plt.tight_layout()
    plt.show()
    
    return fig


def beta_exposure_statistics(expo_df):
    """
    Calcula estatísticas descritivas da exposição beta.
    """
    stats = pd.DataFrame({
        'Mean': expo_df.mean(),
        'Std': expo_df.std(),
        'Min': expo_df.min(),
        'Max': expo_df.max(),
        'Mean_Abs': expo_df.abs().mean(),
    })
    
    print("\n" + "="*70)
    print("ESTATÍSTICAS DE EXPOSIÇÃO BETA AGREGADA")
    print("="*70)
    print(stats.round(4))
    print("\n📊 Interpretação:")
    print(f"  • Exposição média total: {expo_df.abs().sum(axis=1).mean():.4f}")
    print(f"  • Exposição mediana total: {expo_df.abs().sum(axis=1).median():.4f}")
    print(f"  • Maior exposição absoluta: {expo_df.abs().max().max():.4f} (fator: {expo_df.abs().max().idxmax()})")
    
    # Diagnóstico
    max_expo = expo_df.abs().max().max()
    if max_expo < 0.1:
        print("\n✅ EXCELENTE: Exposição muito baixa (< 0.1) → Boa neutralidade!")
    elif max_expo < 0.3:
        print("\n⚠️  ACEITÁVEL: Exposição moderada (< 0.3) → Neutralidade razoável")
    else:
        print("\n❌ PROBLEMA: Exposição alta (> 0.3) → Viés direcional significativo!")
    
    return stats


# =============================================================================
# TESTE 2: Regressão do PnL contra Fatores PCA
# =============================================================================

def regress_pnl_on_factors(ret_net, Factor_PCA, pcs):
    """
    Regride retornos da estratégia contra retornos dos fatores PCA.
    Mede quanto do PnL é explicado por exposição aos fatores.
    
    Retorna
    -------
    results : dict
        {'r2': float, 'betas': pd.Series, 'alpha_ann': float}
    """
    # Alinhar datas
    common_idx = ret_net.index.intersection(Factor_PCA.index)
    y = ret_net.loc[common_idx].values
    X = Factor_PCA.loc[common_idx, pcs].values
    
    # Verificar se há dados suficientes
    if len(y) < 50 or X.shape[1] == 0:
        print("⚠️  Dados insuficientes para regressão!")
        return None
    
    # Regressão linear
    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)
    betas_strategy = pd.Series(model.coef_, index=pcs)
    alpha_daily = model.intercept_
    alpha_ann = alpha_daily * 252  # anualizado
    
    results = {
        'r2': r2,
        'betas': betas_strategy,
        'alpha_ann': alpha_ann,
        'alpha_daily': alpha_daily,
        'model': model,
    }
    
    return results


def plot_pnl_regression(results, title_prefix=""):
    """
    Visualiza resultados da regressão PnL vs Fatores.
    """
    if results is None:
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Betas da estratégia
    betas = results['betas']
    betas.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
    axes[0].axhline(0, color='red', linestyle='--', alpha=0.7)
    axes[0].set_title(f'{title_prefix}Exposição da Estratégia aos Fatores PCA (Betas)', 
                      fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Beta')
    axes[0].set_xlabel('Fator PCA')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Magnitude absoluta dos betas (top 10)
    betas_abs = betas.abs().sort_values(ascending=False).head(10)
    betas_abs.plot(kind='barh', ax=axes[1], color='coral', edgecolor='black')
    axes[1].set_title(f'{title_prefix}Top 10 Fatores com Maior Exposição (|Beta|)', 
                      fontsize=13, fontweight='bold')
    axes[1].set_xlabel('|Beta|')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    
    return fig


def pnl_regression_statistics(results):
    """
    Imprime estatísticas da regressão PnL vs Fatores.
    """
    if results is None:
        return
    
    r2 = results['r2']
    alpha_ann = results['alpha_ann']
    betas = results['betas']
    
    print("\n" + "="*70)
    print("REGRESSÃO: RETORNO DA ESTRATÉGIA vs FATORES PCA")
    print("="*70)
    print(f"  R² (variância explicada por fatores): {r2:.4f} ({r2*100:.2f}%)")
    print(f"  Alpha anualizado (excesso de retorno):  {alpha_ann:.4f} ({alpha_ann*100:.2f}%)")
    print(f"  Alpha diário médio:                      {results['alpha_daily']:.6f}")
    print(f"\n  Beta médio (|valor|):                    {betas.abs().mean():.4f}")
    print(f"  Beta máximo (|valor|):                   {betas.abs().max():.4f} (fator: {betas.abs().idxmax()})")
    
    print("\n📊 Interpretação:")
    if r2 < 0.05:
        print("  ✅ EXCELENTE: R² < 5% → Estratégia é genuinamente market-neutral!")
        print("     A maior parte do retorno vem de alpha idiossincrático.")
    elif r2 < 0.15:
        print("  ⚠️  ACEITÁVEL: R² < 15% → Alguma exposição aos fatores, mas controlada.")
        print("     Alpha ainda domina o retorno.")
    else:
        print("  ❌ PROBLEMA: R² > 15% → Estratégia tem exposição significativa aos fatores!")
        print("     Retorno pode ser contaminado por movimentos sistemáticos.")
    
    if alpha_ann > 0.05:
        print(f"\n  ✅ Alpha positivo ({alpha_ann*100:.2f}%/ano) → Estratégia gera valor!")
    elif alpha_ann > 0:
        print(f"\n  ⚠️  Alpha marginal ({alpha_ann*100:.2f}%/ano)")
    else:
        print(f"\n  ❌ Alpha negativo ({alpha_ann*100:.2f}%/ano) → Destruição de valor!")


# =============================================================================
# ANÁLISE COMPLETA (wrapper conveniente)
# =============================================================================

def analyze_strategy(algo_weights, betas, ret_net, Factor_PCA, pcs, 
                     plot=True, title_prefix=""):
    """
    Executa análise completa da estratégia:
    1. Exposição beta agregada
    2. Regressão PnL vs fatores
    
    Parâmetros
    ----------
    algo_weights : pd.DataFrame
        Pesos da carteira [date x stocks]
    betas : pd.DataFrame
        Betas de cada ação [date x stocks]
    ret_net : pd.Series
        Retornos líquidos da estratégia
    Factor_PCA : pd.DataFrame
        Retornos dos fatores PCA [date x pcs]
    pcs : list
        Lista de nomes dos fatores
    plot : bool
        Se True, gera gráficos
    title_prefix : str
        Prefixo para títulos dos gráficos
    
    Retorna
    -------
    results : dict
        Dicionário com todos os resultados
    """
    print("\n" + "🔍 "*35)
    print("ANÁLISE DE NEUTRALIDADE DA ESTRATÉGIA")
    print("🔍 "*35 + "\n")
    
    # Teste 1: Exposição Beta
    print("\n[1/2] Calculando exposição beta agregada...")
    expo_df = compute_beta_exposure(algo_weights, betas, pcs)
    expo_stats = beta_exposure_statistics(expo_df)
    
    if plot:
        fig1 = plot_beta_exposure(expo_df, title_prefix)
    
    # Teste 2: Regressão PnL
    print("\n[2/2] Regredindo PnL contra fatores...")
    regress_results = regress_pnl_on_factors(ret_net, Factor_PCA, pcs)
    pnl_regression_statistics(regress_results)
    
    if plot and regress_results is not None:
        fig2 = plot_pnl_regression(regress_results, title_prefix)
    
    # Consolidar resultados
    results = {
        'beta_exposure': expo_df,
        'beta_stats': expo_stats,
        'regression': regress_results,
    }
    
    print("\n" + "="*70)
    print("✅ Análise concluída!")
    print("="*70 + "\n")
    
    return results