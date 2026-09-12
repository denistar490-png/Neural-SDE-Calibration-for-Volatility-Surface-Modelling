"""Generate the 15 numbered thesis figures from explicit tables and a model."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent
DAYS = [8, 15, 22, 30, 60, 90, 180]
MODELS = {'Local Volatility': 'local_vol_iv', 'Hagan SABR': 'hagan_sabr_iv',
          'L=1 SABR simulation': 'l1_iv', 'Neural LSV': 'model_implied_volatility'}

def create_figures(data, output, *, model, history_dir, selection=None, ranges=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = data.sort_values(['maturity_days', 'forward_log_moneyness']).copy()
    plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .2})
    files = []

    def save(fig, name):
        fig.tight_layout()
        for suffix in ('pdf',):
            path = output / f'{name}.{suffix}'
            fig.savefig(path, dpi=160, bbox_inches='tight')
            files.append(path.name)
        plt.close(fig)

    def panels(days, columns, name):
        fig, axes = plt.subplots((len(days)+1)//2, 2, figsize=(12, 3*((len(days)+1)//2)), squeeze=False)
        for ax, day in zip(axes.flat, days):
            group = data[data.maturity_days == day]
            x = group.forward_log_moneyness
            ax.plot(x, 100*group.heston_target_implied_volatility, 'k-', label='Heston')
            for label, column in columns.items():
                ax.plot(x, 100*group[column], label=label)
            region = 'market-supported' if day <= 30 else 'Heston extrapolation'
            ax.set(title=f'{day} days ({region})', xlabel='Forward log-moneyness', ylabel='Implied volatility (%)')
        for ax in list(axes.flat)[len(days):]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=8)
        save(fig, name)

    def heatmap(values, name, label, signed=False):
        
        
        table = data.assign(value=values, plot_k=data.forward_log_moneyness.round(2)).pivot(index='maturity_days', columns='plot_k', values='value')
        if table.shape != (7, 29):
            raise ValueError('The thesis heatmaps require seven maturities and 29 common strikes')
        fig, ax = plt.subplots(figsize=(10, 4))
        limit = np.nanmax(abs(table.to_numpy())) if signed else None
        im = ax.imshow(table, aspect='auto', origin='lower', cmap='RdBu_r' if signed else 'viridis',
                       vmin=-limit if signed else None, vmax=limit)
        ax.set_yticks(range(7), [f'{d}d'+(' (extrap.)' if d > 30 else '') for d in DAYS])
        ticks = np.arange(0, table.shape[1], 4)
        ax.set_xticks(ticks, [f'{table.columns[i]:.2f}' for i in ticks])
        ax.set(xlabel='Forward log-moneyness', ylabel='Maturity')
        fig.colorbar(im, ax=ax, label=label)
        save(fig, name)

    domain = pd.read_csv(ROOT/'inputs/local_vol_domain.csv')
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(domain.maturity_days, domain.lower_x, domain.upper_x, alpha=.25)
    ax.plot(domain.maturity_days, domain.lower_x, label='Lower boundary')
    ax.plot(domain.maturity_days, domain.upper_x, label='Upper boundary')
    ax.set(xlabel='Maturity (days)', ylabel='Log spot / initial spot', title='Validated Fourier-Dupire interpolation domain')
    ax.legend()
    save(fig, '4_1_local_vol_ragged_domain_explanation')
    panels(DAYS, {}, '5_1_heston_smiles_by_maturity')
    heatmap(100*data.heston_target_implied_volatility, '5_2_heston_iv_surface_heatmap', 'Heston IV (%)')
    for column, name in [('local_vol_iv', '5_3_lv_heston_iv_scatter'), ('hagan_sabr_iv', '5_5_sabr_global_iv_scatter')]:
        fig, ax = plt.subplots(figsize=(6, 5))
        for extrap, label in [(False, 'Market-supported'), (True, 'Heston extrapolation')]:
            group = data[(data.maturity_days > 30) == extrap]
            ax.scatter(100*group.heston_target_implied_volatility, 100*group[column], s=12, label=label)
        lo, hi = 100*data.heston_target_implied_volatility.min(), 100*data.heston_target_implied_volatility.max()
        ax.plot([lo, hi], [lo, hi], 'k--')
        ax.set(xlabel='Heston IV (%)', ylabel='Model IV (%)'); ax.legend()
        save(fig, name)
    heatmap(100*(data.local_vol_iv-data.heston_target_implied_volatility), '5_4_lv_signed_iv_error_heatmap', 'Model minus Heston IV (percentage points)', True)

    if selection is None:
        selection = pd.read_csv(ROOT/'reference/sequential_training_summary.csv')
    fig, axes = plt.subplots(4, 2, figsize=(12, 11))
    for i, ax in enumerate(axes.flat[:7], 1):
        history = pd.read_csv(Path(history_dir)/f'F{i}_training.csv')
        loss = 'weighted_price_loss' if 'weighted_price_loss' in history else 'training_weighted_loss'
        ax.semilogy(history.iteration, history[loss], alpha=.75)
        chosen = selection.loc[selection['slice'] == f'F{i}', 'best_iteration']
        if len(chosen): ax.axvline(chosen.iloc[0], color='black', linestyle='--', label='Selected checkpoint')
        ax.set(title=f'F{i}: {DAYS[i-1]} days', xlabel='Iteration (zero-based)', ylabel='Weighted price loss')
    axes.flat[0].legend(); axes.flat[-1].set_visible(False)
    save(fig, '5_6_neural_lsv_training_loss_by_slice')
    panels(DAYS, {'Neural LSV': 'model_implied_volatility'}, '5_7_neural_lsv_fit_smiles_all_maturities')
    heatmap(100*(data.model_implied_volatility-data.heston_target_implied_volatility), '5_8_neural_lsv_signed_iv_error_heatmap', 'Model minus Heston IV (percentage points)', True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(list(MODELS), [100*np.sqrt(np.nanmean((data[c]-data.heston_target_implied_volatility)**2)) for c in MODELS.values()])
    ax.set(ylabel='IV RMSE (percentage points)')
    save(fig, '5_9_model_iv_rmse_comparison')
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, column in MODELS.items():
        values = [100*np.sqrt(np.nanmean((g[column]-g.heston_target_implied_volatility)**2)) for _, g in data.groupby('maturity_days')]
        ax.plot(DAYS, values, 'o-', label=label)
    ax.axvspan(30, 180, color='grey', alpha=.1, label='Heston extrapolation (>30d)')
    ax.set(xlabel='Maturity (days)', ylabel='IV RMSE (percentage points)'); ax.legend(fontsize=8)
    save(fig, '5_10_model_iv_rmse_by_maturity')
    panels(DAYS[:4], MODELS, '5_11_neural_lsv_model_comparison_smiles_market_supported')
    panels(DAYS[4:], MODELS, '5_12_neural_lsv_model_comparison_smiles_extrapolation')
    fig, ax = plt.subplots(figsize=(9, 4))
    factors = data.groupby('maturity_days').variance_reduction_factor.agg(['median','mean'])
    ax.plot(factors.index, factors['median'], 'o-', label='Median')
    ax.plot(factors.index, factors['mean'], 's-', label='Mean')
    ax.axhline(1, color='black', linestyle='--', linewidth=.8)
    ax.set(xlabel='Maturity (days)', ylabel='Raw variance / reduced variance'); ax.legend()
    save(fig, '6_1_neural_lsv_variance_reduction_by_maturity')
    if ranges is None:
        ranges = pd.read_csv(ROOT/'reference/leverage_ranges.csv')
    fig, axes = plt.subplots(4, 2, figsize=(12, 11))
    buffer = model.alpha0
    with torch.no_grad():
        for i, network in enumerate(model.leverage_networks, 1):
            row = ranges.loc[ranges['slice'] == f'F{i}'].iloc[0]
            x = torch.linspace(row.lower, row.upper, 801, dtype=buffer.dtype, device=buffer.device)
            ax = axes.flat[i-1]
            ax.plot(x.cpu().numpy(), (1+network(x)).cpu().numpy())
            ax.axhline(1, color='black', linestyle='--', linewidth=.7)
            ax.set(xlabel='Discounted log state X', ylabel='Leverage L = 1 + F', title=f'F{i}: empirical central state range')
    axes.flat[-1].set_visible(False)
    save(fig, '6_2_neural_lsv_leverage_functions')
    print(f'Generated {len(files)} figures.', flush=True)
