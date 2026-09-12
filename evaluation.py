"""Monte Carlo batching, thesis comparison tables and time step checks."""
import math
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from simulation import simulate, generate_standard_normals, S0, RISK_FREE_RATE
from black_scholes import implied_volatility

@dataclass
class MomentAccumulator:
    count: int = 0
    total: np.ndarray | float | None = None
    total_square: np.ndarray | float | None = None

    def add(self, mean: np.ndarray | float, variance: np.ndarray | float, count: int) -> None:
        mean_array = np.asarray(mean, dtype=np.float64)
        variance_array = np.asarray(variance, dtype=np.float64)
        if self.total is None:
            self.total = np.zeros_like(mean_array)
            self.total_square = np.zeros_like(mean_array)
        self.total = self.total + count * mean_array
        self.total_square = (
            self.total_square
            + (count - 1) * variance_array
            + count * np.square(mean_array)
        )
        self.count += count

    def finish(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.count <= 1 or self.total is None or self.total_square is None:
            raise RuntimeError("Moment accumulator is empty")
        total = np.asarray(self.total, dtype=np.float64)
        total_square = np.asarray(self.total_square, dtype=np.float64)
        mean = total / self.count
        variance = np.maximum(
            (total_square - np.square(total) / self.count) / (self.count - 1), 0.0
        )
        return mean, variance, np.sqrt(variance / self.count)

def invert_prices(prices, frame):
    values = []
    for price, row in zip(prices, frame.itertuples(index=False)):
        iv = implied_volatility(float(price), S0, float(row.strike), float(row.maturity_years),
                                RISK_FREE_RATE, option_type=row.option_type, q=0.)
        values.append(iv if iv is not None and np.isfinite(iv) and iv > 0 else np.nan)
    return np.array(values)

def accumulators():
    return {i: {'raw': MomentAccumulator(), 'variance_reduced': MomentAccumulator()} for i in range(1, 8)}

def accumulate(moments, simulation, paths):
    for i in range(1, 8):
        for kind in ('raw', 'variance_reduced'):
            stats = simulation[i][kind]
            moments[i][kind].add(stats['mean'].detach().cpu().numpy(),
                                 stats['variance'].detach().cpu().numpy(), paths)

def price_table(moments, targets):
    tables = []
    for i in range(1, 8):
        frame = targets.loc[targets.maturity_index == i].sort_values('strike', kind='mergesort').copy()
        raw, raw_var, raw_se = moments[i]['raw'].finish()
        price, variance, se = moments[i]['variance_reduced'].finish()
        frame['raw_mc_price'] = raw
        frame['raw_mc_standard_error'] = raw_se
        frame['variance_reduced_price'] = price
        frame['variance_reduced_standard_error'] = se
        frame['model_implied_volatility'] = invert_prices(price, frame)
        frame['variance_reduction_factor'] = np.divide(raw_var, variance,
                                                       out=np.full_like(raw_var, np.nan), where=variance > 0)
        tables.append(frame)
    return pd.concat(tables, ignore_index=True)

@torch.no_grad()
def evaluate(model, targets, settings, *, unit_leverage=False):
    if settings['paths'] != settings['batch_paths']*len(settings['seeds']):
        raise ValueError('Evaluation batch plan does not match the total path count')
    moments = accumulators()
    samples = {i: [] for i in range(1, 8)}
    for seed in settings['seeds']:
        result, states = simulate(model, targets, settings['batch_paths'], 1, seed,
                                   unit_leverage=unit_leverage, collect_states=not unit_leverage)
        accumulate(moments, result, settings['batch_paths'])
        if states is not None:
            for i in range(1, 8):
                samples[i].extend(states[i])
        del result, states
    ranges = []
    if not unit_leverage:
        for i in range(1, 8):
            low, high = np.quantile(np.concatenate(samples[i]), [.001, .999])
            padding = .05*max(high-low, 1e-6)
            ranges.append(dict(slice=f'F{i}', lower=low-padding, upper=high+padding))
    return price_table(moments, targets), pd.DataFrame(ranges)

@torch.no_grad()
def neural_time_step_comparison(model, targets, settings):
    """One/two step paths share Brownian increments, as in the thesis."""
    if settings['paths'] != settings['batch_paths']*len(settings['seeds']):
        raise ValueError('Convergence batch plan does not match its total paths')
    coarse, fine = accumulators(), accumulators()
    for seed in settings['seeds']:
        normals = generate_standard_normals(360, settings['batch_paths'], seed=seed,
                                            dtype=model.alpha0.dtype, device=model.alpha0.device)
        coarse_normals = (normals[0::2]+normals[1::2])/math.sqrt(2.)
        result, _ = simulate(model, targets, settings['batch_paths'], 2, seed, external_normals=normals)
        accumulate(fine, result, settings['batch_paths'])
        del result
        result, _ = simulate(model, targets, settings['batch_paths'], 1, seed, external_normals=coarse_normals)
        accumulate(coarse, result, settings['batch_paths'])
        del result, normals, coarse_normals
    a, b = price_table(coarse, targets), price_table(fine, targets)
    output = a[['maturity_days', 'strike', 'option_type']].copy()
    output['one_spd_vr_price'] = a.variance_reduced_price
    output['two_spd_vr_price'] = b.variance_reduced_price
    output['one_spd_implied_volatility'] = a.model_implied_volatility
    output['two_spd_implied_volatility'] = b.model_implied_volatility
    output['signed_price_difference_2spd_minus_1spd'] = b.variance_reduced_price-a.variance_reduced_price
    output['signed_iv_difference_2spd_minus_1spd'] = b.model_implied_volatility-a.model_implied_volatility
    return output

MODELS = {'Neural LSV': ('variance_reduced_price', 'model_implied_volatility'),
          'Local Volatility': ('local_vol_price', 'local_vol_iv'),
          'Global Hagan SABR': ('hagan_sabr_price', 'hagan_sabr_iv'),
          'Simulated L=1 SABR': ('l1_price', 'l1_iv')}

def metric_rows(frame, scope):
    rows = []
    for model, (price, iv) in MODELS.items():
        p = (frame[price]-frame.heston_target_price).to_numpy()
        v = (frame[iv]-frame.heston_target_implied_volatility).to_numpy()
        finite = np.isfinite(v)
        v = v[finite]
        rows.append(dict(model=model, scope=scope, option_count=len(frame), invalid_iv_count=int((~finite).sum()),
                         price_rmse=np.sqrt(np.mean(p*p)), price_mae=np.mean(abs(p)), maximum_absolute_price_error=np.max(abs(p)),
                         iv_rmse=np.sqrt(np.mean(v*v)) if len(v) else np.nan,
                         iv_mae=np.mean(abs(v)) if len(v) else np.nan,
                         maximum_absolute_iv_error=np.max(abs(v)) if len(v) else np.nan))
    return rows

def write_tables(frame, output):
    pd.DataFrame(metric_rows(frame, 'overall')).to_csv(output/'comparison_overall.csv', index=False)
    maturity_rows = []
    for day, group in frame.groupby('maturity_days'):
        maturity_rows.extend(dict(**row, maturity_days=day) for row in metric_rows(group, f'{day}d'))
    pd.DataFrame(maturity_rows).to_csv(output/'comparison_by_maturity.csv', index=False)
    regions = metric_rows(frame[frame.maturity_days <= 30], 'market_supported')
    regions += metric_rows(frame[frame.maturity_days > 30], 'heston_extrapolation')
    pd.DataFrame(regions).to_csv(output/'comparison_by_region.csv', index=False)
    vrf = frame.groupby('maturity_days').variance_reduction_factor.agg(['median', 'mean'])
    vrf.to_csv(output/'variance_reduction.csv')

@torch.no_grad()
def sequential_results(model, targets, configuration, selection):
    """The independent per slice IV comparison reported in thesis Table 5.3."""
    from training import new_model
    rows = []
    for settings, plan in zip(configuration['training_slices'], configuration['slice_evaluation']):
        i = settings['maturity_index']
        stage = new_model(model.frozen_sabr_parameters, getattr(torch, settings['dtype']), model.alpha0.device)
        # Restore the appropriate precision of each historical stage.
        state = stage.state_dict()
        for k, value in model.state_dict().items():
            if k.startswith('leverage_networks.'):
                state[k] = value.to(dtype=stage.alpha0.dtype, device=stage.alpha0.device)
        stage.load_state_dict(state)
        frame = targets.loc[targets.maturity_index == i].sort_values('strike',kind='mergesort')
        errors = []
        invalid = []
        for unit in (True, False):
            total = MomentAccumulator()
            for seed in plan['seeds']:
                result, _ = simulate(stage, targets, plan['batch_paths'], 1, seed,
                                     maturity_index=i, unit_leverage=unit,
                                     numerical_mode=settings['numerical_mode'], hedge_mode=settings['hedge_volatility_mode'])
                stats = result[i]['variance_reduced']
                total.add(stats['mean'].cpu().numpy(),stats['variance'].cpu().numpy(),plan['batch_paths'])
            prices, _, _ = total.finish()
            iv = invert_prices(prices, frame)
            finite = np.isfinite(iv)
            invalid.append(int((~finite).sum()))
            errors.append(float(np.sqrt(np.mean((iv[finite]-frame.heston_target_implied_volatility.to_numpy()[finite])**2))) if finite.any() else np.nan)
        chosen = selection.loc[selection['slice'] == f'F{i}','best_iteration'].iloc[0]
        rows.append(dict(slice=f'F{i}', maturity_days=int(frame.maturity_days.iloc[0]),best_iteration=int(chosen),
                         baseline_iv_rmse=errors[0],slice_iv_rmse=errors[1],reduction_percent=100*(1-errors[1]/errors[0]),
                         baseline_invalid_iv_count=invalid[0],slice_invalid_iv_count=invalid[1]))
    return pd.DataFrame(rows)

