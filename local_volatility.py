"""Ragged Dupire interpolation, pure Local Volatility simulation, and pricing."""
import math
from pathlib import Path
import numpy as np
import pandas as pd
from targets import implied_volatility_from_price
from fourier_dupire import heston_local_variance_fourier

S0 = np.float64(7165.08)
RATE = np.float64(.0359)
DAYS = [8, 15, 22, 30, 60, 90, 180]

class LocalVolSurface:
    def __init__(self, path, domain_path):
        data = pd.read_csv(path, float_precision='round_trip')
        domain = pd.read_csv(domain_path, float_precision='round_trip')
        days = np.r_[np.arange(1, 33)*.25, np.arange(9, 181)]
        if len(data) != 54107 or not np.array_equal(domain.maturity_days, days):
            raise ValueError('Incomplete Local Volatility grid')
        self.times = days/365.0
        self.states, self.variances = [], []
        for boundary in domain.itertuples(index=False):
            group = data.loc[data.maturity_days == boundary.maturity_days]
            x = group.x.to_numpy(np.float64)
            variance = group.local_variance.to_numpy(np.float64)
            if len(x) < 2 or not np.allclose(np.diff(x), .005, atol=1e-12, rtol=0):
                raise ValueError('Invalid Local Volatility state grid')
            if abs(x[0]-boundary.lower_x) > 1e-12 or abs(x[-1]-boundary.upper_x) > 1e-12:
                raise ValueError('Surface and domain endpoints differ')
            if not np.isfinite(variance).all() or np.any(variance <= 0):
                raise ValueError('Nonpositive or nonfinite Dupire variance')
            self.states.append(x)
            self.variances.append(variance)

    def volatility(self, time, spots):
        if not np.isfinite(time) or time < 0 or time > self.times[-1]:
            raise ValueError('Time outside the 180-day horizon')
        if not np.isfinite(spots).all() or np.any(spots <= 0):
            raise FloatingPointError('Invalid Local Volatility spot')
        x = np.log(spots/S0)
        upper = int(np.searchsorted(self.times, time, side='left'))
        lower = upper if upper == 0 or time == self.times[upper] else upper-1
        # Constant spatial extension at each slice's endpoints. The first
        # 0.25-day slice is also used at earlier times, no variance clipping.
        low = np.interp(x, self.states[lower], self.variances[lower],
                        left=self.variances[lower][0], right=self.variances[lower][-1])
        if lower == upper:
            variance = low
        else:
            high = np.interp(x, self.states[upper], self.variances[upper],
                             left=self.variances[upper][0], right=self.variances[upper][-1])
            weight = (time-self.times[lower])/(self.times[upper]-self.times[lower])
            variance = low+weight*(high-low)
        if not np.isfinite(variance).all() or np.any(variance <= 0):
            raise FloatingPointError('Invalid interpolated variance')
        return np.sqrt(variance, dtype=np.float64)

def simulate_paths(surface, paths, steps_per_day, seed):
    if paths < 4 or paths % 2 or steps_per_day not in (4, 8):
        raise ValueError('Use an even path count and four or eight steps/day')
    dt = np.float64(1.0)/np.float64(365*steps_per_day)
    sqrt_dt = np.sqrt(dt, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    logs = np.full(paths, np.log(S0), dtype=np.float64)
    spots = np.full(paths, S0, dtype=np.float64)
    terminals = {}
    for step in range(180*steps_per_day):
        sigma = surface.volatility(np.float64(step)*dt, spots)
        z = rng.standard_normal(paths//2, dtype=np.float64)
        z = np.concatenate((z, -z)).astype(np.float64, copy=False)
        logs = (logs+(RATE-np.float64(.5)*sigma**2)*dt+sigma*sqrt_dt*z).astype(np.float64, copy=False)
        spots = np.exp(logs, dtype=np.float64)
        if not np.isfinite(spots).all() or np.any(spots <= 0):
            raise FloatingPointError('Invalid simulated Local Volatility state')
        if step+1 in [d*steps_per_day for d in DAYS]:
            terminals[(step+1)//steps_per_day] = spots.copy()
    return terminals

def price_options(surface, grid, settings):
    terminals = simulate_paths(surface, settings['paths'], settings['steps_per_day'], settings['seed'])
    rows = []
    for row in grid.itertuples(index=False):
        terminal = terminals[int(row.maturity_days)]
        payoff = np.maximum(terminal-row.strike, 0.) if row.option_type == 'call' else np.maximum(row.strike-terminal, 0.)
        discounted = np.exp(-RATE*row.maturity_years)*payoff
        half = len(discounted)//2
        pairs = np.float64(.5)*(discounted[:half]+discounted[half:])
        price = float(np.mean(pairs, dtype=np.float64))
        se = float(np.std(pairs, ddof=1, dtype=np.float64)/np.sqrt(half))
        iv = implied_volatility_from_price(price, float(S0), row.strike, row.maturity_years, row.option_type)
        rows.append(dict(maturity_days=row.maturity_days, strike=row.strike,
                         option_type=row.option_type, mc_price=price, mc_standard_error=se,
                         mc_implied_volatility=iv if iv is not None and np.isfinite(iv) and iv > 0 else np.nan))
    return pd.DataFrame(rows)

def rebuild_surface(parameters, domain_path, output):
    
    domain = pd.read_csv(domain_path, float_precision='round_trip')
    rows = []
    for boundary in domain.itertuples(index=False):
        day = boundary.maturity_days
        for tick in range(round(boundary.lower_x/.005), round(boundary.upper_x/.005)+1):
            x = tick*.005
            variance = heston_local_variance_fourier(float(S0), float(S0*math.exp(x)), day/365., float(RATE), parameters)
            rows.append(dict(maturity_days=day, x=x, local_variance=variance))
        print(f'Local Volatility surface: day {day:g}/180', flush=True)
    pd.DataFrame(rows).to_csv(output, index=False, float_format='%.17g')
    LocalVolSurface(output, domain_path)
