"""Run from PyCharm or a terminal"""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.matplotlib-cache'))

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def main():
    parser = argparse.ArgumentParser(description='Simplified thesis reproduction and training')
    parser.add_argument('command', choices=['figures', 'reproduce', 'train', 'surface', 'heston-calibration'])
    parser.add_argument('--profile', choices=['smoke', 'thesis'], default='thesis')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--output', type=Path, help='New result directory; default: results/<command>_<profile>')
    parser.add_argument('--surface', type=Path, help='Use a regenerated Local Volatility surface')
    parser.add_argument('--resume', action='store_true', help='Resume completed training maturities in --output')
    args = parser.parse_args()
    if args.resume and (args.command != 'train' or args.output is None):
        parser.error('--resume requires train and --output')
    import numpy as np
    import pandas as pd
    import torch
    from training import load_model, train_all
    from targets import price_grid, prepare_targets
    from sabr import calibrate, build_fit_results
    from local_volatility import LocalVolSurface, price_options, rebuild_surface
    from evaluation import evaluate, neural_time_step_comparison, write_tables, sequential_results
    from figures import create_figures
    device = torch.device('cuda:0' if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu or a Colab GPU runtime')
    if device.type == 'cpu':
        torch.set_num_threads(2)
    config = read_json(ROOT/'configs'/f'{args.profile}.json')
    out = (args.output or ROOT/'results'/f'{args.command}_{args.profile}').resolve()
    if out == ROOT or out.is_relative_to(ROOT) and not out.is_relative_to(ROOT/'results'):
        raise ValueError('Choose an output under results/ or outside this package')
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError('This result directory is not empty. Choose a new --output directory.')
    out.mkdir(parents=True, exist_ok=True)
    inputs = ROOT/'inputs'
    print(f'{args.command}: {args.profile} profile on {device}', flush=True)
    heston = read_json(inputs/'heston_params_corrected.json')
    heston_vector = [heston[k] for k in ('v0', 'kappa', 'theta', 'sigma', 'rho')]
    if args.command == 'heston-calibration':
        from heston_calibration import load_calibration_data, select_representative_subset, run_least_squares
        _, subset, _ = select_representative_subset(load_calibration_data(inputs/'options_clean.csv'))
        original = read_json(inputs/'heston_params_original.json')
        settings = dict(method='trf', loss='linear', x_scale='jac', max_nfev=60, ftol=1e-6, xtol=1e-6, gtol=1e-6)
        result, _ = run_least_squares(subset, np.array([original[k] for k in ('v0','kappa','theta','sigma','rho')]), settings)
        if not result.success:
            raise RuntimeError('Heston calibration did not converge; parameters were not saved')
        (out/'experimental_heston_parameters.json').write_text(json.dumps(dict(zip(('v0','kappa','theta','sigma','rho'),result.x)),indent=2)+'\n')
    elif args.command == 'surface':
        rebuild_surface(heston_vector, inputs/'local_vol_domain.csv', out/'local_vol_surface.csv')
    elif args.command == 'figures':
        frame = pd.read_csv(ROOT/'reference/neural_lsv_final_results.csv').rename(columns={'l1_sabr_sde_price':'l1_price', 'l1_sabr_sde_iv':'l1_iv'})
        model = load_model(inputs/'trained_model.pt', device)
        write_tables(frame, out)
        create_figures(frame, out/'figures', model=model, history_dir=ROOT/'training_history')
    else:
        print('Pricing Heston targets...', flush=True)
        grid = price_grid(heston_vector)
        from heston_calibration import load_calibration_data, corrected_model_prices_and_ivs
        market = load_calibration_data(inputs/'options_clean.csv')
        _, market_model_iv = corrected_model_prices_and_ivs(heston_vector, market)
        market['iv_error'] = market_model_iv-market.IV
        market_rows = []
        for label, group in [('overall',market)]+list(market.groupby('T')):
            error = group.iv_error.to_numpy()
            valid = np.isfinite(error)
            market_rows.append(dict(maturity_days=label if label == 'overall' else int(round(label*365)),
                                    iv_rmse=float(np.sqrt(np.mean(error[valid]**2))),iv_mae=float(np.mean(abs(error[valid]))),
                                    invalid_iv_count=int((~valid).sum())))
        pd.DataFrame(market_rows).to_csv(out/'heston_market_fit.csv',index=False)
        parameters = calibrate(grid) if args.command == 'train' else read_json(inputs/'sabr_beta1_global_params.json')
        vector = np.array([parameters[k] for k in ('alpha0','nu','rho')])
        sabr = build_fit_results(grid, vector)
        targets = prepare_targets(grid, sabr)
        if args.command == 'train':
            (out/'sabr_parameters.json').write_text(json.dumps(parameters, indent=2)+'\n')
            model = train_all(out, targets, parameters, config, device, args.resume)
        else:
            model = load_model(inputs/'trained_model.pt', device)
        print('Evaluating Neural LSV and the L=1 SABR baseline...', flush=True)
        neural, ranges = evaluate(model, targets, config['evaluation'])
        baseline, _ = evaluate(model, targets, config['evaluation'], unit_leverage=True)
        surface = LocalVolSurface(args.surface or inputs/'local_vol_surface.csv', inputs/'local_vol_domain.csv')
        lv = [price_options(surface, grid, settings) for settings in config['local_volatility_runs']]
        keys = ['maturity_days', 'strike', 'option_type']
        frame = neural.merge(lv[0][keys+['mc_price','mc_implied_volatility']].rename(columns={'mc_price':'local_vol_price','mc_implied_volatility':'local_vol_iv'}),on=keys,validate='one_to_one')
        frame = frame.merge(sabr[keys+['sabr_black_scholes_price','sabr_implied_volatility']].rename(columns={'sabr_black_scholes_price':'hagan_sabr_price','sabr_implied_volatility':'hagan_sabr_iv'}),on=keys,validate='one_to_one')
        frame = frame.merge(baseline[keys+['variance_reduced_price','model_implied_volatility']].rename(columns={'variance_reduced_price':'l1_price','model_implied_volatility':'l1_iv'}),on=keys,validate='one_to_one')
        frame = frame.drop(columns=['black_scholes_target_vega','unnormalized_vega_weight','normalized_vega_weight'])
        frame.to_csv(out/'prices.csv', index=False)
        write_tables(frame, out)
        print('Computing the thesis time-step comparisons...', flush=True)
        neural_time_step_comparison(model, targets, config['convergence']).to_csv(out/'neural_time_steps.csv', index=False)
        comparison = lv[0].merge(lv[1],on=keys,suffixes=('_4spd','_8spd'),validate='one_to_one')
        comparison['price_difference_8spd_minus_4spd'] = comparison.mc_price_8spd-comparison.mc_price_4spd
        comparison['iv_difference_8spd_minus_4spd'] = comparison.mc_implied_volatility_8spd-comparison.mc_implied_volatility_4spd
        comparison.to_csv(out/'local_vol_time_steps.csv', index=False)
        ranges.to_csv(out/'leverage_ranges.csv', index=False)
        selected = pd.read_csv(out/'training_summary.csv') if args.command == 'train' else pd.read_csv(ROOT/'reference/sequential_training_summary.csv')
        sequential_results(model, targets, config, selected).to_csv(out/'sequential_results.csv',index=False)
        create_figures(frame, out/'figures', model=model, history_dir=out if args.command == 'train' else ROOT/'training_history',
                       selection=selected, ranges=ranges)
    print(f'Saved results: {out}', flush=True)

if __name__ == '__main__':
    main()
