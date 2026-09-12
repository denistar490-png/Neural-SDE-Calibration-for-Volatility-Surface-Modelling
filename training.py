"""Sequential Adam training"""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
from neural_model import NeuralLSVModel
from simulation import simulate
from control_variate import fixed_vega_weighted_price_loss
from training_config import AcceptedSliceConfig

def new_model(parameters, dtype, device):
    model = NeuralLSVModel(**parameters, dtype=dtype, device=device)
    for i, network in enumerate(model.leverage_networks, 1):
        torch.manual_seed(20260830+i)
        if torch.device(device).type == 'cuda':
            torch.cuda.manual_seed_all(20260830+i)
        network.reset_parameters()
    model.freeze_all()
    return model

def save_model(path, model, **selection):
    payload = dict(model_state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                   frozen_sabr_parameters=model.frozen_sabr_parameters, **selection)
    temporary = Path(path).with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)

def load_model(path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    model = new_model(payload['frozen_sabr_parameters'], torch.float64, device)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    model.freeze_all()
    return model

def price_loss(prices, frame):
    target = torch.as_tensor(frame.heston_target_price.to_numpy(np.float64), dtype=prices.dtype, device=prices.device)
    weights = torch.as_tensor(frame.normalized_vega_weight.to_numpy(np.float64), dtype=prices.dtype, device=prices.device)
    return fixed_vega_weighted_price_loss(prices, target, weights)

def train_slice(model, targets, settings):
    index = settings.maturity_index
    model.set_trainable_slice(index)
    active = list(model.leverage_networks[index-1].parameters())
    optimizer = torch.optim.Adam(active, lr=settings.learning_rate)
    best_loss, best_iteration, best_state = math.inf, -1, None
    history = []
    for iteration in range(settings.maximum_iterations):
        optimizer.zero_grad(set_to_none=True)
        result, _ = simulate(model, targets, settings.paths_for_iteration(iteration),
                             settings.steps_per_day, settings.training_seed_base+iteration,
                             maturity_index=index, numerical_mode=settings.numerical_mode,
                             hedge_mode=settings.hedge_volatility_mode)
        current = result[index]
        loss = price_loss(current['variance_reduced']['mean'], current['frame'])
        if not torch.isfinite(loss) or not loss.requires_grad:
            raise FloatingPointError('Training loss is invalid or detached')
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in active):
            raise FloatingPointError('Invalid neural gradient')
        optimizer.step()
        history.append(dict(iteration=iteration, weighted_price_loss=float(loss.detach().cpu())))
        if (iteration+1) % settings.validation_interval == 0 or iteration+1 == settings.maximum_iterations:
            model.freeze_all()
            with torch.no_grad():
                price_sum = torch.zeros(29, dtype=model.alpha0.dtype, device=model.alpha0.device)
                for seed in settings.validation_seeds:
                    sample, _ = simulate(model, targets, settings.validation_paths_per_seed,
                                         settings.steps_per_day, seed, maturity_index=index,
                                         numerical_mode=settings.numerical_mode, hedge_mode=settings.hedge_volatility_mode)
                    price_sum += settings.validation_paths_per_seed*sample[index]['variance_reduced']['mean']
                n = settings.validation_paths_per_seed*len(settings.validation_seeds)
                value = float(price_loss(price_sum/n, sample[index]['frame']).cpu())
            if not math.isfinite(value):
                raise FloatingPointError('Invalid validation loss')
            if value < best_loss:
                best_loss, best_iteration = value, iteration
                best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            model.set_trainable_slice(index)
    if best_state is None:
        raise RuntimeError('No validation checkpoint was selected')
    model.load_state_dict(best_state)
    model.freeze_all()
    return pd.DataFrame(history), dict(slice=f'F{index}', best_iteration=best_iteration,
                                      best_validation_weighted_price_loss=best_loss)

def train_all(out, targets, parameters, configuration, device, resume=False):
    """Seven completed-slice checkpoints """
    checkpoint_dir = out/'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    previous = None
    summary = []
    for values in configuration['training_slices']:
        settings = AcceptedSliceConfig(**values)
        i = settings.maturity_index
        model = new_model(parameters, getattr(torch, settings.dtype), device)
        if previous is not None:
            state = model.state_dict()
            for k, v in previous.items():
                if k.startswith('leverage_networks.'):
                    state[k] = v.to(dtype=model.alpha0.dtype, device=device)
            model.load_state_dict(state)
        checkpoint = checkpoint_dir/f'F{i}.pt'
        if resume and checkpoint.exists():
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if saved['settings'] != values or saved['frozen_sabr_parameters'] != parameters:
                raise ValueError('Checkpoint settings or SABR parameters differ')
            model.load_state_dict(saved['model_state_dict'])
            selection = saved['selection']
        else:
            print(f'Training F{i}/7 ({settings.maximum_iterations} iterations)...', flush=True)
            history, selection = train_slice(model, targets, settings)
            history.to_csv(out/f'F{i}_training.csv', index=False)
            save_model(checkpoint, model, settings=values, selection=selection)
        previous = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        summary.append(selection)
    pd.DataFrame(summary).to_csv(out/'training_summary.csv', index=False)
    save_model(out/'trained_model.pt', model)
    return model
