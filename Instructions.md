# Simplified thesis code

This folder runs independently of the original repository. It preserves the thesis's numerical models, training settings, seeds and validation-based checkpoint selection, with only the outputs used for thesis results and figures.

## Run locally (including PyCharm)

Open this folder as the project and select Python 3.12 or 3.13. In its terminal:

```sh
python -m pip install -r requirements.txt
python run.py reproduce --profile smoke --device cpu
```

The smoke profile uses small samples and two training iterations per maturity. It verifies execution; its estimates are not thesis-quality results. Use a new `--output` directory to repeat a run.

| Command | Purpose |
| --- | --- |
| `python run.py figures` | Generate the 15 numbered figures and comparison tables from the supplied accepted results, histories and trained network. |
| `python run.py reproduce --profile thesis --device cuda` | Reprice fixed Heston targets and evaluate the supplied trained model, SABR and Local Volatility using the thesis path counts. Generate figures, error tables, variance reduction and time-step comparisons. |
| `python run.py train --profile thesis --device cuda --output results/new_training` | Recalibrate global SABR, freeze its parameters, train F1–F7 sequentially and evaluate the resulting model. |
| `python run.py train --profile thesis --device cuda --output results/new_training --resume` | Resume completed maturity checkpoints. An interrupted maturity restarts from its beginning. |
| `python run.py surface --output results/new_surface` | Recompute Fourier-Dupire local variance on the supplied validated domain. This is a long CPU calculation. |
| `python run.py reproduce --profile thesis --device cuda --surface results/new_surface/local_vol_surface.csv` | Evaluate using a newly generated surface. |
| `python run.py heston-calibration` | Optional historical Heston calibration experiment. Saves distinct experimental parameters; never replaces the fixed thesis parameters. |

`--device auto` uses CUDA when available, otherwise CPU. Full neural training is intended for a Colab GPU. `Colab_run.ipynb` supports either the GitHub repository or an uploaded ZIP of this folder, and saves runs/checkpoints to Google Drive. Push this new folder to GitHub before choosing the notebook's GitHub option.

## Files

- `run.py`: command-line entry point and output writing.
- `heston_pricing.py`, `heston_calibration.py`, `targets.py`, `black_scholes.py`: semi-analytical Heston pricing, optional calibration, target construction and IV inversion.
- `fourier_dupire.py`, `local_volatility.py`: Fourier derivatives, Dupire variance, interpolation and pure Local Volatility simulation.
- `sabr_formula.py`, `sabr.py`: beta=1 Hagan formula and the accepted 48-start global calibration.
- `neural_model.py`, `control_variate.py`, `simulation.py`, `training.py`, `training_config.py`: seven leverage networks, detached Black-Scholes hedge, simulation and sequential training.
- `evaluation.py`, `figures.py`: the thesis comparison tables, independent slice evaluation, time-step comparisons, variance reduction and figures.
- `inputs/`: market data, fixed parameters, compact local-variance grid/domain and accepted trained network.
- `configs/`: full thesis and small execution-test settings, including independent slice-evaluation seeds.
- `reference/`, `training_history/`: numerical inputs for regenerating figures without retraining.

## What is produced

`figures` produces 15 PDFs plus the overall, maturity, region and variance-reduction tables. The simulation commands additionally produce `prices.csv`, the Heston market-fit table, independent sequential results, the neural and Local Volatility time-step comparisons, and the seven state ranges used by the leverage figure. IVs in CSV files are decimal volatilities; the figures display percentages or percentage-point errors.

Training also saves the calibrated SABR parameters, seven loss histories, selected-iteration summary, completed-maturity checkpoints and the final trained model. These files are needed for reuse, resuming training and the training-loss figure.

## Numerical method retained

The fixed ground truth is the thesis's corrected Heston calibration, with rho approximately -0.29776633, not the earlier rho=-0.20 calibration included only to initialize the optional historical experiment. Calls come from the characteristic function; OTM puts use put-call parity. The comparison grid is 29 strikes at 8, 15, 22, 30, 60, 90 and 180 days. The last three maturities are Heston extrapolations.

Local Volatility interpolates **variance**, then takes its square root. The first 0.25-day slice applies at earlier times; each slice's boundary values extend constantly outside its validated state interval. No negative local variance is clipped. The compact CSV retains all 54,107 numerical variance values. Surface rebuilding retains the Fourier contour/integrand method but omits the old exhaustive audit reports and exploratory domain search.

SABR is calibrated with the global Hagan IV objective and simulated independently with L=1. Neural leverage is the signed function L=1+F_i with the same network architecture. Only the current network is trained; SABR and earlier networks are frozen. The loss uses normalized inverse target Vega, and the hedge is detached. F1–F6 retain float32 legacy/running-volatility training; F7 and final model evaluation retain float64 robust log-state/alpha-only simulation. The best validation checkpoint is restored at every stage.

Independent per-slice evaluation uses the accepted stage precisions and seed families. Its L=1 comparison uses those same independent paths; baseline numbers may therefore differ slightly from the historical table's separately computed baseline. Figure 6.1 shows mean/median variance reduction, and Figure 6.2 uses empirical central state ranges, matching the thesis captions. The other figures retain the working package's simpler layout.

The new CPU smoke runs matched the working package's prices and trained checkpoint tensors. Full Local Volatility reruns at 200,000 and 100,000 paths matched accepted prices within 6e-14; nine Fourier spot checks agreed within 2e-15 in variance. Development comparisons are kept outside this folder. 

Generated results belong in `results/` (or an explicit output directory). Exclude generated outputs, caches and virtual environments if you submit this folder again after running it.