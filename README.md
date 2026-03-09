# Spillover-Aware Portfolio Optimization

This repository implements a spillover-aware portfolio framework using VAR + generalized FEVD
and evaluates allocation and backtest performance with stress-window analysis and empirical
sanity checks.

## Quick Start

1. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

3. Run the full workflow (from cache/build to results)
```bash
python scripts/run_all.py
```

By default, `run_all.py` also runs plots, network metrics, crisis evaluation,
and empirical TCI checks. You can skip any step with `--no-*` flags:
```bash
python scripts/run_all.py --no-plots --no-network --no-crisis --no-empirical
```

If spillover caches are missing, `run_experiments.py` will automatically
recompute them before continuing.

## Reproducibility

- All experiment knobs live in `config.py`.
- Outputs are centralized under `results/`:
  - `results/cache/` for spillover `.npz` caches
  - `results/reports/` for tables and weights
  - `results/plots/` for figures
- Each run saves a manifest and config snapshot to `results/reports/`.
- `run_experiments.py` also exports:
  - `results/reports/data_provenance_<RUN_TAG>.json`
  - `results/reports/data_provenance_<RUN_TAG>.csv`
  - `results/reports/return_descriptive_stats_<RUN_TAG>.csv`
  - `results/reports/return_cross_section_summary_<RUN_TAG>.csv`
- Raw price cache files are stored in `cache/` (outside `results/`).

## Typical Workflow

```bash
python returns.py                # build returns (uses cached prices if present)
python scripts/check_price_panel.py  # optional smoke test for inspecting the cleaned price panel
python rolling_spillover.py      # VAR + FEVD rolling spillovers
python run_experiments.py        # backtests + reports (equal, mean-var, min-var, spillover)
python run_robustness.py         # one-factor-at-a-time robustness grid + sensitivity tables
python report_plots.py           # plots (optional)
python network_metrics.py        # network metrics (optional)
python crisis_eval.py            # stress evaluation (optional)
python scripts/empirical_checks.py  # TCI visual sanity check
```

Robustness shortcuts:
```bash
python run_robustness.py --axis all                    # axis robustness (default section)
python run_robustness.py --axis window
python run_robustness.py --section lambda_sensitivity
python run_robustness.py --section all --axis all      # run both axis + lambda sections
```

## Notes

- The data provider is LSEG by default. If you do not have access, ensure you load
  cached price data in `cache/` or adjust `data_loader.py`.
- A reproducible run uses fixed `START_DATE`, `END_DATE`, and `RUN_TAG` in `config.py`.
