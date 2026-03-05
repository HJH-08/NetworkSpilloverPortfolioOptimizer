"""
run_all.py

Convenience runner for the full pipeline.

Usage:
    python scripts/run_all.py
    python scripts/run_all.py --force-recompute --with-plots --with-network --with-crisis --with-empirical
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root on sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from returns import get_returns_bundle
from rolling_spillover import run_rolling_spillovers
import run_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Run spillover pipeline end-to-end.")
    parser.add_argument("--force-recompute", action="store_true", help="Recompute spillovers even if cache exists.")
    parser.add_argument("--no-plots", action="store_true", help="Skip plots.")
    parser.add_argument("--no-network", action="store_true", help="Skip network metrics.")
    parser.add_argument("--no-crisis", action="store_true", help="Skip crisis evaluation.")
    parser.add_argument("--no-empirical", action="store_true", help="Skip empirical TCI plot.")
    args = parser.parse_args()

    print("[run_all] Step 1/3: build returns (may use cache)")
    _ = get_returns_bundle(use_cache=True)

    print("[run_all] Step 2/3: rolling spillovers")
    run_rolling_spillovers(use_cache=True, force_recompute=args.force_recompute)

    print("[run_all] Step 3/3: experiments + backtests")
    run_experiments.main()

    if not args.no_plots:
        import report_plots
        report_plots.main()
        print("[run_all] plots done")

    if not args.no_network:
        import network_metrics
        network_metrics.main()
        print("[run_all] network metrics done")

    if not args.no_crisis:
        import crisis_eval
        crisis_eval.main()
        print("[run_all] crisis eval done")

    if not args.no_empirical:
        import scripts.empirical_checks as empirical_checks
        empirical_checks.main()
        print("[run_all] empirical checks done")

    print("[run_all] done")


if __name__ == "__main__":
    main()
