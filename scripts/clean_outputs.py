"""
clean_outputs.py

Delete all generated outputs under results/, keeping README.md if present.
Usage:
    python scripts/clean_outputs.py [--dry-run]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import sys

# Allow running from repo root or any working directory
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import RESULTS_DIR


def _delete_path(path: Path, *, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] Would delete: {path}")
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean generated outputs under results/.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be deleted.")
    args = parser.parse_args()

    if not RESULTS_DIR.exists():
        print(f"[clean] No results dir found at {RESULTS_DIR}")
        return

    for p in RESULTS_DIR.iterdir():
        if p.name == "README.md":
            continue
        _delete_path(p, dry_run=args.dry_run)

    print("[clean] Done.")


if __name__ == "__main__":
    main()
