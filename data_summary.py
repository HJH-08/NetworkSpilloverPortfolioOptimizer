"""
data_summary.py

Export reproducibility-focused data provenance artefacts for reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Tuple

import pandas as pd

from config import REPORTS_DIR, RUN_TAG


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return value


def _csv_string(value: Any) -> str:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(_json_safe(value), ensure_ascii=True)
    if value is None:
        return ""
    return str(value)


def export_data_provenance_reports(
    provenance: Mapping[str, Any],
    *,
    run_tag: str = RUN_TAG,
    reports_dir: Path = REPORTS_DIR,
) -> Tuple[Path, Path]:
    """
    Write data provenance as:
      - results/reports/data_provenance_<RUN_TAG>.json
      - results/reports/data_provenance_<RUN_TAG>.csv  (key-value rows)
    """
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    safe_payload = {str(k): _json_safe(v) for k, v in provenance.items()}

    json_path = reports_dir / f"data_provenance_{run_tag}.json"
    with json_path.open("w") as f:
        json.dump(safe_payload, f, indent=2, sort_keys=True)

    rows = [{"field": k, "value": _csv_string(v)} for k, v in safe_payload.items()]
    csv_path = reports_dir / f"data_provenance_{run_tag}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    return json_path, csv_path
