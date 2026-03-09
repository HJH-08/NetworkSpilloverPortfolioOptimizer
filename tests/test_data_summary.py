import json

from data_summary import export_data_provenance_reports


def test_export_data_provenance_reports_writes_json_and_csv(tmp_path):
    payload = {
        "provider_used": "lseg",
        "assets_retained": ["XLF", "XLK"],
        "assets_dropped": [{"asset": "BAD", "reason": "coverage_below_threshold"}],
    }

    json_path, csv_path = export_data_provenance_reports(
        payload,
        run_tag="unit_test",
        reports_dir=tmp_path,
    )

    assert json_path.exists()
    assert csv_path.exists()

    with json_path.open() as f:
        out = json.load(f)
    assert out["provider_used"] == "lseg"
