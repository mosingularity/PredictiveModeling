import pandas as pd
import pytest

from models.tariff_history import assemble_entity_series


def _make_data():
    lpu_rows = [
        {"PodID": "POD_X", "ReportingMonth": f"2022-{m:02d}-01", "TariffType": "LPU"}
        for m in range(1, 13)
    ]
    spu_rows = [
        {"PodID": "POD_X", "ReportingMonth": f"2023-{m:02d}-01", "TariffType": "SPU"}
        for m in range(1, 13)
    ]
    stable_rows = (
        [{"PodID": "POD_Y", "ReportingMonth": f"2022-{m:02d}-01", "TariffType": "LPU"} for m in range(1, 13)]
        + [{"PodID": "POD_Y", "ReportingMonth": f"2023-{m:02d}-01", "TariffType": "LPU"} for m in range(1, 13)]
    )
    df = pd.DataFrame(lpu_rows + spu_rows + stable_rows)
    df["ReportingMonth"] = pd.to_datetime(df["ReportingMonth"])
    return df


DATA = _make_data()


def test_full_history_for_tariff_change_entity():
    result = assemble_entity_series("POD_X", DATA)
    assert len(result) == 24, f"expected 24 rows, got {len(result)}"
    assert set(result["TariffType"].unique()) == {"LPU", "SPU"}


def test_series_sorted_by_reporting_month():
    result = assemble_entity_series("POD_X", DATA)
    months = result["ReportingMonth"].tolist()
    assert months == sorted(months)


def test_no_change_entity_correct_count():
    result = assemble_entity_series("POD_Y", DATA)
    assert len(result) == 24, f"expected 24 rows, got {len(result)}"
    assert set(result["TariffType"].unique()) == {"LPU"}
