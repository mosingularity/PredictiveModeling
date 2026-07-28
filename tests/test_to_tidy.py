"""
Tests for the tidy forecast contract and adapter (results_analysis.tidy).

Run from the project root:
    pytest tests/test_to_tidy.py -v
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.performance import EntityPerformanceData, ForecastResults
from results_analysis.tidy import (
    REQUIRED_COLUMNS,
    TIDY_COLUMNS,
    to_tidy,
    validate_tidy,
)

# ── fixtures ──────────────────────────────────────────────────────────────────

GRAIN = ["EntityID", "consumption_type", "model", "param_set_id", "ds"]


def _forecast_series(start: str = "2025-01-01", n: int = 3) -> pd.Series:
    """A nested forecast Series like the one stored in performance_data_frame."""
    idx = pd.date_range(start, periods=n, freq="MS")
    return pd.Series(np.arange(1.0, n + 1.0) * 10.0, index=idx)


def _perf_frame(consumption_types, extra=None) -> pd.DataFrame:
    """A per-entity performance_data_frame: one row per consumption_type, each
    carrying a nested ``forecast`` Series plus the metric columns the runners emit."""
    rows = []
    for ct in consumption_types:
        row = {
            "pod_id": "E001",
            "customer_id": "C001",
            "consumption_type": ct,
            "forecast": _forecast_series(),
            "RMSE": 1.0,
            "MAE": 0.5,
            "R2": 0.9,
        }
        if extra:
            row.update(extra)
        rows.append(row)
    return pd.DataFrame(rows)


def _entity(entity_id="E001", tariff="LPU", etype="POD", method="ARIMA",
            consumption_types=("PeakConsumption", "OffPeakConsumption"), extra=None):
    return EntityPerformanceData(
        entity_id=entity_id,
        entity_type=etype,
        tariff_type=tariff,
        customer_id="C001",
        forecast_method_name=method,
        user_forecast_method_id=99,
        performance_data_frame=_perf_frame(list(consumption_types), extra=extra),
        tariff_id=7,
    )


def _results(method="ARIMA", entities=None) -> ForecastResults:
    res = ForecastResults(forecast_method_name=method)
    res.entity_performance = entities if entities is not None else [_entity(method=method)]
    return res


# ── to_tidy: schema + grain ─────────────────────────────────────────────────

def test_to_tidy_produces_full_schema_in_order():
    tidy = to_tidy(_results())
    assert list(tidy.columns) == TIDY_COLUMNS


def test_to_tidy_flattens_forecast_series_to_one_row_per_ds():
    # 2 consumption types × 3 forecast months = 6 rows.
    tidy = to_tidy(_results())
    assert len(tidy) == 6
    assert pd.api.types.is_datetime64_any_dtype(tidy["ds"])
    assert set(tidy["consumption_type"]) == {"PeakConsumption", "OffPeakConsumption"}
    # y_hat carries the nested Series values; CI + actual are point-only nulls.
    assert sorted(tidy["y_hat"].unique().tolist()) == [10.0, 20.0, 30.0]
    assert tidy["y"].isna().all()
    assert tidy["y_hat_lower"].isna().all()
    assert tidy["y_hat_upper"].isna().all()


def test_to_tidy_grain_is_unique():
    tidy = to_tidy(_results())
    assert not tidy.duplicated(subset=GRAIN).any()


def test_to_tidy_stamps_identifiers_and_model():
    tidy = to_tidy(_results(method="SARIMA"))
    assert (tidy["EntityID"] == "E001").all()
    assert (tidy["TariffType"] == "LPU").all()
    assert (tidy["EntityType"] == "POD").all()
    assert (tidy["model"] == "SARIMA").all()
    assert (tidy["param_set_id"] == "default").all()


def test_to_tidy_param_set_id_override():
    tidy = to_tidy(_results(), param_set_id="rf_n100")
    assert (tidy["param_set_id"] == "rf_n100").all()


def test_to_tidy_unions_across_models_via_concat():
    arima = to_tidy(_results(method="ARIMA"))
    xgb = to_tidy(_results(method="XGBoost"))
    unioned = pd.concat([arima, xgb], ignore_index=True)
    assert set(unioned["model"]) == {"ARIMA", "XGBoost"}
    assert not unioned.duplicated(subset=GRAIN).any()


def test_to_tidy_carries_validator_marks_where_present():
    extra = {"scenario": "gap", "validation_reason": "gaps ≤ 3 months — proceed with gap_handling"}
    tidy = to_tidy(_results(entities=[_entity(extra=extra)]))
    assert (tidy["scenario"] == "gap").all()
    assert (tidy["validation_reason"].str.startswith("gaps")).all()


def test_to_tidy_empty_results_returns_typed_empty_frame():
    tidy = to_tidy(_results(entities=[]))
    assert len(tidy) == 0
    assert list(tidy.columns) == TIDY_COLUMNS


def test_to_tidy_skips_rows_without_a_forecast_series():
    frame = _perf_frame(["PeakConsumption"])
    frame.loc[0, "forecast"] = None  # invalid / missing forecast
    entity = EntityPerformanceData(
        entity_id="E001", entity_type="POD", tariff_type="LPU", customer_id="C001",
        forecast_method_name="ARIMA", user_forecast_method_id=99,
        performance_data_frame=frame, tariff_id=7,
    )
    tidy = to_tidy(_results(entities=[entity]))
    assert len(tidy) == 0


# ── validate_tidy: required columns + coercion + tolerance ──────────────────

def test_validate_tidy_passes_through_to_tidy_output():
    tidy = to_tidy(_results())
    out = validate_tidy(tidy)
    assert list(out.columns) == TIDY_COLUMNS


@pytest.mark.parametrize("dropped", REQUIRED_COLUMNS)
def test_validate_tidy_rejects_missing_required_column(dropped):
    tidy = to_tidy(_results()).drop(columns=[dropped])
    with pytest.raises(ValueError, match=dropped):
        validate_tidy(tidy)


def test_validate_tidy_coerces_ds_to_datetime():
    df = pd.DataFrame({
        "EntityID": ["E001"], "TariffType": ["LPU"], "EntityType": ["POD"],
        "consumption_type": ["PeakConsumption"], "param_set_id": ["default"],
        "model": ["ARIMA"], "ds": ["2025-01-01"], "y_hat": [10.0],
    })
    out = validate_tidy(df)
    assert pd.api.types.is_datetime64_any_dtype(out["ds"])
    assert out["ds"].iloc[0] == pd.Timestamp("2025-01-01")


def test_validate_tidy_tolerates_point_only_frame():
    # No CI / actual / mark columns at all — should be accepted and back-filled.
    df = pd.DataFrame({
        "EntityID": ["E001"], "TariffType": ["LPU"], "EntityType": ["POD"],
        "consumption_type": ["PeakConsumption"], "param_set_id": ["default"],
        "model": ["ARIMA"], "ds": ["2025-01-01"], "y_hat": [10.0],
    })
    out = validate_tidy(df)
    assert list(out.columns) == TIDY_COLUMNS
    for col in ("y", "y_hat_lower", "y_hat_upper", "scenario", "validation_reason"):
        assert out[col].isna().all()
