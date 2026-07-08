"""
Tests for forecast_for_entity across ARIMA, XGBoost, and RandomForest.

Run from the project root:
    pytest tests/test_forecast_for_entity.py -v
"""
import os
import sys
import types
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.queries import ForecastConfig
from evaluation.performance import EntityPerformanceData, PredictionUnit, UnbundledResults
from results_analysis.tidy import to_tidy

# ── fixtures ──────────────────────────────────────────────────────────────────

def _make_series(entity_id: str, customer_id: str, n_months: int = 36) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-01", periods=n_months, freq="MS")
    values = rng.normal(100, 10, n_months) + np.arange(n_months) * 0.5
    return pd.DataFrame(
        {
            "PodID": [entity_id] * n_months,
            "CustomerID": [customer_id] * n_months,
            "PeakConsumption": values,
        },
        index=pd.DatetimeIndex(dates, name="ReportingMonth"),
    )


def _make_ufm_config(method_name: str) -> ForecastConfig:
    return ForecastConfig(
        forecast_method_id=1,
        forecast_method_name=method_name,
        model_parameters="",
        region="TEST",
        status="Active",
        user_forecast_method_id=99,
        start_date=pd.Timestamp("2025-01-01"),
        end_date=pd.Timestamp("2025-06-01"),
        databrick_task_id=0,
    )


def _make_forecast_model_stub():
    return types.SimpleNamespace(config=types.SimpleNamespace(log=False))


def _make_unit(tariff_type: str) -> PredictionUnit:
    return PredictionUnit(
        entity_id="E001",
        entity_type="POD",
        tariff_type=tariff_type,
        customer_id="C001",
        tariff_id=1,
        series=_make_series("E001", "C001"),
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _assert_entity_result(result, tariff_type: str):
    assert isinstance(result, EntityPerformanceData)
    assert result.entity_id == "E001"
    assert result.entity_type == "POD"
    assert result.tariff_type == tariff_type
    assert result.customer_id == "C001"
    assert result.performance_data_frame is not None
    assert len(result.performance_data_frame) > 0


TARIFF_TYPES = ["LPU", "SPU", "PPU"]

# ── ARIMA ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("tariff_type", TARIFF_TYPES)
def test_arima_forecast_for_entity(tariff_type):
    from models.algorithms.autoarima import forecast_for_entity

    unit = _make_unit(tariff_type)
    ufm_config = _make_ufm_config("ARIMA")
    order = (1, 1, 1)

    with patch("models.algorithms.autoarima.report_validation_error"):
        result = forecast_for_entity(unit, order, ufm_config, _make_forecast_model_stub())

    _assert_entity_result(result, tariff_type)


# ── SARIMA (shares the ARIMA path) ────────────────────────────────────────────

@pytest.mark.parametrize("tariff_type", TARIFF_TYPES)
def test_sarima_forecast_for_entity(tariff_type):
    from models.algorithms.autoarima import forecast_for_entity

    unit = _make_unit(tariff_type)
    ufm_config = _make_ufm_config("SARIMA")
    order = (1, 1, 1)
    seasonal_order = (1, 0, 0, 12)

    with patch("models.algorithms.autoarima.report_validation_error"):
        result = forecast_for_entity(
            unit, order, ufm_config, _make_forecast_model_stub(),
            seasonal_order=seasonal_order,
        )

    _assert_entity_result(result, tariff_type)


# ── XGBoost ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("tariff_type", TARIFF_TYPES)
def test_xgb_forecast_for_entity(tariff_type):
    from models.algorithms.tree_algorithms.xgb import forecast_for_entity

    unit = _make_unit(tariff_type)
    ufm_config = _make_ufm_config("XGBoost")

    with patch("models.algorithms.tree_algorithms.xgb.report_validation_error"):
        result = forecast_for_entity(unit, ufm_config, _make_forecast_model_stub())

    _assert_entity_result(result, tariff_type)


# ── RandomForest ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("tariff_type", TARIFF_TYPES)
def test_rf_forecast_for_entity(tariff_type):
    from models.algorithms.tree_algorithms.rf import forecast_for_entity

    unit = _make_unit(tariff_type)
    ufm_config = _make_ufm_config("RandomForest")

    with patch("models.algorithms.tree_algorithms.rf.report_validation_error"):
        result = forecast_for_entity(unit, ufm_config, _make_forecast_model_stub())

    _assert_entity_result(result, tariff_type)


# ══════════════════════════════════════════════════════════════════════════════
# Output tests — assert the actual predictions via the real to_tidy contract.
# ══════════════════════════════════════════════════════════════════════════════

MODELS = ["ARIMA", "SARIMA", "XGBoost", "RandomForest"]


def _forecast(model_name: str, unit: PredictionUnit, ufm_config, forecast_model=None):
    """Run one model's forecast_for_entity, patching the logger like the smoke tests."""
    fm = forecast_model or _make_forecast_model_stub()
    if model_name in ("ARIMA", "SARIMA"):
        from models.algorithms import autoarima
        seasonal = (1, 0, 0, 12) if model_name == "SARIMA" else None
        with patch.object(autoarima, "report_validation_error"):
            return autoarima.forecast_for_entity(unit, (1, 1, 1), ufm_config, fm, seasonal_order=seasonal)
    from models.algorithms.tree_algorithms import rf, xgb
    mod = {"RandomForest": rf, "XGBoost": xgb}[model_name]
    with patch.object(mod, "report_validation_error"):
        return mod.forecast_for_entity(unit, ufm_config, fm)


def _tidy(epd: EntityPerformanceData, model_name: str) -> pd.DataFrame:
    """Flatten one entity's result through the production tidy adapter."""
    results = UnbundledResults(forecast_method_name=model_name)
    results.entity_performance.append(epd)
    return to_tidy(results)


def _future_rows(tidy: pd.DataFrame) -> pd.DataFrame:
    return tidy[tidy["is_forecast"] == True]  # noqa: E712 — nullable-boolean mask


# ── 1. Horizon correctness — forecast index == the ufm_config monthly range ─────

@pytest.mark.parametrize("model_name", MODELS)
def test_forecast_horizon_matches_ufm_range(model_name):
    unit = _make_unit("LPU")
    ufm = _make_ufm_config(model_name)
    epd = _forecast(model_name, unit, ufm)
    fc = _future_rows(_tidy(epd, model_name))

    expected = list(pd.date_range(ufm.start_date, ufm.end_date, freq="MS"))
    for ctype, grp in fc.groupby("consumption_type"):
        got = sorted(pd.to_datetime(grp["ds"]).tolist())
        assert got == expected, f"{model_name}/{ctype}: {got} != {expected}"
        assert len(grp) == len(expected)  # no rows dropped or duplicated


# ── 2. Value sanity — finite forecasts (no NaN/inf) ─────────────────────────────

@pytest.mark.parametrize("model_name", MODELS)
def test_forecast_values_are_finite(model_name):
    epd = _forecast(model_name, _make_unit("LPU"), _make_ufm_config(model_name))
    yhat = _future_rows(_tidy(epd, model_name))["y_hat"].astype(float)
    assert len(yhat) > 0
    assert np.isfinite(yhat).all()
    # NB: non-negativity is NOT asserted — plain (log=False) ARIMA is unconstrained
    # and may forecast negative; non-negativity only holds under log mode (below).


# ── 6. Short series — routes through the validation path, does not raise ─────────

def test_arima_short_series_reports_reason_without_raising():
    from models.algorithms import autoarima

    short = PredictionUnit(
        entity_id="E001", entity_type="POD", tariff_type="LPU",
        customer_id="C001", tariff_id=1,
        series=_make_series("E001", "C001", n_months=6),  # below the per-channel min
    )
    ufm = _make_ufm_config("ARIMA")
    with patch.object(autoarima, "report_validation_error") as rep:
        epd = autoarima.forecast_for_entity(short, (1, 1, 1), ufm, _make_forecast_model_stub())

    assert epd is not None            # graceful: no exception
    assert rep.called                 # the short-series path was reported
    tidy = _tidy(epd, "ARIMA")
    assert (tidy["validation_reason"] == "series too short").any()
    # still emits a (zero) forecast covering the horizon rather than dropping rows
    assert len(_future_rows(tidy)) == len(pd.date_range(ufm.start_date, ufm.end_date, freq="MS"))


# ── 7. NaN months + zero consumption — handled without crashing ─────────────────

def test_arima_handles_nan_and_zero_without_crashing():
    from models.algorithms import autoarima

    series = _make_series("E001", "C001", n_months=24)
    col = series.columns.get_loc("PeakConsumption")
    series.iloc[3, col] = np.nan
    series.iloc[5, col] = 0.0
    unit = PredictionUnit("E001", "POD", "LPU", "C001", 1, series)
    ufm = _make_ufm_config("ARIMA")

    with patch.object(autoarima, "report_validation_error"):
        epd = autoarima.forecast_for_entity(unit, (1, 1, 1), ufm, _make_forecast_model_stub())

    fc = _future_rows(_tidy(epd, "ARIMA"))
    assert len(fc) > 0                                      # produced output, no crash
    assert np.isfinite(fc["y_hat"].astype(float)).all()    # NaN did not propagate


# ── 8 (corrected). Log mode round-trips through the forecast path → non-negative ─

def test_arima_log_mode_yields_non_negative_forecast():
    from models.algorithms import autoarima

    log_stub = types.SimpleNamespace(config=types.SimpleNamespace(log=True))
    with patch.object(autoarima, "report_validation_error"):
        epd = autoarima.forecast_for_entity(_make_unit("LPU"), (1, 1, 1),
                                            _make_ufm_config("ARIMA"), log_stub)
    yhat = _future_rows(_tidy(epd, "ARIMA"))["y_hat"].astype(float)
    assert np.isfinite(yhat).all()
    assert (yhat >= 0).all()   # exp back-transform of the log-fit forecast


# ── primitive: fit_time_series_model dispatches ARIMA vs SARIMA ──────────────────

def test_fit_time_series_model_dispatch():
    from models.algorithms.autoarima import fit_time_series_model

    series = _make_series("E", "C", n_months=36)["PeakConsumption"]
    arima = fit_time_series_model(series, (1, 1, 1), None)
    sarima = fit_time_series_model(series, (1, 1, 1), (1, 0, 0, 12))

    assert arima.model.seasonal_order == (0, 0, 0, 0)      # plain ARIMA path
    assert sarima.model.seasonal_order == (1, 0, 0, 12)    # seasonal path
