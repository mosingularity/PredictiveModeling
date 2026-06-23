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
from evaluation.performance import EntityPerformanceData, PredictionUnit

# ── fixtures ──────────────────────────────────────────────────────────────────

def _make_series(entity_id: str, customer_id: str, n_months: int = 36) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-01", periods=n_months, freq="MS")
    values = rng.normal(100, 10, n_months) + np.arange(n_months) * 0.5
    return pd.DataFrame(
        {
            "PodID": [entity_id] * n_months,
            "CustomerID": [customer_id] * n_months,
            "TotalConsumption": values,
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
