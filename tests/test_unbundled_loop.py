"""
Loop-isolation tests for forecast_arima_unbundled, forecast_xgb_unbundled,
forecast_rf_unbundled.

Run from the project root:
    pytest tests/test_unbundled_loop.py -v
"""
import os
import sys
import types
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.queries import ForecastConfig
from evaluation.performance import EntityPerformanceData, UnbundledResults

# ── fixture data ──────────────────────────────────────────────────────────────

TARIFF_TYPES = ["LPU", "SPU", "PPU"]
ENTITY_IDS = ["E001", "E002"]
N_MONTHS = 24

ENTITY_TYPE_MAP = {"LPU": "POD", "SPU": "Combo", "PPU": "CSA"}


def _make_mock_data() -> pd.DataFrame:
    """3 tariff types × 2 entities = 6 groups, matching fixture contract columns."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-01-01", periods=N_MONTHS, freq="MS")
    rows = []
    for tariff_type in TARIFF_TYPES:
        for entity_id in ENTITY_IDS:
            for date in dates:
                rows.append({
                    "TariffType": tariff_type,
                    "EntityID": entity_id,
                    "EntityType": ENTITY_TYPE_MAP[tariff_type],
                    "CustomerID": f"C_{entity_id}",
                    "TariffID": TARIFF_TYPES.index(tariff_type) + 1,
                    "ReportingMonth": date,
                    "TotalConsumption": float(rng.normal(100, 10)),
                })
    return pd.DataFrame(rows)


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


def _make_model_stub(method_name: str):
    ufm_config = _make_ufm_config(method_name)
    dataset = types.SimpleNamespace(ufm_config=ufm_config)
    return types.SimpleNamespace(
        dataset=dataset,
        config=types.SimpleNamespace(log=False),
    )


def _entity_perf_stub(unit, *args, **kwargs) -> EntityPerformanceData:
    return EntityPerformanceData(
        entity_id=unit.entity_id,
        entity_type=unit.entity_type,
        tariff_type=unit.tariff_type,
        customer_id=unit.customer_id,
        forecast_method_name="TEST",
        user_forecast_method_id=99,
        performance_data_frame=pd.DataFrame({"value": [1.0]}),
    )


MOCK_DATA = _make_mock_data()
N_GROUPS = len(TARIFF_TYPES) * len(ENTITY_IDS)  # 6

# ── helpers ───────────────────────────────────────────────────────────────────

def _run_loop(import_path, func_name, method_name, data_patch_path, entity_patch_path):
    model = _make_model_stub(method_name)
    spy = MagicMock(side_effect=_entity_perf_stub)
    with patch(data_patch_path, return_value=MOCK_DATA), \
         patch(entity_patch_path, spy):
        module = __import__(import_path, fromlist=[func_name])
        fn = getattr(module, func_name)
        result = fn(model, spark=None)
    return result, spy


# ── ARIMA ─────────────────────────────────────────────────────────────────────

def test_arima_loop_groups_correctly():
    result, spy = _run_loop(
        "models.algorithms.autoarima",
        "forecast_arima_unbundled",
        "ARIMA",
        "models.algorithms.autoarima.get_predictive_data",
        "models.algorithms.autoarima.forecast_for_entity",
    )
    assert isinstance(result, UnbundledResults)
    assert spy.call_count == N_GROUPS
    assert len(result.entity_performance) == N_GROUPS


def test_xgb_loop_groups_correctly():
    result, spy = _run_loop(
        "models.algorithms.tree_algorithms.xgb",
        "forecast_xgb_unbundled",
        "XGBoost",
        "models.algorithms._unbundled.get_predictive_data",
        "models.algorithms.tree_algorithms.xgb.forecast_for_entity",
    )
    assert isinstance(result, UnbundledResults)
    assert spy.call_count == N_GROUPS
    assert len(result.entity_performance) == N_GROUPS


def test_rf_loop_groups_correctly():
    result, spy = _run_loop(
        "models.algorithms.tree_algorithms.rf",
        "forecast_rf_unbundled",
        "RandomForest",
        "models.algorithms._unbundled.get_predictive_data",
        "models.algorithms.tree_algorithms.rf.forecast_for_entity",
    )
    assert isinstance(result, UnbundledResults)
    assert spy.call_count == N_GROUPS
    assert len(result.entity_performance) == N_GROUPS


# ── entity_id and tariff_type correctness ─────────────────────────────────────

def _extract_unit_kwargs(spy) -> list:
    return [c.args[0] for c in spy.call_args_list]


def test_loop_entity_ids_correct():
    """Each call receives the correct entity_id and tariff_type for one algorithm."""
    _, spy = _run_loop(
        "models.algorithms.autoarima",
        "forecast_arima_unbundled",
        "ARIMA",
        "models.algorithms.autoarima.get_predictive_data",
        "models.algorithms.autoarima.forecast_for_entity",
    )
    units = _extract_unit_kwargs(spy)
    seen = {(u.tariff_type, u.entity_id) for u in units}
    expected = {(t, e) for t in TARIFF_TYPES for e in ENTITY_IDS}
    assert seen == expected
