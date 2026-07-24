"""Tests for the PodID-contract fixture loader (db.queries.get_unbundled_predictive_data).

Locks down the plan-01/02 loader contract:
  * the fixture env vars short-circuit the DB entirely (CSV and parquet capable),
  * a `PredictiveInputData`-shaped CSV loads PodID-keyed with the UTF-8 BOM stripped,
  * the live path issues exactly ONE `PredictiveInputData(UFMID)` query,
  * the decomposed `db/unbundled_query.py` apparatus stays retired.

The CSV fixture is synthesised in-test (BOM header, fake PodIDs/values) so the suite
carries no real customer data — the production loader is exercised the same way.

Run from the project root:
    pytest tests/test_fixture_loader.py -v
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from db.queries import get_unbundled_predictive_data

PODID_CONTRACT_COLUMNS = {
    "UserForecastMethodID", "PodID", "CustomerID", "TariffID", "TariffType",
    "TariffSubType", "ReportingMonth",
    "PeakConsumption", "StandardConsumption", "OffPeakConsumption",
}


@pytest.fixture
def results_csv(tmp_path):
    """A synthetic PredictiveInputData(421)-shaped CSV — fake data, UTF-8-BOM header
    (so the loader's ``utf-8-sig`` BOM-stripping is exercised). Stands in for the
    real ``data/fixtures/Results.csv`` so no customer data lives in the repo."""
    path = tmp_path / "Results.csv"
    dates = pd.date_range("2023-04-01", periods=24, freq="MS").strftime("%Y-%m-%d")
    rows = [
        {"UserForecastMethodID": 421, "PodID": pod, "CustomerID": "6000000000",
         "TariffID": "RATE", "TariffType": "Consumption", "TariffSubType": "Consumption",
         "ReportingMonth": d, "PeakConsumption": 100.0,
         "StandardConsumption": 200.0, "OffPeakConsumption": 50.0}
        for pod in ("P1.RATE", "P2.RATE", "P3.RATE") for d in dates
    ]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")   # BOM header
    return path


# ── fixture path: CSV loads PodID-keyed, no Spark ─────────────────────────────


def test_loads_results_csv_podid_keyed(monkeypatch, results_csv):
    monkeypatch.setenv("PREDICTIVE_FIXTURE_PATH", str(results_csv))
    df = get_unbundled_predictive_data(None, UFMID=421)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == PODID_CONTRACT_COLUMNS   # BOM stripped, no ﻿ column
    assert df["PodID"].nunique() > 0
    assert (df["UserForecastMethodID"] == 421).all()
    assert (df["TariffType"] == "Consumption").all()


def test_unbundled_fixture_var_takes_precedence(monkeypatch, tmp_path, results_csv):
    other = tmp_path / "other.parquet"
    pd.DataFrame({"PodID": ["P1"], "PeakConsumption": [1.0]}).to_parquet(other)
    monkeypatch.setenv("PREDICTIVE_FIXTURE_PATH", str(results_csv))
    monkeypatch.setenv("UNBUNDLED_FIXTURE_PATH", str(other))
    df = get_unbundled_predictive_data(None, UFMID=421)
    assert list(df["PodID"]) == ["P1"]                 # the override won


def test_fixture_hook_does_not_touch_spark(monkeypatch, results_csv):
    monkeypatch.setenv("PREDICTIVE_FIXTURE_PATH", str(results_csv))
    spark = MagicMock()
    get_unbundled_predictive_data(spark, UFMID=421)
    spark.read.format.assert_not_called()


# ── live path: exactly one PredictiveInputData(UFMID) call ────────────────────


def test_live_path_issues_single_predictive_input_data_query(monkeypatch):
    monkeypatch.delenv("PREDICTIVE_FIXTURE_PATH", raising=False)
    monkeypatch.delenv("UNBUNDLED_FIXTURE_PATH", raising=False)
    with patch("db.queries.read_sql_query") as rsq:
        get_unbundled_predictive_data(spark="SPARK", UFMID=420)
    assert rsq.call_count == 1                          # one call, no per-filter fan-out
    query = rsq.call_args.args[0]
    assert query.strip() == "SELECT * FROM dbo.PredictiveInputData(420)"


# ── retirement: the decomposed query apparatus stays gone ─────────────────────


def test_unbundled_query_module_stays_retired():
    assert not (PROJECT_ROOT / "db" / "unbundled_query.py").exists()
    source = (PROJECT_ROOT / "db" / "queries.py").read_text()
    for symbol in ("unbundled_query", "UNBUNDLED_FILTERS", "to_contract", "build_query"):
        assert symbol not in source


# ── SAVE_UNBUNDLED_FIXTURE: persist the fetched input frame for gradio/QA ──────

import types

from evaluation.performance import EntityPerformanceData
from models.algorithms import unbundled


def _podid_frame(ufmid=422):
    dates = pd.date_range("2024-01-01", periods=24, freq="MS")
    rows = []
    for pod in ("P1", "P2"):
        for d in dates:
            rows.append({
                "UserForecastMethodID": ufmid, "PodID": pod, "CustomerID": "C1",
                "TariffID": "T1", "TariffType": "Consumption", "TariffSubType": "Consumption",
                "ReportingMonth": d, "PeakConsumption": 10.0,
                "StandardConsumption": 20.0, "OffPeakConsumption": 5.0,
            })
    return pd.DataFrame(rows)


def _model_stub(ufmid=422):
    return types.SimpleNamespace(
        dataset=types.SimpleNamespace(ufm_config=types.SimpleNamespace(
            forecast_method_name="SARIMA", user_forecast_method_id=ufmid)),
        config=types.SimpleNamespace(log=False))


def _stub_forecast(unit, ufm_config, m):
    return EntityPerformanceData(unit.entity_id, "", unit.tariff_type, unit.customer_id,
                                 "SARIMA", 422, pd.DataFrame({"value": [1.0]}))


def test_save_hook_writes_results_csv_when_flag_set(monkeypatch, tmp_path):
    monkeypatch.setenv("SAVE_UNBUNDLED_FIXTURE", str(tmp_path / "Results_422.csv"))
    with patch.object(unbundled, "get_unbundled_predictive_data", return_value=_podid_frame(422)):
        unbundled.run_unbundled(_model_stub(422), spark=None, forecast_for_entity=_stub_forecast)
    out = tmp_path / "Results_422.csv"
    assert out.exists()
    saved = pd.read_csv(out)
    assert PODID_CONTRACT_COLUMNS <= set(saved.columns)
    assert (saved["UserForecastMethodID"] == 422).all()


def test_save_hook_noop_when_flag_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("SAVE_UNBUNDLED_FIXTURE", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "fixtures").mkdir(parents=True)
    with patch.object(unbundled, "get_unbundled_predictive_data", return_value=_podid_frame(422)):
        unbundled.run_unbundled(_model_stub(422), spark=None, forecast_for_entity=_stub_forecast)
    assert list((tmp_path / "data" / "fixtures").iterdir()) == []   # nothing written
