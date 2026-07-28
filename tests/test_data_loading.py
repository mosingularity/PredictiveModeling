"""Integration tests for the data-loading path — a behavior snapshot before the
query-layer refactor.

Covers the three seams, driven through the ``PREDICTIVE_FIXTURE_PATH`` env hook:

  * db/queries.py        — get_predictive_data / get_unbundled_predictive_data
  * data/dml.py          — load_and_prepare_data / clean_dataframe
  * data/dataset.py      — ForecastDataset.load_data

The query-seam tests run against the REAL committed parquets in data/fixtures/.
The prep/dataset tests use a synthetic-but-valid bundled input written to a temp
parquet, because none of the committed fixtures satisfy the bundled prep contract
(raw_predictive_input.parquet lost ReportingMonth when saved with index=False; the
unbundled fixtures carry no PodID) — see the coverage note at the bottom of the file.

Run from the project root:
    pytest tests/test_data_loading.py -v
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from db.queries import ForecastConfig, get_predictive_data, get_unbundled_predictive_data
from data.dml import clean_dataframe, load_and_prepare_data
from data.dataset import ForecastDataset

FIXTURES = PROJECT_ROOT / "data" / "fixtures"

CONSUMPTION_COLS = [
    "PeakConsumption", "StandardConsumption", "OffPeakConsumption",
    "Block1Consumption", "Block2Consumption", "Block3Consumption",
    "Block4Consumption", "NonTOUConsumption",
]


def _ufm_config(method_name: str = "ARIMA") -> ForecastConfig:
    return ForecastConfig(
        forecast_method_id=1,
        forecast_method_name=method_name,
        model_parameters="",
        region="LOCAL",
        status="Active",
        user_forecast_method_id=99,
        start_date=pd.Timestamp("2025-01-01"),
        end_date=pd.Timestamp("2025-06-01"),
        databrick_task_id=0,
    )


def _bundled_input(pods=("POD_A", "POD_B"), months=6) -> pd.DataFrame:
    """A DB-shaped bundled input: ReportingMonth + PodID + CustomerID + consumption.

    This is the shape get_predictive_data returns from the live DB and that
    load_and_prepare_data expects — the committed bundled fixture no longer has it.
    """
    dates = pd.date_range("2024-01-01", periods=months, freq="MS").strftime("%Y-%m-%d")
    rows = []
    for i, pod in enumerate(pods):
        for d in dates:
            row = {"ReportingMonth": d, "PodID": pod, "CustomerID": 1000 + i, "TariffID": 7}
            row.update({c: float(10 + i) for c in CONSUMPTION_COLS})
            rows.append(row)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Part A — query seam (db/queries.py), against the REAL committed fixtures
# ══════════════════════════════════════════════════════════════════════════════

def test_get_predictive_data_returns_bundled_fixture(monkeypatch):
    monkeypatch.setenv("PREDICTIVE_FIXTURE_PATH", str(FIXTURES / "raw_predictive_input.parquet"))
    df = get_predictive_data(None, UFMID=64)
    assert isinstance(df, pd.DataFrame)         # seam contract: pandas, not Spark
    assert df.shape == (12, 11)
    assert set(df.columns) == {"PodID", "CustomerID", "TariffID", *CONSUMPTION_COLS}


def test_get_unbundled_data_returns_ermelo_fixture(monkeypatch):
    # NOTE (PodID migration): these two tests assert LOADER behavior only — the
    # fixture file is returned verbatim, whatever its shape. The entity-keyed
    # frames they read are legacy artifacts; the production loop now requires the
    # PodID contract (see tests/test_fixture_loader.py for the contract tests).
    monkeypatch.setenv("PREDICTIVE_FIXTURE_PATH", str(FIXTURES / "unbundled_real_ermelo.parquet"))
    df = get_unbundled_predictive_data(None, UFMID=64)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (150, 14)
    # legacy entity-keyed columns, returned as-is by the loader
    assert {"EntityID", "TariffType", "TariffID", "EntityType",
            "ReportingMonth", *CONSUMPTION_COLS} <= set(df.columns)


def test_get_unbundled_data_returns_synthetic_fixture(monkeypatch):
    monkeypatch.setenv("PREDICTIVE_FIXTURE_PATH",
                       str(FIXTURES / "unbundled_predictive_input.parquet"))
    df = get_unbundled_predictive_data(None, UFMID=64)
    assert df.shape == (218, 15)
    assert {"CustomerID", "Scenario"} <= set(df.columns)


def test_fixture_hook_does_not_touch_spark(monkeypatch):
    """The fixture branch must short-circuit before any Spark call."""
    monkeypatch.setenv("PREDICTIVE_FIXTURE_PATH", str(FIXTURES / "raw_predictive_input.parquet"))
    spark = MagicMock()
    get_predictive_data(spark, UFMID=64)
    spark.read.format.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# Part B — prepared frame (data/dml.py), synthetic valid input via the env hook
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def bundled_fixture(tmp_path, monkeypatch):
    """Write a valid bundled input to a temp parquet and point the hook at it."""
    path = tmp_path / "bundled_input.parquet"
    _bundled_input().to_parquet(path, index=False)
    monkeypatch.setenv("PREDICTIVE_FIXTURE_PATH", str(path))
    monkeypatch.setenv("ENV", "DEV")
    monkeypatch.chdir(tmp_path)   # keep the dataset/ cache dir out of the repo
    return path


def test_load_and_prepare_sets_reporting_month_index(bundled_fixture):
    df = load_and_prepare_data(_ufm_config(), spark=MagicMock())
    assert df.index.name == "ReportingMonth"
    assert pd.api.types.is_datetime64_any_dtype(df.index)


def test_load_and_prepare_shape_and_columns(bundled_fixture):
    df = load_and_prepare_data(_ufm_config(), spark=MagicMock())
    assert df.shape == (12, 11)   # 2 pods × 6 months; ReportingMonth moved to index
    assert set(df.columns) == {"PodID", "CustomerID", "TariffID", *CONSUMPTION_COLS}


def test_load_and_prepare_casts_ids_to_str(bundled_fixture):
    df = load_and_prepare_data(_ufm_config(), spark=MagicMock())
    assert df["PodID"].dtype == object
    assert df["CustomerID"].dtype == object
    assert df["CustomerID"].iloc[0] == str(df["CustomerID"].iloc[0])


def test_load_and_prepare_sorted_by_pod_then_month(bundled_fixture):
    df = load_and_prepare_data(_ufm_config(), spark=MagicMock())
    ordering = list(zip(df["PodID"], df.index))
    assert ordering == sorted(ordering, key=lambda t: (t[0], t[1]))


def test_load_and_prepare_empty_fixture_returns_none(tmp_path, monkeypatch):
    empty = tmp_path / "empty.parquet"
    pd.DataFrame(columns=["PodID", "CustomerID", "ReportingMonth"]).to_parquet(empty, index=False)
    monkeypatch.setenv("PREDICTIVE_FIXTURE_PATH", str(empty))
    monkeypatch.setenv("ENV", "DEV")
    monkeypatch.chdir(tmp_path)
    assert load_and_prepare_data(_ufm_config(), spark=MagicMock()) is None


def test_clean_dataframe_drops_unnamed_and_constant_columns():
    raw = _bundled_input()
    raw.insert(0, "Unnamed: 0", range(len(raw)))   # CSV index artifact
    raw["UserForecastMethodID"] = 99               # constant → dropped
    cleaned = clean_dataframe(raw)
    assert not any(c.startswith("Unnamed") for c in cleaned.columns)
    assert "UserForecastMethodID" not in cleaned.columns
    assert cleaned.index.name == "ReportingMonth"


# ══════════════════════════════════════════════════════════════════════════════
# Part C — ForecastDataset.load_data (data/dataset.py)
# ══════════════════════════════════════════════════════════════════════════════

def test_forecast_dataset_load_data_prepares_raw_df(bundled_fixture):
    with patch.object(ForecastDataset, "load_ufm_config", return_value=_ufm_config()):
        ds = ForecastDataset(databrick_task_id=1, spark=MagicMock())
        ds.load_data()
    assert ds.raw_df is not None
    assert ds.raw_df.shape == (12, 11)
    assert ds.raw_df.index.name == "ReportingMonth"
    assert set(ds.raw_df.columns) == {"PodID", "CustomerID", "TariffID", *CONSUMPTION_COLS}


def test_forecast_dataset_processed_df_aliases_raw_df(bundled_fixture):
    with patch.object(ForecastDataset, "load_ufm_config", return_value=_ufm_config()):
        ds = ForecastDataset(databrick_task_id=1, spark=MagicMock())
        ds.load_data()
    assert ds.processed_df is ds.raw_df
