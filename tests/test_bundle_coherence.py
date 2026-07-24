"""Output-contract parity + coherence for the bundle path (plan 06).

The bundled path reuses ``ForecastResults`` / ``EntityPerformanceData`` unchanged, so
``to_forecast_fact()`` on a bundled run has the SAME columns and dtypes as an unbundled
run of the same UFM; disaggregated members reconcile to the bundle every month; and
neither mode writes — plan 01 removed the DB-write boundary entirely.
"""
import os
import sys
import types

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.algorithms import bundled
from models.algorithms.autoarima import forecast_arima_unbundled
from models.bundle import forecast_for_bundle

FIXTURE = "data/fixtures/Results_422.csv"


def _model(method="SARIMA"):
    """A ForecastModel stand-in pointing at the Results_422 fixture (both paths read it
    via PREDICTIVE_FIXTURE_PATH)."""
    fx = pd.read_csv(FIXTURE, encoding="utf-8-sig")
    mx = pd.to_datetime(fx["ReportingMonth"]).max()
    cfg = types.SimpleNamespace(
        forecast_method_name=method, model_parameters="",
        user_forecast_method_id=int(fx["UserForecastMethodID"].iloc[0]),
        start_date=mx + pd.DateOffset(months=1), end_date=mx + pd.DateOffset(months=6))
    return types.SimpleNamespace(
        dataset=types.SimpleNamespace(ufm_config=cfg),
        config=types.SimpleNamespace(log=False))


def test_bundled_columns_match_unbundled(monkeypatch):
    """Same UFM, both ways: identical ForecastFact columns and dtypes."""
    monkeypatch.setenv("PREDICTIVE_FIXTURE_PATH", FIXTURE)
    unbundled_fact = forecast_arima_unbundled(_model(), spark=None).to_forecast_fact()
    bundled_fact = bundled.run_bundled(_model(), spark=None).to_forecast_fact()
    assert list(unbundled_fact.columns) == list(bundled_fact.columns), (
        f"contract drift:\n  unbundled {list(unbundled_fact.columns)}"
        f"\n  bundled   {list(bundled_fact.columns)}")
    assert unbundled_fact.dtypes.equals(bundled_fact.dtypes), "dtype drift"


def test_members_sum_to_bundle():
    """Disaggregation is coherent by construction: members sum to the bundle forecast."""
    idx = pd.date_range("2023-01-01", periods=24, freq="MS")
    members = pd.DataFrame(
        {"A": np.linspace(100, 200, 24), "B": np.linspace(50, 80, 24)}, index=idx)

    def _last_value_fit(series, horizon):
        future = pd.date_range(series.index[-1], periods=horizon + 1, freq="MS")[1:]
        return pd.Series(float(series.iloc[-1]), index=future)

    parts = forecast_for_bundle(members, method="SARIMA", horizon=6, fit_fn=_last_value_fit)
    bundle_forecast = _last_value_fit(members.sum(axis=1), 6)
    assert np.allclose(parts.sum(axis=1).to_numpy(), bundle_forecast.to_numpy()), \
        "members must sum to the bundle forecast every month"


def test_no_writes_on_either_mode():
    """Neither mode can write: plan 01 removed the DB-write boundary entirely, so no
    write-capable function is left in db.utilities for either path to reach."""
    import db.utilities
    write_fns = [n for n in dir(db.utilities)
                 if "write" in n.lower() and callable(getattr(db.utilities, n))]
    assert write_fns == [], f"a DB-write boundary still exists: {write_fns}"
