"""Real-forecaster tests: bundle_forecaster + forecast_for_bundle (plan 03).

The bundle fits ONE model on the aggregate through an injected fit_fn; bundle_forecaster
builds that fit_fn for each family (ARIMA / SARIMA / RandomForest / XGBoost) from the UFM
config, so the placeholder _naive_seasonal_forecast is never the real estimator.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bundle import forecast_for_bundle
from models.bundle_forecasters import bundle_forecaster


def _members(cols, months=30):
    """Positive series with trend + seasonality, enough history for every family
    (SARIMA needs >=18 months, the tree STL needs two seasons)."""
    idx = pd.date_range("2022-07-01", periods=months, freq="MS")
    t = np.arange(months)
    return pd.DataFrame(
        {c: 1000 + 100 * i + 50 * np.sin(2 * np.pi * t / 12) + 10 * t
         for i, c in enumerate(cols)}, index=idx)


@pytest.mark.parametrize("method", ["ARIMA", "SARIMA", "RandomForest", "XGBoost"])
def test_each_family_fits_via_fit_fn(method):
    members = _members(["P1", "P2"])
    parts = forecast_for_bundle(members, method=method, horizon=6,
                                fit_fn=bundle_forecaster(method))
    assert parts.shape == (6, 2)                 # horizon x members
    assert not parts.isna().any().any()          # a real forecast, no gaps


def test_real_fit_differs_from_naive():
    members = _members(["P1", "P2"])
    naive = forecast_for_bundle(members, method="SARIMA", horizon=6)                       # fit_fn=None
    real = forecast_for_bundle(members, method="SARIMA", horizon=6,
                               fit_fn=bundle_forecaster("SARIMA"))
    assert not naive.equals(real), "fit_fn is not being used — still on the placeholder"


def test_exactly_one_fit_per_bundle():
    """One fit on the aggregate, not one per member — the whole point of the path."""
    calls = {"n": 0}

    def _counting_fit(series, horizon):
        calls["n"] += 1
        future = pd.date_range(series.index[-1], periods=horizon + 1, freq="MS")[1:]
        return pd.Series(float(series.iloc[-1]), index=future)

    members = _members(["A", "B", "C"])          # three members
    forecast_for_bundle(members, method="SARIMA", horizon=6, fit_fn=_counting_fit)
    assert calls["n"] == 1                        # one aggregate fit, not three
