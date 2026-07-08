"""
Tests for the dashboard app's gradio-free logic (forecasting.dashboard_app).

The Gradio shell is not exercised here (optional dep); the cache / train-on-gap /
view-assembly logic is. Uses a tiny dataset so the real engine train is quick.

Run from the project root:
    pytest tests/test_dashboard_app.py -v
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forecasting.dashboard_app import (
    DashboardState,
    channels_for,
    config_choices,
    forecast_view,
    is_trained,
    metrics_view,
    pods_for,
    scenarios_for,
    subject_tidy,
    train_config,
)


def _raw():
    months = pd.date_range("2022-01-01", periods=30, freq="MS")
    rng = np.random.default_rng(5)
    rows = []
    for k, m in enumerate(months):
        season = 40 * np.sin(k / 12 * 2 * np.pi)
        rows.append({"EntityID": "POD_A", "TariffType": "LPU", "EntityType": "POD",
                     "ReportingMonth": m, "PeakConsumption": 300 + season + rng.normal(0, 6),
                     "StandardConsumption": 0.0, "OffPeakConsumption": 0.0, "NonTOUConsumption": 0.0})
    # a too-short entity → scenario filter
    for m in months[:8]:
        rows.append({"EntityID": "POD_SHORT", "TariffType": "LPU", "EntityType": "POD",
                     "ReportingMonth": m, "PeakConsumption": 200.0,
                     "StandardConsumption": 0.0, "OffPeakConsumption": 0.0, "NonTOUConsumption": 0.0})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def state():
    return DashboardState(raw_df=_raw(), horizon_months=6)


# ── picker (scenario filter from the real validator) ────────────────────────────


def test_pods_filtered_by_scenario(state):
    assert set(pods_for(state, "LPU", "(any)")) == {"POD_A", "POD_SHORT"}
    assert pods_for(state, "LPU", "short_history") == ["POD_SHORT"]


def test_channels_for_offers_only_populated_channels_busiest_first():
    # An entity that uses NonTOU (not Peak) must not default the view to all-zero Peak.
    months = pd.date_range("2022-01-01", periods=24, freq="MS")
    rows = [{"EntityID": "COMBO", "TariffType": "SPU", "EntityType": "Combo",
             "ReportingMonth": m, "PeakConsumption": 0.0, "StandardConsumption": 0.0,
             "OffPeakConsumption": 0.0, "NonTOUConsumption": 500.0 + i}
            for i, m in enumerate(months)]
    st = DashboardState(raw_df=pd.DataFrame(rows))
    chans = channels_for(st, "COMBO")
    assert chans[0] == "NonTOUConsumption"      # the populated one, first
    assert "PeakConsumption" not in chans       # all-zero channel excluded


def test_scenarios_for_only_lists_present_conditions(state):
    # The picker must only offer scenarios that actually have PodIDs in the tariff
    # (regression: SPU + happy_path emptied the picker because no SPU is happy_path).
    scen = scenarios_for(state, "LPU")
    assert scen[0] == "(any)"
    assert "short_history" in scen          # POD_SHORT is in LPU
    present = set(state.scenarios_df[state.scenarios_df.TariffType == "LPU"]["scenario"])
    assert set(scen) - {"(any)"} == present  # nothing offered that has zero pods


# ── retrieve-or-train cache ──────────────────────────────────────────────────────


def test_train_then_browse_is_cached(state):
    assert not is_trained(state, "ARIMA", "(2,1,2)")
    train_config(state, "ARIMA", "(2,1,2)")
    assert is_trained(state, "ARIMA", "(2,1,2)")
    assert "(2,1,2)" in config_choices(state, "ARIMA")


def test_subject_tidy_pulls_only_cached_configs(state):
    train_config(state, "ARIMA", "(1,1,1)")
    tidy = subject_tidy(state, "POD_A", "PeakConsumption",
                        {"ARIMA": "(1,1,1)", "XGBoost": "(100,5,0.1,0.8,0.8)"})
    # ARIMA is cached → present; XGBoost not trained → absent.
    assert set(tidy["model"]) == {"ARIMA"}


def test_forecast_view_empty_until_trained(state):
    fig = forecast_view(state, "POD_A", "PeakConsumption", False,
                        {"SARIMA": "(0,1,1)(0,1,1,12)"})  # not trained
    assert "click Train" in fig.layout.title.text


def test_forecast_view_draws_after_training(state):
    train_config(state, "ARIMA", "(2,1,2)")
    fig = forecast_view(state, "POD_A", "PeakConsumption", False, {"ARIMA": "(2,1,2)"})
    names = [t.name for t in fig.data]
    assert any(n and n.startswith("ARIMA") for n in names)  # model line drawn
    assert any(t.name == "actual" for t in fig.data)         # actuals overlaid


def test_metrics_view_uses_cache(state):
    train_config(state, "ARIMA", "(2,1,2)")
    fig = metrics_view(state, "RMSE")
    # either boxes (data present) or a friendly placeholder, never an exception
    assert fig is not None
