"""
Tests for the dashboard figure + data layer (results_analysis.figures.dashboard).

Run from the project root:
    pytest tests/test_dashboard.py -v
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from results_analysis.figures.dashboard import (
    build_forecast_figure,
    build_metrics_figure,
    entity_scenarios,
)
from results_analysis.tidy import validate_tidy

# ── a tidy slice carrying the seam columns (is_forecast + real metrics) ──────────


def _tidy_subject(model_metrics):
    """One subject·channel: per model, 6 in-sample months + 6 forecast months."""
    hist = pd.date_range("2024-07-01", periods=6, freq="MS")
    fut = pd.date_range("2025-01-01", periods=6, freq="MS")
    rows = []
    for model, (rmse, mae, r2) in model_metrics.items():
        for ds in hist:
            rows.append(dict(EntityID="P1", TariffType="LPU", EntityType="POD",
                             consumption_type="PeakConsumption", param_set_id="default",
                             model=model, ds=ds, y=np.nan, y_hat=100.0, y_hat_lower=np.nan,
                             y_hat_upper=np.nan, is_forecast=False, RMSE=rmse, MAE=mae, R2=r2,
                             scenario=None, validation_reason=None))
        for ds in fut:
            rows.append(dict(EntityID="P1", TariffType="LPU", EntityType="POD",
                             consumption_type="PeakConsumption", param_set_id="default",
                             model=model, ds=ds, y=np.nan, y_hat=110.0, y_hat_lower=95.0,
                             y_hat_upper=125.0, is_forecast=True, RMSE=rmse, MAE=mae, R2=r2,
                             scenario=None, validation_reason=None))
    return validate_tidy(pd.DataFrame(rows))


@pytest.fixture
def tidy():
    return _tidy_subject({"ARIMA": (12.5, 9.1, 0.83), "XGBoost": (15.0, 11.0, 0.7)})


# ── Tab 1 — forecast figure ─────────────────────────────────────────────────────


def test_actual_line_drawn_from_passed_actuals(tidy):
    actuals = pd.Series([90, 95, 100, 98, 102, 99],
                        index=pd.date_range("2024-07-01", periods=6, freq="MS"))
    fig = build_forecast_figure(tidy, actuals)
    assert any(t.name == "actual" for t in fig.data)


def test_each_model_has_solid_fit_and_dotted_forecast(tidy):
    fig = build_forecast_figure(tidy)
    arima = [t for t in fig.data if t.legendgroup == "ARIMA"]
    dashes = {(getattr(t.line, "dash", None) or "solid") for t in arima}
    assert "solid" in dashes and "dot" in dashes  # fit solid + forecast dotted


def test_legend_label_carries_real_metrics_not_recomputed(tidy):
    fig = build_forecast_figure(tidy)
    labels = [t.name for t in fig.data if t.showlegend is not False]
    arima = next(l for l in labels if l and l.startswith("ARIMA"))
    assert "RMSE 12.5" in arima and "MAE 9.1" in arima and "R² 0.83" in arima


def test_ci_ribbon_is_toggled_off_by_default_on_when_requested(tidy):
    off = build_forecast_figure(tidy, show_ci=False)
    on = build_forecast_figure(tidy, show_ci=True)
    assert not any(t.fill == "tonexty" for t in off.data)
    assert any(t.fill == "tonexty" for t in on.data)


def test_failed_fit_is_skipped_and_flagged_not_drawn_as_zero():
    # ARIMA scored; FAILED carries NaN metrics (the zero-fallback) → must not be
    # drawn as a confident flat-zero line, and must be flagged instead.
    rows = []
    for ds in pd.date_range("2025-01-01", periods=4, freq="MS"):
        rows.append(dict(EntityID="P1", TariffType="LPU", EntityType="POD",
                         consumption_type="PeakConsumption", param_set_id="default",
                         model="ARIMA", ds=ds, y=np.nan, y_hat=100.0, y_hat_lower=np.nan,
                         y_hat_upper=np.nan, is_forecast=True, RMSE=10.0, MAE=8.0, R2=0.5,
                         scenario=None, validation_reason=None))
        rows.append(dict(EntityID="P1", TariffType="LPU", EntityType="POD",
                         consumption_type="PeakConsumption", param_set_id="default",
                         model="FAILED", ds=ds, y=np.nan, y_hat=0.0, y_hat_lower=np.nan,
                         y_hat_upper=np.nan, is_forecast=True, RMSE=np.nan, MAE=np.nan, R2=np.nan,
                         scenario=None, validation_reason=None))
    fig = build_forecast_figure(validate_tidy(pd.DataFrame(rows)))
    drawn = {t.legendgroup for t in fig.data if t.legendgroup}
    assert "ARIMA" in drawn and "FAILED" not in drawn
    notes = " ".join(a.text for a in fig.layout.annotations)
    assert "FAILED" in notes and "fit failed" in notes


# ── Tab 2 — metrics figure ───────────────────────────────────────────────────────


def test_metrics_figure_median_bar_plus_entity_dots():
    # two entities, two models → median bar + per-entity dots (no box for small n).
    rows = []
    for eid, rmse in [("P1", 10.0), ("P2", 14.0)]:
        for model in ["ARIMA", "XGBoost"]:
            rows.append(dict(EntityID=eid, TariffType="LPU", EntityType="POD",
                             consumption_type="PeakConsumption", param_set_id="default",
                             model=model, ds=pd.Timestamp("2025-01-01"), y=np.nan, y_hat=1.0,
                             y_hat_lower=np.nan, y_hat_upper=np.nan, is_forecast=True,
                             RMSE=rmse + (0 if model == "ARIMA" else 3), MAE=1.0, R2=0.5,
                             scenario=None, validation_reason=None))
    fig = build_metrics_figure(pd.DataFrame(rows), metric="RMSE")
    types = {t.type for t in fig.data}
    assert "bar" in types and "scatter" in types and "box" not in types
    # the bar's median for ARIMA is (10+10)/... actually one entity per (P1,P2): median of [10,10]=10
    bar = next(t for t in fig.data if t.type == "bar")
    assert "ARIMA" in list(bar.x) and "XGBoost" in list(bar.x)


def test_metrics_unknown_metric_raises(tidy):
    with pytest.raises(ValueError, match="unknown metric"):
        build_metrics_figure(tidy, metric="MASE")  # dropped on purpose


# ── entity-picker scenarios (real validator) ─────────────────────────────────────


def test_entity_scenarios_use_real_validator():
    # a healthy long series vs a too-short one → the real validator's verdicts.
    months = pd.date_range("2022-01-01", periods=24, freq="MS")
    good = pd.DataFrame({"EntityID": "GOOD", "TariffType": "LPU", "EntityType": "POD",
                         "ReportingMonth": months, "PeakConsumption": 200.0})
    short = pd.DataFrame({"EntityID": "SHORT", "TariffType": "LPU", "EntityType": "POD",
                          "ReportingMonth": months[:8], "PeakConsumption": 200.0})
    marks = entity_scenarios(pd.concat([good, short], ignore_index=True), method="ARIMA")
    by = dict(zip(marks["EntityID"], marks["scenario"]))
    assert by["SHORT"] == "short_history"
    assert by["GOOD"] in {"happy_path", "negative_values"}  # 24 valid months
