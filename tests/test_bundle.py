"""Tests for models.bundle.forecast_for_bundle.

Focus is the 10.3 deliverable: validate_series runs on the *aggregated* series
before fitting, and an ok=False result raises ValueError(reason) — a bundle
cannot be partially disaggregated. Also checks that a valid bundle disaggregates
coherently (member forecasts sum to the bundle forecast).

Run from the project root:
    pytest tests/test_bundle.py -v
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bundle import forecast_for_bundle

# ── fixtures ──────────────────────────────────────────────────────────────────


def _members(n_months: int, member_values: dict) -> pd.DataFrame:
    """Member series indexed by ReportingMonth, one column per member."""
    dates = pd.date_range("2024-01-01", periods=n_months, freq="MS")
    return pd.DataFrame(
        member_values,
        index=pd.DatetimeIndex(dates, name="ReportingMonth"),
    )


# ── validation guard (the 10.3 deliverable) ──────────────────────────────────


def test_too_short_bundle_raises_valueerror():
    # 6 months < ARIMA min of 18 → aggregated series fails length check.
    members = _members(6, {"E1": [100.0] * 6, "E2": [50.0] * 6})
    with pytest.raises(ValueError, match="series too short: 6 months, need 18 for ARIMA"):
        forecast_for_bundle(members, "ARIMA", horizon=3)


def test_all_zero_bundle_raises_valueerror():
    # Members are individually zero → aggregate is all-zero → fatal.
    members = _members(18, {"E1": [0.0] * 18, "E2": [0.0] * 18})
    with pytest.raises(ValueError, match="all-zero series"):
        forecast_for_bundle(members, "ARIMA", horizon=3)


def test_members_offset_but_aggregate_nonzero_is_valid():
    # Each member has zeros in some months, but the *aggregate* is never zero —
    # validation runs on the bundle, not the members, so this is accepted.
    e1 = [100.0] * 18
    e2 = [0.0] * 18
    members = _members(18, {"E1": e1, "E2": e2})
    result = forecast_for_bundle(members, "ARIMA", horizon=3)
    assert not result.empty


# ── coherent disaggregation on a valid bundle ────────────────────────────────


def test_valid_bundle_disaggregates_coherently():
    rng = np.random.default_rng(0)
    members = _members(
        24,
        {
            "E1": rng.uniform(80, 120, 24),
            "E2": rng.uniform(30, 60, 24),
            "E3": rng.uniform(10, 20, 24),
        },
    )
    result = forecast_for_bundle(members, "XGBoost", horizon=3)

    assert list(result.columns) == ["E1", "E2", "E3"]
    assert len(result) == 3

    # Member forecasts sum back to the bundle forecast at every horizon step.
    bundle_total = members.sum(axis=1).to_numpy()[-12:]  # naive-seasonal source
    expected_bundle_fc = bundle_total[:3]
    np.testing.assert_allclose(result.sum(axis=1).to_numpy(), expected_bundle_fc, rtol=1e-9)


def test_fit_fn_is_used_when_supplied():
    members = _members(18, {"E1": [100.0] * 18, "E2": [100.0] * 18})

    def flat_fit(bundle_series, horizon):
        idx = pd.date_range(bundle_series.index[-1], periods=horizon + 1, freq="MS")[1:]
        return pd.Series([1000.0] * horizon, index=idx, name="bundle_fc")

    result = forecast_for_bundle(members, "XGBoost", horizon=2, fit_fn=flat_fit)
    # Equal members → equal 50/50 split of the supplied flat forecast.
    np.testing.assert_allclose(result.to_numpy(), np.full((2, 2), 500.0), rtol=1e-9)
