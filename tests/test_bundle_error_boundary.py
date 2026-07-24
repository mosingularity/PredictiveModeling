"""Error-boundary tests for the bundle run path (plan 05).

A bundle whose aggregate fails validation, or whose fit raises, must be contained:
logged with its member ids and skipped, so the run continues to the next bundle and the
output holds only the successful ones. Only ``Exception`` is caught, so
``KeyboardInterrupt`` / ``SystemExit`` propagate.
"""
import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.algorithms.bundled import forecast_bundle_guarded
from validation.run_summary import RunSummary


def _members(cols, n=24, value=100.0):
    """A wide member frame: n monthly-start rows, one column per member id."""
    idx = pd.date_range("2023-01-01", periods=n, freq="MS")
    return pd.DataFrame({c: np.full(n, float(value)) for c in cols}, index=idx)


def _last_value_fit(series, horizon):
    """A trivial fit_fn: repeat the last observed value (fast, deterministic)."""
    idx = pd.date_range(series.index[-1], periods=horizon + 1, freq="MS")[1:]
    return pd.Series(float(series.iloc[-1]), index=idx)


def test_failed_aggregate_validation_is_logged_with_members():
    # all-zero aggregate → validate_series fails → ValueError → contained
    all_zero = _members(["E1", "E2", "E3"], value=0.0)
    with patch("models.algorithms.bundled.report_validation_error") as mock_log:
        result = forecast_bundle_guarded(all_zero, "SARIMA", 6, fit_fn=_last_value_fit)
    assert result is None
    args = str(mock_log.call_args)
    assert "E1" in args and "E2" in args and "E3" in args, \
        "the log must name every member lost with this bundle"


def test_failed_bundle_does_not_abort_run():
    summary = RunSummary("bundled SARIMA")
    bad = _members(["E1", "E2", "E3"], value=0.0)    # all-zero → fails
    good = _members(["P1", "P2"], value=500.0)       # succeeds
    forecasts = []
    for members, label in [(bad, "Peak"), (good, "Standard")]:
        parts = forecast_bundle_guarded(members, "SARIMA", 6, fit_fn=_last_value_fit,
                                        summary=summary, label=label)
        if parts is not None:
            forecasts.append(parts)
    assert len(forecasts) == 1                        # only the good bundle survived
    assert list(forecasts[0].columns) == ["P1", "P2"]
    assert summary.ok == 1 and sum(summary.skipped.values()) == 1


def test_all_bundles_succeed_nothing_logged():
    summary = RunSummary("bundled SARIMA")
    with patch("models.algorithms.bundled.report_validation_error") as mock_log:
        for members, label in [(_members(["P1", "P2"], value=300.0), "Peak"),
                               (_members(["P1", "P2"], value=400.0), "Standard")]:
            parts = forecast_bundle_guarded(members, "SARIMA", 6, fit_fn=_last_value_fit,
                                            summary=summary, label=label)
            assert parts is not None
    mock_log.assert_not_called()
    assert summary.ok == 2 and sum(summary.skipped.values()) == 0


def test_keyboard_interrupt_propagates():
    good = _members(["P1", "P2"], value=500.0)        # passes validation → reaches fit_fn

    def _interrupt(series, horizon):
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        forecast_bundle_guarded(good, "SARIMA", 6, fit_fn=_interrupt)
