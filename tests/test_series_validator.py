"""Tests for validation.series.validate_series.

Parametrised over the 7 cases documented in
unbundled-modelling/plans/10-data-quality-preflight.md. Each case asserts both
the ok flag AND the exact reason string, so the documented message templates
stay locked down — not just pass/fail.

Run from the project root:
    pytest tests/test_series_validator.py -v
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.performance import PredictionUnit
from validation.series import validate_series

# ── builders ────────────────────────────────────────────────────────────────


def _contiguous(start: str, n: int) -> list:
    """n consecutive month-start timestamps from `start`."""
    return list(pd.date_range(start, periods=n, freq="MS"))


def _make_unit(dates: list, values: list) -> PredictionUnit:
    """A PredictionUnit whose series is indexed by ReportingMonth."""
    series = pd.DataFrame(
        {
            "PodID": ["E001"] * len(dates),
            "CustomerID": ["C001"] * len(dates),
            "TotalConsumption": values,
        },
        index=pd.DatetimeIndex(dates, name="ReportingMonth"),
    )
    return PredictionUnit("E001", "POD", "LPU", "C001", 1, series)


# Reusable date specs. Gap cases keep enough rows to clear the minimum-length
# check first, so the gap check is what actually fires.
_LARGE_GAP = _contiguous("2024-01-01", 12) + _contiguous("2025-06-01", 7)   # 5-month hole
_SMALL_GAP = _contiguous("2024-01-01", 12) + _contiguous("2025-03-01", 4)   # 2-month hole
_NEGATIVE_VALUES = [100.0] * 18
_NEGATIVE_VALUES[3] = -5.0

# ── cases: (method, unit, expected_ok, expected_reason) ──────────────────────

CASES = [
    (
        "arima_too_short",
        "ARIMA",
        _make_unit(_contiguous("2024-01-01", 6), [100.0] * 6),
        False,
        "series too short: 6 months, need 18 for ARIMA",
    ),
    (
        "xgb_too_short",
        "XGBoost",
        _make_unit(_contiguous("2024-01-01", 6), [100.0] * 6),
        False,
        "series too short: 6 months, need 12 for XGBoost",
    ),
    (
        "large_gap_fails",
        "XGBoost",
        _make_unit(_LARGE_GAP, [100.0] * len(_LARGE_GAP)),
        False,
        "gap of 5 months at 2025-01",
    ),
    (
        "small_gap_passes_with_warning",
        "XGBoost",
        _make_unit(_SMALL_GAP, [100.0] * len(_SMALL_GAP)),
        True,
        "gaps ≤ 3 months — proceed with gap_handling",
    ),
    (
        "all_zero_fails",
        "ARIMA",
        _make_unit(_contiguous("2024-01-01", 18), [0.0] * 18),
        False,
        "all-zero series",
    ),
    (
        "negative_passes_with_warning",
        "ARIMA",
        _make_unit(_contiguous("2024-01-01", 18), _NEGATIVE_VALUES),
        True,
        "negative values present — verify credit handling",
    ),
    (
        "clean_series_passes",
        "ARIMA",
        _make_unit(_contiguous("2024-01-01", 18), [100.0] * 18),
        True,
        "ok",
    ),
]


@pytest.mark.parametrize(
    "method,unit,expected_ok,expected_reason",
    [(method, unit, ok, reason) for _, method, unit, ok, reason in CASES],
    ids=[case_id for case_id, *_ in CASES],
)
def test_validate_series(method, unit, expected_ok, expected_reason):
    ok, reason = validate_series(unit, method)
    assert ok is expected_ok
    assert reason == expected_reason
