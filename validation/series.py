"""Pre-fit validation for a prediction unit's consumption series.

A series must clear four checks before a forecasting method is fitted to it.
The first three are hard stops (too short, large gaps, all zeros); negative
values are a warning only, since utility credits are legitimate.

Each check returns the first noteworthy status it finds, in order — so a
series that reaches the end having tripped nothing returns (True, "ok").
"""
import pandas as pd

from evaluation.performance import PredictionUnit

# Minimum observed months required before fitting, by method family.
ARIMA_MIN = 18
TREE_MIN = 12
ARIMA_METHODS = {"ARIMA", "SARIMA"}

# Identifier/metadata columns carried alongside the consumption values. Covers
# both the bundled (PodID/CustomerID) and unbundled (EntityID/TariffType/…/
# Scenario) column sets — everything here is excluded before the numeric checks
# so non-consumption strings never reach the all-zero/negative sums.
_NON_CONSUMPTION_COLUMNS = [
    "PodID", "CustomerID", "ReportingMonth",
    "EntityID", "TariffType", "TariffID", "EntityType", "Scenario",
]

# Canonical consumption-value columns (allowlist). The single source of truth for
# the wide-frame builders in the forecasting engine and the dashboard, kept here
# beside the validator so the two never drift out of sync.
CONSUMPTION_COLUMNS = [
    "PeakConsumption", "StandardConsumption", "OffPeakConsumption",
    "Block1Consumption", "Block2Consumption", "Block3Consumption",
    "Block4Consumption", "NonTOUConsumption",
]


def consumption_columns(series: pd.DataFrame) -> list:
    """Consumption value columns — everything that isn't id/metadata.

    The single source of truth for "which columns hold consumption values",
    shared by the validator and the forecasting algorithms so the bundled and
    unbundled column sets stay in sync.
    """
    return list(series.columns.difference(_NON_CONSUMPTION_COLUMNS))


def _reporting_months(series: pd.DataFrame) -> pd.PeriodIndex:
    """Monthly periods for the series, deduplicated and sorted ascending.

    ReportingMonth is normally the DatetimeIndex, but fall back to a column
    of the same name if a caller hands over a column-indexed frame.
    """
    if "ReportingMonth" in series.columns:
        values = pd.to_datetime(series["ReportingMonth"])
    else:
        values = pd.to_datetime(series.index)
    return pd.PeriodIndex(values, freq="M").drop_duplicates().sort_values()


def validate_series(unit: PredictionUnit, method: str) -> tuple[bool, str]:
    """Validate a unit's series for the given forecasting method.

    Returns (ok, reason). ``ok`` is False for hard stops (too short, gap > 3
    months, all-zero) and True otherwise; ``reason`` always carries a message,
    including the informational/warning cases that still return True.
    """
    series = unit.series
    cons_cols = consumption_columns(series)

    # Check 1 — minimum length, enforced before any fit.
    n = len(series)
    min_n = ARIMA_MIN if method in ARIMA_METHODS else TREE_MIN
    if n < min_n:
        return (False, f"series too short: {n} months, need {min_n} for {method}")

    # Check 2 — a gap larger than 3 months is a hard stop.
    months = _reporting_months(series)
    for prev, curr in zip(months[:-1], months[1:]):
        gap = (curr - prev).n - 1  # consecutive missing months between the two
        if gap > 3:
            gap_start = prev + 1
            return (False, f"gap of {gap} months at {gap_start}")
    has_small_gaps = bool(len(months)) and (months[-1] - months[0]).n + 1 > len(months)

    # Check 3 — all-zero series (hard stop). Must run before the small-gap pass
    # below: an all-zero series that also has a minor gap would otherwise return
    # early on the gap and never be rejected for being all-zero.
    if series[cons_cols].to_numpy().sum() == 0:
        return (False, "all-zero series")

    # Check 4 — minor gaps are an informational pass, once the hard stops clear.
    if has_small_gaps:
        return (True, "gaps ≤ 3 months — proceed with gap_handling")

    # Check 5 — negative values (warning only — credits are legitimate).
    if (series[cons_cols].to_numpy() < 0).any():
        return (True, "negative values present — verify credit handling")

    return (True, "ok")


def classify_reason(reason: str) -> str:
    """Map a :func:`validate_series` reason to a stable scenario slug.

    Co-located with the reason strings it matches so the dashboard's entity
    picker keys off this one function instead of re-parsing the prose in another
    module — the two can no longer drift apart across files.
    """
    if reason.startswith("series too short"):
        return "short_history"
    if reason.startswith("gaps ≤ 3 months"):
        return "gap_within_limit"
    if reason.startswith("gap of"):
        return "gap_too_large"
    if reason == "all-zero series":
        return "all_zero"
    if reason.startswith("negative values"):
        return "negative_values"
    return "happy_path"
