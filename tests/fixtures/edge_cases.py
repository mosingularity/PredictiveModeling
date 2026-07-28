"""Edge-case fixtures for the unbundled (PodID-keyed) forecasting path.

Two layers live here, sharing one contract:

1. **Legacy scenario builders** (:func:`build_short_series_rows` and friends) —
   the four single-entity row builders the edge-case test module (``tests/
   test_fixture_edge_cases.py``) has always imported. They used to sit in
   ``unbundled-modelling/scripts/build_fixture.py``, a gitignored scratch
   directory that was never in the repo, so the test module skipped itself on
   every machine. They now live here, rekeyed on ``PodID`` (each row carries a
   ``PodID`` natively, so the loop no longer has to mirror ``EntityID`` into it),
   and still carry ``EntityID``/``Scenario`` so the existing assertions hold.

2. **The shared three-member fixture** (:data:`CASES`, :func:`build_case_frame`,
   the :func:`edge_case` pytest fixture) — one fixture on the
   ``Results_<UFMID>.csv`` contract (PodID · CustomerID · TariffType · TariffID ·
   ReportingMonth · Peak/Standard/OffPeak) with a named case per edge condition.
   Each case is a three-member set: two healthy members plus one member
   (:data:`AFFECTED_MEMBER`) carrying the condition, so every case can be
   compared against its own healthy siblings in the same frame. Plan 13's bundled
   path imports this same fixture rather than building its own, so the two paths
   can never disagree about what a case even is.

Everything is deterministic — no wall-clock, fixed RNG seed — so a case's rows
are byte-stable across runs and machines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from db.queries import ForecastConfig

# ── contract columns ──────────────────────────────────────────────────────────
# The three canonical consumption channels of dbo.PredictiveInputData (the
# PodID contract). validation.series keys off exactly these.
CONSUMPTION_COLUMNS = ["PeakConsumption", "StandardConsumption", "OffPeakConsumption"]

# The Results_<UFMID>.csv column order (Results_99.csv form: no UFMID/BundledInd).
RESULTS_COLUMNS = [
    "PodID", "CustomerID", "TariffType", "TariffID", "ReportingMonth",
    *CONSUMPTION_COLUMNS,
]


# ══════════════════════════════════════════════════════════════════════════════
# Layer 1 — legacy single-entity scenario builders
#
# Each returns one entity's rows with EntityID (legacy key), PodID (current key,
# mirrored so the loop's groupby routes it), Scenario, TariffType and the three
# consumption channels. Deterministic values.
# ══════════════════════════════════════════════════════════════════════════════

def _months(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="MS")


def _scenario_frame(
    entity_id: str,
    scenario: str,
    tariff_type: str,
    months: pd.DatetimeIndex,
    peak: np.ndarray,
    standard: np.ndarray | None = None,
    offpeak: np.ndarray | None = None,
    customer_id: str = "SYNTH_EDGE",
    tariff_id: str = "0",
) -> pd.DataFrame:
    """Assemble one entity's rows on the edge-case contract.

    ``PodID`` is the live routing key and is set to ``entity_id`` directly, so
    the loop's ``groupby("PodID")`` finds it without the module having to mirror
    ``EntityID`` after the fact. ``EntityID``/``Scenario`` remain for the
    assertions that predate the PodID contract.
    """
    n = len(months)
    standard = peak * 2.0 if standard is None else standard
    offpeak = peak * 1.5 if offpeak is None else offpeak
    return pd.DataFrame({
        "EntityID": entity_id,
        "PodID": entity_id,
        "Scenario": scenario,
        "TariffType": tariff_type,
        "CustomerID": customer_id,
        "TariffID": tariff_id,
        "ReportingMonth": months,
        "PeakConsumption": np.asarray(peak, dtype=float),
        "StandardConsumption": np.asarray(standard, dtype=float),
        "OffPeakConsumption": np.asarray(offpeak, dtype=float),
    })


def build_short_series_rows() -> pd.DataFrame:
    """6 months (2024-07 → 2024-12) — too short for any method to fit."""
    months = _months("2024-07-01", 6)
    peak = 190.0 + 10.0 * np.arange(6)
    return _scenario_frame("POD_SHORT_SERIES", "short_series", "LPU", months, peak)


def build_gapped_series_rows() -> pd.DataFrame:
    """20 months over a 2022-01 → 2023-12 span with a 4-month hole
    (2022-10 → 2023-01) — a gap larger than 3 months, a hard stop."""
    full = _months("2022-01-01", 24)
    gap = pd.date_range("2022-10-01", "2023-01-01", freq="MS")
    months = full[~full.isin(gap)]
    peak = 200.0 + 5.0 * np.arange(len(months))
    return _scenario_frame("POD_GAPPED_SERIES", "gapped_series", "LPU", months, peak)


def build_all_zero_rows() -> pd.DataFrame:
    """24 all-zero months (2022-01 → 2023-12) — rejected before any fit."""
    months = _months("2022-01-01", 24)
    zeros = np.zeros(24)
    return _scenario_frame("COMBO_ZERO_CONSUMPTION", "all_zero", "SPU", months,
                           zeros, standard=zeros.copy(), offpeak=zeros.copy())


def build_outlier_spike_rows() -> pd.DataFrame:
    """36 months (2022-01 → 2024-12), flat at 200 with one 9999 spike at
    index 20 — passes every pre-flight check and reaches the fit."""
    months = _months("2022-01-01", 36)
    peak = np.full(36, 200.0)
    peak[20] = 9999.0
    return _scenario_frame("POD_OUTLIER_SPIKE", "outlier_spike", "LPU", months, peak)


def build_edge_fixture() -> pd.DataFrame:
    """All four legacy edge entities in one frame (the in-memory loop fixture)."""
    df = pd.concat(
        [build_short_series_rows(), build_gapped_series_rows(),
         build_all_zero_rows(), build_outlier_spike_rows()],
        ignore_index=True,
    )
    df["TariffID"] = df["TariffID"].astype(str)
    df["CustomerID"] = df["CustomerID"].astype(str)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Layer 2 — the shared three-member fixture (Results_<UFMID>.csv contract)
# ══════════════════════════════════════════════════════════════════════════════

# The forecast window every case is judged against: 2025-01 → 2025-06 (6 months).
# Healthy members end 2024-12 so they sit one month before this start and forecast.
DEFAULT_START = pd.Timestamp("2025-01-01")
DEFAULT_END = pd.Timestamp("2025-06-01")

# Base healthy history: 24 months, 2023-01 → 2024-12.
BASE_START = "2023-01-01"
BASE_MONTHS = 24

# The member each case mutates; the other two stay healthy for comparison.
AFFECTED_MEMBER = "POD_ALPHA"

# (PodID, TariffType, CustomerID, TariffID, base level) for the three members.
_MEMBERS = [
    ("POD_ALPHA", "LPU", "CUST_A", "TAR_L", 300.0),
    ("POD_BETA", "SPU", "CUST_B", "TAR_S", 500.0),
    ("POD_GAMMA", "PPU", "CUST_G", "TAR_P", 220.0),
]

# One line per case: what the frame contains and what the unbundled path should do
# with the affected member. Keyed by case name; CASE_NAMES preserves order.
CASES: dict[str, str] = {
    "happy":
        "three healthy 24-month members — all forecast, none dropped",
    "gapped_one_month":
        "one member missing a single interior month — gap ≤ 3, proceeds with gap handling",
    "gapped_six_months":
        "one member with a six-month interior hole — gap > 3, rejected at validation "
        "(same case seen as the loop's skip and as the validator's verdict)",
    "too_short":
        "one member with only 6 months — below the fit minimum, refused",
    "all_zero":
        "one member zero for every month — rejected before any fit is attempted",
    "duplicate_month":
        "one member with two rows for the same month — the path keeps the first, drops the duplicate",
    "appears_mid_history":
        "one member that joins mid-way (18 recent months) — a new connection that still forecasts",
    "disappears_mid_history":
        "one member whose rows stop a year before the others — churn, caught as forecast gap too large",
    "negative_months":
        "one member with credit months (negative) — a warning only, still forecast",
    "outlier_spike":
        "one member flat except a single month 20x larger — forecast, spike carried into the fit",
    "horizon_longer_than_history":
        "healthy members but the horizon (36 months) exceeds the 24-month history",
    "mixed_tariffs":
        "three healthy members carrying LPU, SPU and PPU — no tariff type dropped or special-cased",
}
CASE_NAMES = list(CASES)


def _healthy_series(level: float, months: pd.DatetimeIndex, seed: int) -> np.ndarray:
    """A positive, seasonal, mildly-trending series — enough variation to fit."""
    rng = np.random.default_rng(seed)
    n = len(months)
    trend = np.linspace(0, level * 0.15, n)
    season = level * 0.12 * np.sin(np.arange(n) / 12 * 2 * np.pi)
    noise = rng.normal(0, level * 0.02, n)
    return np.clip(level + trend + season + noise, 1.0, None)


def _member_frame(
    pod_id: str, tariff_type: str, customer_id: str, tariff_id: str,
    months: pd.DatetimeIndex, peak: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame({
        "PodID": pod_id,
        "CustomerID": customer_id,
        "TariffType": tariff_type,
        "TariffID": tariff_id,
        "ReportingMonth": months,
        "PeakConsumption": np.asarray(peak, dtype=float),
        "StandardConsumption": np.asarray(peak, dtype=float) * 1.8,
        "OffPeakConsumption": np.asarray(peak, dtype=float) * 1.3,
    })


def _healthy_member(idx: int) -> pd.DataFrame:
    pod_id, tariff, cust, tar, level = _MEMBERS[idx]
    months = _months(BASE_START, BASE_MONTHS)
    peak = _healthy_series(level, months, seed=100 + idx)
    return _member_frame(pod_id, tariff, cust, tar, months, peak)


def _affected_member(case: str) -> pd.DataFrame:
    """The POD_ALPHA member, mutated to carry the case's condition."""
    pod_id, tariff, cust, tar, level = _MEMBERS[0]
    full = _months(BASE_START, BASE_MONTHS)          # 2023-01 → 2024-12
    peak = _healthy_series(level, full, seed=100)

    if case in ("happy", "mixed_tariffs", "horizon_longer_than_history"):
        return _member_frame(pod_id, tariff, cust, tar, full, peak)

    if case == "gapped_one_month":
        drop = pd.date_range("2023-07-01", periods=1, freq="MS")
        keep = ~full.isin(drop)
        return _member_frame(pod_id, tariff, cust, tar, full[keep], peak[keep])

    if case == "gapped_six_months":
        drop = pd.date_range("2023-06-01", periods=6, freq="MS")
        keep = ~full.isin(drop)
        return _member_frame(pod_id, tariff, cust, tar, full[keep], peak[keep])

    if case == "too_short":
        months = _months("2024-07-01", 6)
        return _member_frame(pod_id, tariff, cust, tar, months,
                             _healthy_series(level, months, seed=100))

    if case == "all_zero":
        return _member_frame(pod_id, tariff, cust, tar, full, np.zeros(len(full)))

    if case == "duplicate_month":
        frame = _member_frame(pod_id, tariff, cust, tar, full, peak)
        # A second row for one month, same value: a true duplicate the path must resolve.
        dup = frame[frame["ReportingMonth"] == pd.Timestamp("2024-03-01")]
        return pd.concat([frame, dup], ignore_index=True)

    if case == "appears_mid_history":
        # Starts well after the others but still carries ≥18 months to the present
        # (2023-07 → 2024-12): a late-joining connection that clears validation and
        # forecasts, rather than one so new it is refused for being too short.
        months = _months("2023-07-01", 18)
        return _member_frame(pod_id, tariff, cust, tar, months,
                             _healthy_series(level, months, seed=100))

    if case == "disappears_mid_history":
        # 24 months ending 2023-12 — long enough to clear validation, but a full
        # year before the forecast window, so it is caught at the forecast-gap
        # stage (ForecastGapTooLarge), not the length stage. That is the churn
        # signal, distinct from a merely-short series.
        months = _months("2022-01-01", 24)
        return _member_frame(pod_id, tariff, cust, tar, months,
                             _healthy_series(level, months, seed=100))

    if case == "negative_months":
        peak = peak.copy()
        peak[[8, 16]] = [-40.0, -25.0]              # two credit months
        return _member_frame(pod_id, tariff, cust, tar, full, peak)

    if case == "outlier_spike":
        peak = np.full(len(full), 200.0)
        peak[20] = 4000.0                            # one month 20x the rest
        return _member_frame(pod_id, tariff, cust, tar, full, peak)

    raise ValueError(f"unknown case: {case!r}")


def build_case_frame(case: str) -> pd.DataFrame:
    """A three-member frame on the Results contract for the named ``case``.

    Member ``POD_ALPHA`` carries the condition; ``POD_BETA`` and ``POD_GAMMA``
    stay healthy so each case has its own in-frame baseline. Rows are returned
    sorted by (PodID, ReportingMonth).
    """
    if case not in CASES:
        raise ValueError(f"unknown case {case!r}; known: {CASE_NAMES}")
    members = [_affected_member(case), _healthy_member(1), _healthy_member(2)]
    df = pd.concat(members, ignore_index=True)[RESULTS_COLUMNS]
    return df.sort_values(["PodID", "ReportingMonth"]).reset_index(drop=True)


# ── shared forecast configs (single source of truth for tests + evidence) ──────

def default_ufm_config() -> ForecastConfig:
    """Forecast window 2025-01 → 2025-06 (6 months)."""
    return ForecastConfig(
        forecast_method_id=1,
        forecast_method_name="ARIMA",
        model_parameters="",
        region="TEST",
        status="Active",
        user_forecast_method_id=99,
        start_date=DEFAULT_START,
        end_date=DEFAULT_END,
        databrick_task_id=0,
    )


def long_horizon_ufm_config() -> ForecastConfig:
    """A 36-month window (2025-01 → 2027-12) — longer than the 24-month history,
    for the horizon-longer-than-history case."""
    cfg = default_ufm_config()
    cfg.end_date = pd.Timestamp("2027-12-01")
    return cfg


# ── pytest fixture (imported by both this path's tests and plan 13's) ──────────

try:
    import pytest

    @pytest.fixture(params=CASE_NAMES)
    def edge_case(request):
        """Yield ``(case_name, three_member_frame)`` once per named case.

        Imported by both the unbundled tests here and plan 13's bundled tests, so
        neither path builds its own version of a case.
        """
        name = request.param
        return name, build_case_frame(name)
except ImportError:  # pytest absent (e.g. the evidence script imports the builders only)
    pass
