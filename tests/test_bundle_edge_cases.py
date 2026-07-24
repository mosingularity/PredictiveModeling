"""Bundled edge cases across bundling, prediction and disaggregation (plan 13).

The bundle path is well tested for its maths and thin at its edges. This module runs
plan 12's shared three-member fixture (``tests/fixtures/edge_cases.py``) through the
bundled path and asserts what happens at each stage: bundling (membership and
aggregation), prediction (validation and fit) and disaggregation (shares and split),
plus the output contract.

The point of a bundle is that individual members disappear behind an aggregate — which
is also what makes a bad member invisible. Several cases therefore only mean something
next to the unbundled answer, so where the two paths treat one case differently that
divergence is asserted here deliberately (see ``test_gapped_member_divergence*``) rather
than discovered in production later.

A recurring structural fact drives many of these: ``build_member_series`` reindexes every
member onto one continuous monthly range and zero-fills it, so by the time the aggregate
reaches ``validate_series`` it can never carry a *gap* — a member's hole has already
become zero months. That is exactly why the gapped-member and aggregate-gap cases behave
the opposite way to the unbundled path.

The four disaggregation cases blocked on EFP2-698 (near-zero / negative bundle totals)
are xfailed, not skipped: the data is built and ready, only the guard rule is a pending
decision, so ``pytest -rs`` still reports no skips.

Run from the project root:
    pytest tests/test_bundle_edge_cases.py -v
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.performance import PredictionUnit
from models.algorithms.bundled import run_bundled
from models.algorithms.autoarima import forecast_arima_unbundled
from models.bundle import (
    BUNDLE_CONSUMPTION_COLUMN, aggregate_members, build_member_series,
    compute_shares, forecast_for_bundle,
)
from tests.fixtures.edge_cases import (
    AFFECTED_MEMBER, CONSUMPTION_COLUMNS, build_case_frame,
)
from validation.series import validate_series

METHOD = "ARIMA"          # 18-month minimum; the fixture's 24-month members clear it


def _last_value_fit(series, horizon):
    """A trivial, deterministic fit_fn: repeat the last observed aggregate value.

    The real fits are covered by test_bundle_fit.py; these edge tests exercise the
    aggregation / validation / share / split logic AROUND the fit, so a fixed fit keeps
    them fast and keeps the assertions about the surrounding logic, not the estimator.
    """
    idx = pd.date_range(series.index[-1], periods=horizon + 1, freq="MS")[1:]
    return pd.Series(float(series.iloc[-1]), index=idx)


def _wide(frame, channel="PeakConsumption"):
    """The wide member frame the bundle path builds from a long case frame."""
    return build_member_series(frame, channel)


def _alpha_unit(frame, channel="PeakConsumption"):
    """A PredictionUnit for the affected member alone — the unbundled path's view of it."""
    alpha = frame[frame["PodID"] == AFFECTED_MEMBER].set_index("ReportingMonth")
    alpha.index = pd.to_datetime(alpha.index)
    return PredictionUnit(entity_id=AFFECTED_MEMBER, entity_type="", tariff_type="LPU",
                          customer_id="CUST_A", tariff_id="TAR_L", series=alpha)


def _drop_months(frame, months):
    """Remove the given months from EVERY member — an aggregate-level condition built
    from the shared members (a bundle-only shape the shared fixture, being member-level,
    does not carry; the members themselves are unchanged)."""
    return frame[~pd.to_datetime(frame["ReportingMonth"]).isin(months)].copy()


def _set_member(frame, pod_id, months, value):
    """Overwrite one member's consumption at the given months (or all if months is None)."""
    frame = frame.copy()
    mask = frame["PodID"] == pod_id
    if months is not None:
        mask &= pd.to_datetime(frame["ReportingMonth"]).isin(months)
    frame.loc[mask, CONSUMPTION_COLUMNS] = value
    return frame


# ══════════════════════════════════════════════════════════════════════════════
# Bundling stage — membership and aggregation
# ══════════════════════════════════════════════════════════════════════════════

class TestBundlingStage:

    def test_happy_three_members_all_present(self):
        wide = _wide(build_case_frame("happy"))
        assert list(wide.columns) == ["POD_ALPHA", "POD_BETA", "POD_GAMMA"]
        parts = forecast_for_bundle(wide, method=METHOD, horizon=6, fit_fn=_last_value_fit)
        assert parts.shape[1] == 3 and not parts.isna().any().any()

    def test_duplicate_member_month_is_summed(self):
        """Two rows for one member-month sum into one value — documented in the
        build_member_series docstring, asserted here for the first time (plan 13 B4)."""
        dup = _wide(build_case_frame("duplicate_month"))
        single = _wide(build_case_frame("happy"))
        m = pd.Timestamp("2024-03-01")
        assert dup.loc[m, "POD_ALPHA"] == pytest.approx(2 * single.loc[m, "POD_ALPHA"]), \
            "the duplicate month must be summed, not deduplicated to one row"

    def test_member_appears_mid_history_backfilled_with_zeros(self):
        """A late-joining member is zero-filled before it existed and its share is
        depressed by those zeros — carried, never dropped (plan 13 B5)."""
        wide = _wide(build_case_frame("appears_mid_history"))
        assert wide.loc[pd.Timestamp("2023-01-01"), "POD_ALPHA"] == 0.0   # pre-existence
        assert AFFECTED_MEMBER in wide.columns
        assert compute_shares(wide)[AFFECTED_MEMBER] < \
            compute_shares(_wide(build_case_frame("happy")))[AFFECTED_MEMBER]

    def test_member_disappears_mid_history_still_forecast(self):
        """A member that went quiet a year ago still receives a share and a forecast, and
        nothing in the run flags that it stopped reporting (plan 13 B6)."""
        wide = _wide(build_case_frame("disappears_mid_history"))
        assert wide.loc[pd.Timestamp("2024-12-01"), "POD_ALPHA"] == 0.0   # recent silence
        parts = forecast_for_bundle(wide, method=METHOD, horizon=6, fit_fn=_last_value_fit)
        assert AFFECTED_MEMBER in parts.columns
        assert compute_shares(wide)[AFFECTED_MEMBER] > 0     # still a positive share


def test_gapped_member_divergence_bundle_keeps_unbundled_rejects():
    """THE path divergence, in one test that names both behaviours (plan 13, `-k gapped`).

    Same member, a six-month interior hole, both ways:
      * unbundled — the per-entity validator rejects it as a gap > 3 months, so the
        member produces no output at all;
      * bundled — build_member_series zero-fills the hole into the aggregate, the member
        is forecast, and its share is permanently depressed by the six zeros it now
        carries.

    If either side changes silently this test fails, which is the whole point: the
    divergence must be a decision, not an accident.
    """
    frame = build_case_frame("gapped_six_months")

    # Unbundled view: the affected member is rejected outright.
    ok, reason = validate_series(_alpha_unit(frame), METHOD)
    assert ok is False and reason.startswith("gap of"), \
        f"unbundled path must reject the gapped member; got ({ok}, {reason!r})"

    # Bundled view: the same member is kept, forecast, and its share is depressed.
    wide = _wide(frame)
    parts = forecast_for_bundle(wide, method=METHOD, horizon=6, fit_fn=_last_value_fit)
    assert AFFECTED_MEMBER in parts.columns, "bundled path must keep the gapped member"
    gapped_share = compute_shares(wide)[AFFECTED_MEMBER]
    healthy_share = compute_shares(_wide(build_case_frame("happy")))[AFFECTED_MEMBER]
    assert gapped_share < healthy_share, \
        "the six zero-fills must depress the member's share below its healthy share"


# ══════════════════════════════════════════════════════════════════════════════
# Prediction stage — validation and fit
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictionStage:

    def test_aggregate_gap_is_zero_filled_not_rejected(self):
        """Plan 13 B14. Every member missing the SAME six months does NOT reject on the
        bundled path: build_member_series reindexes onto a continuous range and zero-fills
        it, so the aggregate reaches validation gap-free. The opposite of the unbundled
        path, where that same hole is a hard stop. Asserted so the divergence is on record.
        """
        hole = pd.date_range("2023-06-01", periods=6, freq="MS")
        wide = _wide(_drop_months(build_case_frame("happy"), hole))
        agg = aggregate_members(wide)[BUNDLE_CONSUMPTION_COLUMN]
        full = pd.date_range(agg.index.min(), agg.index.max(), freq="MS")
        assert agg.index.equals(full), "aggregate index must be continuous (zero-filled)"
        assert (agg.loc[hole] == 0).all(), "the hole must be zero, not missing"
        # Forecasts rather than raising — the gap never reaches the gap check.
        parts = forecast_for_bundle(wide, method=METHOD, horizon=6, fit_fn=_last_value_fit)
        assert parts.shape[1] == 3

    def test_aggregate_negative_months_warn_not_block(self):
        """Plan 13 B15. An aggregate that goes negative in a few months (credits) is a
        warning, not a block — the run forecasts."""
        # A credit big enough to push those three months' aggregate below zero, but small
        # enough that the bundle's annual total stays positive (so this is the warn case,
        # not the EFP2-698 non-positive-total rejection).
        neg = pd.to_datetime(["2023-04-01", "2023-09-01", "2024-02-01"])
        frame = _set_member(build_case_frame("happy"), AFFECTED_MEMBER, neg, -3000.0)
        wide = _wide(frame)
        agg = aggregate_members(wide)
        assert (agg[BUNDLE_CONSUMPTION_COLUMN] < 0).sum() == 3
        assert agg[BUNDLE_CONSUMPTION_COLUMN].sum() > 0        # annual total stays positive
        unit = PredictionUnit("BUNDLE", "Bundle", "", "", 0, agg)
        ok, reason = validate_series(unit, METHOD, cons_cols=[BUNDLE_CONSUMPTION_COLUMN])
        assert ok is True and "negative" in reason
        parts = forecast_for_bundle(wide, method=METHOD, horizon=6, fit_fn=_last_value_fit)
        assert parts.shape[1] == 3

    def test_outlier_spike_inflates_member_share(self):
        """Plan 13 B16. A single 20x month in one member lifts THAT member's historical
        share (shares are history-total based), so the spike moves the bundle split."""
        spiked = compute_shares(_wide(build_case_frame("outlier_spike")))[AFFECTED_MEMBER]
        clean = compute_shares(_wide(build_case_frame("happy")))[AFFECTED_MEMBER]
        assert spiked > clean, "the outlier month must inflate the member's share"

    def test_horizon_longer_than_history_runs(self):
        """Plan 13 B17. A 36-month horizon from 24 months of history runs and produces a
        full-length forecast per member (it does not refuse or truncate)."""
        wide = _wide(build_case_frame("horizon_longer_than_history"))
        assert len(wide) == 24
        parts = forecast_for_bundle(wide, method=METHOD, horizon=36, fit_fn=_last_value_fit)
        assert parts.shape == (36, 3) and not parts.isna().any().any()


# ══════════════════════════════════════════════════════════════════════════════
# Disaggregation stage — shares and split
# ══════════════════════════════════════════════════════════════════════════════

class TestDisaggregationStage:

    def test_all_zero_member_gets_zero_share_and_flat_forecast(self):
        """An all-zero member takes a 0.0 share and a flat-zero forecast while the others
        forecast normally, and coherence still holds (members sum to the bundle)."""
        wide = _wide(build_case_frame("all_zero"))
        shares = compute_shares(wide)
        assert shares[AFFECTED_MEMBER] == 0.0
        assert (shares.drop(AFFECTED_MEMBER) > 0).all()
        parts = forecast_for_bundle(wide, method=METHOD, horizon=6, fit_fn=_last_value_fit)
        assert (parts[AFFECTED_MEMBER] == 0).all()
        bundle_fc = _last_value_fit(aggregate_members(wide)[BUNDLE_CONSUMPTION_COLUMN], 6)
        assert np.allclose(parts.sum(axis=1).to_numpy(), bundle_fc.to_numpy())

    def test_credit_member_stays_negative_in_healthy_bundle(self):
        """The requirement behind EFP2-698: a credit (negative) member inside a healthy
        positive bundle keeps a negative share AND a negative forecast — the credit stays a
        credit — while members still sum to the bundle. This is the common mixed-sign case,
        and it is forecast, not rejected (its share magnitude is well under the cap)."""
        frame = _set_member(build_case_frame("happy"), AFFECTED_MEMBER, None, -50.0)
        wide = _wide(frame)
        totals = wide.sum(axis=0)
        assert totals[AFFECTED_MEMBER] < 0 < totals.sum()      # member <0, bundle >0
        shares = compute_shares(wide)
        assert shares[AFFECTED_MEMBER] < 0                     # credit share stays negative
        parts = forecast_for_bundle(wide, method=METHOD, horizon=6, fit_fn=_last_value_fit)
        assert (parts[AFFECTED_MEMBER] < 0).all(), "the credit member's forecast must stay negative"
        bundle_fc = _last_value_fit(aggregate_members(wide)[BUNDLE_CONSUMPTION_COLUMN], 6)
        assert np.allclose(parts.sum(axis=1).to_numpy(), bundle_fc.to_numpy()), \
            "members still sum to the bundle (coherent)"

    def test_near_cancel_mixed_sign_bundle_is_rejected(self):
        """EFP2-698, rule A: when positive and credit members nearly cancel, a member's share
        magnitude blows up (its forecast would be an amplified, sign-flipped multiple of the
        aggregate), so the bundle is rejected rather than disaggregated."""
        frame = _set_member(build_case_frame("happy"), "POD_BETA", None, -1000.0)
        frame = _set_member(frame, "POD_GAMMA", None, 0.0)
        frame = _set_member(frame, AFFECTED_MEMBER, None, 1000.0 + 1.0 / 24)  # tiny surplus
        wide = _wide(frame)
        totals = wide.sum(axis=0)
        assert 0 < totals.sum() < 100                          # tiny positive total
        assert totals.abs().max() / totals.sum() > 5           # a member's share blows past the cap
        with pytest.raises(ValueError, match="cannot be disaggregated"):
            forecast_for_bundle(wide, method=METHOD, horizon=6, fit_fn=_last_value_fit)

    def test_negative_bundle_total_is_rejected(self):
        """EFP2-698, rule A: a bundle whose history sums negative is rejected rather than
        silently producing a coherent-but-sign-flipped forecast the coherence check misses."""
        frame = _set_member(build_case_frame("happy"), AFFECTED_MEMBER, None, -50000.0)
        wide = _wide(frame)
        assert wide.sum(axis=0).sum() < 0                      # negative grand total
        with pytest.raises(ValueError, match="not positive"):
            forecast_for_bundle(wide, method=METHOD, horizon=6, fit_fn=_last_value_fit)


# ══════════════════════════════════════════════════════════════════════════════
# Output stage — the reporting surface matches across modes
# ══════════════════════════════════════════════════════════════════════════════

def _model_for(csv, method=METHOD):
    """A ForecastModel stand-in over a written case CSV (both paths read it via
    PREDICTIVE_FIXTURE_PATH), mirroring test_bundle_coherence's stub."""
    import types
    fx = pd.read_csv(csv, encoding="utf-8-sig")
    mx = pd.to_datetime(fx["ReportingMonth"]).max()
    cfg = types.SimpleNamespace(
        forecast_method_name=method, model_parameters="", user_forecast_method_id=99,
        start_date=mx + pd.DateOffset(months=1), end_date=mx + pd.DateOffset(months=6))
    return types.SimpleNamespace(
        dataset=types.SimpleNamespace(ufm_config=cfg),
        config=types.SimpleNamespace(log=False))


class TestOutputStage:

    @pytest.fixture
    def both_surfaces(self, tmp_path, monkeypatch):
        """Run both modes over the SAME happy frame; return their reporting surfaces
        (get_performance_data — the surface that carries TariffID, unlike to_forecast_fact)."""
        csv = tmp_path / "case_happy.csv"
        build_case_frame("happy").to_csv(csv, index=False)
        monkeypatch.setenv("PREDICTIVE_FIXTURE_PATH", str(csv))
        unbundled = forecast_arima_unbundled(_model_for(csv), spark=None).get_performance_data()
        bundled = run_bundled(_model_for(csv), spark=None).get_performance_data()
        return unbundled, bundled

    def test_reporting_surface_dtypes_match(self, both_surfaces):
        """Plan 13 B27. The reporting surface (not only the stored surface) matches across
        modes — in particular TariffID is the same dtype both ways, not text one way and
        an integer the other."""
        unbundled, bundled = both_surfaces
        assert bundled["TariffID"].dtype == unbundled["TariffID"].dtype
        for col in ("PodID", "customer_id", "TariffType", "TariffID"):
            assert bundled[col].dtype == unbundled[col].dtype, f"dtype drift on {col}"

    def test_each_bundled_row_carries_its_pod_identity(self, both_surfaces):
        """Plan 13 B28. Each bundled row carries ITS pod's customer and tariff — not a
        blank customer and a zeroed tariff — identical to the unbundled row for the same pod."""
        unbundled, bundled = both_surfaces
        key = ["PodID", "customer_id", "TariffType", "TariffID"]
        ub = unbundled[key].drop_duplicates().set_index("PodID").sort_index()
        bd = bundled[key].drop_duplicates().set_index("PodID").sort_index()
        assert (bd["customer_id"] != "").all(), "bundled customer must not be blank"
        assert (bd["TariffID"] != 0).all(), "bundled tariff must not be zeroed"
        pd.testing.assert_frame_equal(ub, bd)
