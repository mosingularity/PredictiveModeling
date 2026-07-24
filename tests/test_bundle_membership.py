"""Membership adapter tests: build_member_series (plan 02).

The bundle's members ARE the UFM's pod set (the PodID contract from PredictiveInputData).
build_member_series pivots the long loader frame into the wide member frame
forecast_for_bundle expects — one column per PodID, indexed by ReportingMonth at 'MS'.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bundle import build_member_series, compute_shares, forecast_for_bundle


def _long(pods, months=24, value=100.0, start="2023-01-01"):
    """A long PodID-contract frame: one row per (pod, month)."""
    idx = pd.date_range(start, periods=months, freq="MS")
    return pd.DataFrame([{"PodID": p, "ReportingMonth": m, "PeakConsumption": value}
                         for p in pods for m in idx])


def test_pivots_to_wide_one_column_per_pod():
    wide = build_member_series(_long(["POD_A", "POD_B", "POD_C"]), "PeakConsumption")
    assert list(wide.columns) == ["POD_A", "POD_B", "POD_C"]   # one column per PodID
    assert "EntityID" not in wide.columns                      # no EntityID on the contract
    assert wide.index.name == "ReportingMonth"
    assert wide.index.freqstr == "MS"


def test_missing_month_zero_filled_and_shared_index():
    df = _long(["POD_A", "POD_B"], months=24)
    drop = pd.Timestamp("2024-06-01")
    df = df[~((df["PodID"] == "POD_B") & (df["ReportingMonth"] == drop))]
    wide = build_member_series(df, "PeakConsumption")
    assert wide.isna().sum().sum() == 0            # zero-filled, no NaN
    assert wide.loc[drop, "POD_B"] == 0.0          # the absent month is zero
    assert len(wide) == 24                         # one continuous shared index preserved


def test_single_member_bundle_is_identity():
    wide = build_member_series(_long(["ONLY"], value=500.0), "PeakConsumption")
    assert wide.shape[1] == 1
    assert abs(compute_shares(wide).iloc[0] - 1.0) < 1e-12    # share is 1.0

    def _last_value_fit(series, horizon):
        future = pd.date_range(series.index[-1], periods=horizon + 1, freq="MS")[1:]
        return pd.Series(float(series.iloc[-1]), index=future)

    parts = forecast_for_bundle(wide, method="SARIMA", horizon=6, fit_fn=_last_value_fit)
    # disaggregation is the identity: the lone member equals the whole bundle forecast
    assert np.allclose(parts.iloc[:, 0].to_numpy(), parts.sum(axis=1).to_numpy())


def test_mixed_lpu_spu_membership():
    # tariff type is irrelevant to membership — columns are just the pods (the normal case)
    wide = build_member_series(_long(["LPU_1", "SPU_1", "SPU_2"]), "PeakConsumption")
    assert set(wide.columns) == {"LPU_1", "SPU_1", "SPU_2"}
    assert wide.shape == (24, 3)


def test_rejects_off_contract_entityid_frame():
    # the legacy EntityID-keyed fixture has no PodID → clear error, not a silent mispivot
    idx = pd.date_range("2023-01-01", periods=24, freq="MS")
    df = pd.DataFrame([{"EntityID": "E1", "ReportingMonth": m, "PeakConsumption": 100.0}
                       for m in idx])
    with pytest.raises(ValueError, match="PodID"):
        build_member_series(df, "PeakConsumption")
