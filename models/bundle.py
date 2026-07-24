"""Bundle aggregate-then-disaggregate forecast path.

A bundle is a set of member entities forecast together: their series are summed
into one aggregate, a single model is fit on that aggregate, and the forecast is
split back to members by each member's historical share. Members always sum back
to the bundle forecast exactly.

One pipeline: :func:`build_member_series` builds the wide member frame (one column
per member), then :func:`forecast_for_bundle` aggregates, fits, and disaggregates
via :func:`aggregate_members` and :func:`compute_shares`. A bundle can't be
partially disaggregated, so a failed validation is fatal (raises ``ValueError``),
unlike the unbundled path's per-entity skip.

The fit is pluggable via ``fit_fn`` (see :func:`models.bundle_forecasters.bundle_forecaster`
for the real estimator). The default here is a naive-seasonal placeholder for
testing this module on its own; production always passes a real ``fit_fn``.
"""
import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd

from evaluation.performance import PredictionUnit
from models.algorithms.utilities import prepare_time_series_data
from validation.series import validate_series

logger = logging.getLogger(__name__)

# Single consumption column used to represent an aggregated bundle series.
BUNDLE_CONSUMPTION_COLUMN = "TotalConsumption"
SHARE_MAGNITUDE_CAP = 5.0


def build_member_series(df: pd.DataFrame, consumption_type: str) -> pd.DataFrame:
    """Pivot a loader frame into the wide member frame ``forecast_for_bundle`` expects:
    one column per PodID (the bundle's members), indexed by ReportingMonth at monthly
    ('MS') frequency, valued by ``consumption_type``.

    Membership is just the pod set the frame contains, no BundleID or membership table.
    A member absent for a month (no row, or a gap) is zero-filled, the safe default for
    an additive aggregate. A one-pod frame degrades to a one-column frame with share 1.0;
    that isn't special-cased, it falls out of the general math.

    Raises ``ValueError`` if ``PodID`` or ``consumption_type`` is missing from the frame.
    """
    if "PodID" not in df.columns:
        raise ValueError(
            "build_member_series requires a PodID column (the PredictiveInputData / "
            "Results_<UFMID>.csv contract). The frame has no PodID — an EntityID-keyed "
            "off-contract fixture cannot be used to build a member series."
        )
    if "ReportingMonth" not in df.columns:
        raise ValueError("build_member_series requires a ReportingMonth column.")
    if consumption_type not in df.columns:
        raise ValueError(
            f"consumption_type {consumption_type!r} is not a column of the frame; "
            f"available consumption columns: "
            f"{[c for c in df.columns if 'Consumption' in c]}"
        )

    members = {}
    for pod_id, group in df.groupby("PodID"):
        member = group.set_index("ReportingMonth").sort_index()
        members[str(pod_id)] = prepare_time_series_data(
            member, consumption_type, strategy="sum")
    wide = pd.DataFrame(members)
    first_month = wide.index.min()
    last_month = wide.index.max()
    full_index = pd.date_range(first_month, last_month, freq="MS")
    wide = wide.reindex(full_index).fillna(0.0)
    wide.index.name = "ReportingMonth"
    return wide


def aggregate_members(member_series: pd.DataFrame) -> pd.DataFrame:
    """Sum member columns into a single-column bundle series.

    ``member_series`` is indexed by ReportingMonth with one column per member.
    Returns a DataFrame with the same index and one consumption column.
    """
    bundle = member_series.sum(axis=1)
    return bundle.to_frame(name=BUNDLE_CONSUMPTION_COLUMN)


def compute_shares(member_series: pd.DataFrame) -> pd.Series:
    """Each member's share of the bundle total over history (sums to 1).

    Raises ``ValueError`` when the total cannot be divided by meaningfully; the rule
    is to reject rather than emit an amplified or sign-flipped split:

    * a genuinely all-zero frame is ``"all-zero series"`` (also guarded earlier by
      ``validate_series``; kept here so ``compute_shares`` is safe to call in isolation);
    * a non-positive total (``<= 0``) — the members net to zero or a credit, so a share
      split is undefined;
    * a share magnitude above :data:`SHARE_MAGNITUDE_CAP` — the positive and credit members
      nearly cancel, so a member's share (and hence its forecast) blows up and can flip
      sign. A healthy bundle, including a dominant member plus a minority credit member
      whose credit stays negative, is well under the cap and passes.
    """
    totals = member_series.sum(axis=0)
    grand_total = totals.sum()
    absolute_values = member_series.abs().to_numpy()
    gross = float(absolute_values.sum())
    if gross == 0:
        raise ValueError("all-zero series")
    if grand_total <= 0:
        raise ValueError(
            f"bundle total not positive: {grand_total:.6g}; the members net to zero or a "
            f"credit, so a share split is undefined")
    shares = totals / grand_total
    absolute_shares = shares.abs()
    max_share = float(absolute_shares.max())
    if max_share > SHARE_MAGNITUDE_CAP:
        raise ValueError(
            f"bundle cannot be disaggregated: a member's share is {max_share:.1f}x the "
            f"bundle (cap {SHARE_MAGNITUDE_CAP:g}); its positive and credit members nearly "
            f"cancel, so the split would amplify the aggregate forecast and can flip signs")
    return shares


def _future_index(bundle_series: pd.Series, horizon: int) -> pd.DatetimeIndex:
    """``horizon`` month-start timestamps following the last observed month."""
    last = bundle_series.index[-1]
    return pd.date_range(last, periods=horizon + 1, freq="MS")[1:]


def _naive_seasonal_forecast(bundle_series: pd.Series, horizon: int) -> pd.Series:
    """Repeat the last 12-month pattern; fall back to last value if shorter.

    In-module test default only. Production passes a real ``fit_fn`` from
    :func:`models.bundle_forecasters.bundle_forecaster`; this placeholder is never wired
    from the pipeline.
    """
    values = bundle_series.to_numpy()
    if len(values) >= 12:
        season = values[-12:]
        repeats = horizon // 12 + 1
        forecast = np.tile(season, repeats)[:horizon]
    else:
        forecast = np.full(horizon, values[-1])
    future = _future_index(bundle_series, horizon)
    return pd.Series(forecast, index=future, name="bundle_fc")


def forecast_for_bundle(
    member_series: pd.DataFrame,
    method: str,
    horizon: int,
    *,
    bundle_id: str = "BUNDLE",
    customer_id: str = "",
    tariff_type: str = "",
    tariff_id: int = 0,
    fit_fn: Optional[Callable[[pd.Series, int], pd.Series]] = None,
) -> pd.DataFrame:
    """Forecast a bundle by aggregate-then-disaggregate.

    Aggregates ``member_series`` into one bundle series, validates that bundle
    series before fitting (failure is fatal — raises ``ValueError(reason)``),
    fits one model on the aggregate, then disaggregates the bundle forecast back
    to members by historical share. Returns a DataFrame of member forecasts
    indexed by forecast month, one column per member, coherent by construction.
    """
    bundle_df = aggregate_members(member_series)

    # Represent the aggregated bundle as a temporary PredictionUnit and validate
    # it BEFORE fitting. A bundle cannot be partially disaggregated, so any
    # validation failure is fatal — there is no per-member fallback.
    bundle_unit = PredictionUnit(
        entity_id=bundle_id,
        entity_type="Bundle",
        tariff_type=tariff_type,
        customer_id=customer_id,
        tariff_id=tariff_id,
        series=bundle_df,
    )
    ok, reason = validate_series(bundle_unit, method,
                                 cons_cols=[BUNDLE_CONSUMPTION_COLUMN])
    if not ok:
        raise ValueError(reason)

    shares = compute_shares(member_series)
    if fit_fn is None:
        logger.warning(
            "⚠️ forecast_for_bundle is on the naive-seasonal PLACEHOLDER — no real "
            "fit_fn was supplied. Production must pass a real fit_fn "
            "(models.bundle_forecasters.bundle_forecaster); the naive path is a test default."
        )
    forecaster = fit_fn or _naive_seasonal_forecast
    bundle_forecast = forecaster(bundle_df[BUNDLE_CONSUMPTION_COLUMN], horizon)
    forecast_values = bundle_forecast.to_numpy()
    share_values = shares.to_numpy()
    member_forecasts = np.outer(forecast_values, share_values)
    return pd.DataFrame(member_forecasts, index=bundle_forecast.index, columns=shares.index)
