"""Bundle aggregate forecast path.

A "bundle" is a set of member entities forecast together: their series are
summed into one aggregate series, a single model is fit on that aggregate, and
the aggregate forecast is split back to members by each member's historical
share. Disaggregation is coherent — member forecasts always sum to the bundle
forecast.

Unlike the unbundled path, a bundle cannot be *partially* disaggregated: if the
aggregate series fails validation there is no per-member fallback, so a failed
``validate_series`` is fatal and raises ``ValueError``.

The model fit is pluggable via ``fit_fn`` so callers can supply the real
estimator; the default is a naive-seasonal forecaster that keeps this path
runnable and testable on its own.
"""
from typing import Callable, Optional

import numpy as np
import pandas as pd

from evaluation.performance import PredictionUnit
from models.series_validator import validate_series

# Single consumption column used to represent an aggregated bundle series.
BUNDLE_CONSUMPTION_COLUMN = "TotalConsumption"


def aggregate_members(member_series: pd.DataFrame) -> pd.DataFrame:
    """Sum member columns into a single-column bundle series.

    ``member_series`` is indexed by ReportingMonth with one column per member.
    Returns a DataFrame with the same index and one consumption column.
    """
    bundle = member_series.sum(axis=1)
    return bundle.to_frame(name=BUNDLE_CONSUMPTION_COLUMN)


def compute_shares(member_series: pd.DataFrame) -> pd.Series:
    """Each member's share of the bundle total over history (sums to 1)."""
    totals = member_series.sum(axis=0)
    grand_total = totals.sum()
    if grand_total == 0:
        # Guarded earlier by validate_series ("all-zero series"); kept here so
        # compute_shares is safe to call in isolation.
        raise ValueError("all-zero series")
    return totals / grand_total


def _future_index(bundle_series: pd.Series, horizon: int) -> pd.DatetimeIndex:
    """``horizon`` month-start timestamps following the last observed month."""
    last = bundle_series.index[-1]
    return pd.date_range(last, periods=horizon + 1, freq="MS")[1:]


def _naive_seasonal_forecast(bundle_series: pd.Series, horizon: int) -> pd.Series:
    """Repeat the last 12-month pattern; fall back to last value if shorter."""
    values = bundle_series.to_numpy()
    if len(values) >= 12:
        season = values[-12:]
        repeats = horizon // 12 + 1
        forecast = np.tile(season, repeats)[:horizon]
    else:
        forecast = np.full(horizon, values[-1])
    return pd.Series(forecast, index=_future_index(bundle_series, horizon), name="bundle_fc")


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
    ok, reason = validate_series(bundle_unit, method)
    if not ok:
        raise ValueError(reason)

    # Fit one model on the aggregate, then disaggregate by member share.
    forecaster = fit_fn or _naive_seasonal_forecast
    bundle_forecast = forecaster(bundle_df[BUNDLE_CONSUMPTION_COLUMN], horizon)
    shares = compute_shares(member_series)
    return pd.DataFrame(
        np.outer(bundle_forecast.to_numpy(), shares.to_numpy()),
        index=bundle_forecast.index,
        columns=shares.index,
    )
