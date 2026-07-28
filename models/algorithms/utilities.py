import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def prepare_time_series_data(
    df: pd.DataFrame,
    consumption_type: str,
    strategy: str = "drop",
) -> pd.Series:
    """Convert one consumption column into a monthly time series, resolving
    duplicate timestamps by ``strategy`` (sum / mean / drop)."""
    series = pd.to_numeric(df[consumption_type], errors='coerce')
    series.index = pd.to_datetime(series.index)

    # Handle duplicate timestamps
    if series.index.duplicated().any():
        logging.info("🔁 Duplicate timestamps detected in time index.")
        if strategy == "sum":
            series = series.groupby(series.index).sum()
        elif strategy == "mean":
            series = series.groupby(series.index).mean()
        elif strategy == "drop":
            series = series[~series.index.duplicated(keep='first')]
        else:
            raise ValueError(f"Unknown duplicate handling strategy: {strategy}")

    # Resample to monthly frequency
    series = series.asfreq('MS').fillna(0)
    return series


def process_reporting_months(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy().reset_index()
    consumption_cols = [col for col in df.columns if "Consumption" in col]
    base_keys = [k for k in ('ReportingMonth', 'PodID', 'CustomerID') if k in df.columns]
    subset_keys = base_keys + consumption_cols
    duplicates = df.duplicated(subset=subset_keys, keep=False)
    if duplicates.any():
        logging.info(f"🔁 Found {duplicates.sum()} duplicate rows based on: {subset_keys}")
        df = df.drop_duplicates(subset=subset_keys, keep='first')

    pod_period_keys = ['ReportingMonth', 'PodID']
    if df.duplicated(subset=pod_period_keys).any():
        n_conflict = df.duplicated(subset=pod_period_keys, keep=False).sum()
        logging.warning(
            f"⚠️ {n_conflict} rows share (ReportingMonth, PodID) with conflicting values "
            f"(multiple customers or zero-placeholder rows) — keeping the row with the "
            f"highest total consumption per period."
        )
        df['_total'] = df[consumption_cols].sum(axis=1)
        df = (df.sort_values('_total', ascending=False)
                .drop_duplicates(subset=pod_period_keys, keep='first')
                .drop(columns=['_total']))

    df.set_index('ReportingMonth', inplace=True)
    df.sort_index(inplace=True)
    return df


def mean_forecast(series: pd.Series, n_periods: int) -> pd.Series:
    """Flat forecast: the training series' mean, repeated. The baseline every
    real forecast must beat."""
    return np.full(n_periods, series.mean())


def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """RMSE, MAE, and R2 of a forecast against actuals."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> Tuple[Dict[str, float], Dict[str, float]]:
    """(metrics, baseline_metrics) for a forecast against actuals."""
    metrics = evaluate_forecast(y_true, y_pred)
    actuals = pd.Series(y_true)
    baseline = mean_forecast(actuals, len(actuals))
    baseline_metrics = evaluate_forecast(y_true, baseline)
    return metrics, baseline_metrics
