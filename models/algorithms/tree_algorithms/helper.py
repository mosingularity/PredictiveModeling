import logging
from types import SimpleNamespace
from typing import Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.seasonal import STL
from xgboost import XGBRegressor

from data.dml import get_forecast_range
from evaluation.performance import PodIDPerformanceData
from hyperparameters import get_model_hyperparameters
from models.algorithms.helper import _collect_metrics
from models.algorithms.utilities import evaluate_predictions, process_reporting_months
from validation.input_checks import invalid_forecast_horizon, invalid_length, invalid_series
from validation.metadata import get_error_metadata

logger = logging.getLogger(__name__)


def engineer_data(df: pd.DataFrame, cons: str, lags: list, windows: list):
    n = len(df)
    valid_lags = [lag for lag in lags if lag <= n - 1]
    valid_windows = [w for w in windows if w <= n - 1]
    if len(valid_lags) < len(lags) or len(valid_windows) < len(windows):
        logger.debug("Pruned lag/window features for n=%s", n)

    df, stl_obj = stl_decompose(df, cons, period=12)
    df = engineer_calendar(df)
    df = engineer_lags(df, "deseasoned", valid_lags)
    df = engineer_rolling(df, "deseasoned", valid_windows)
    df = engineer_interactions(df, valid_lags)
    full_history = df["deseasoned"].copy()
    df, feature_cols = assemble_features(df, valid_lags, valid_windows)
    return df, feature_cols, stl_obj, full_history

def stl_decompose(df: pd.DataFrame, target: str, period: int = 12):
    y = df[target].astype(float)
    use_stl = len(y) >= 24  # need >=2 cycles for stable monthly STL

    if use_stl:
        stl = STL(y, period=period, robust=True).fit()
        seasonal = stl.seasonal
        deseasoned = y - seasonal
        # collapse guard: STL sometimes overfits when data are short or quirky
        deseasoned_spread = deseasoned.std(ddof=1)
        if deseasoned_spread < 1e-6 or deseasoned_spread < 0.05 * y.std(ddof=1):
            seasonal = pd.Series(0.0, index=y.index)
            deseasoned = y.copy()
            stl = SimpleNamespace(seasonal=seasonal)
    else:
        seasonal = pd.Series(0.0, index=y.index)
        deseasoned = y.copy()
        stl = SimpleNamespace(seasonal=seasonal)

    out = df.copy()
    out["seasonal"] = seasonal
    out["deseasoned"] = deseasoned
    return out, stl


def engineer_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical month and binary flags.
    """
    df["month"]     = df.index.month
    df["month_sin"] = np.sin(2 * np.pi * df.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.month / 12)
    df["is_feb"]    = (df.month == 2).astype(int)
    return df


def engineer_lags(df: pd.DataFrame, base_series: str, lags: list) -> pd.DataFrame:
    """
    Create lag features for the deseasoned series.
    """
    for lag in lags:
        df[f"ds_lag{lag}"] = df[base_series].shift(lag)
    return df


def engineer_rolling(df: pd.DataFrame, base_series: str, windows: list) -> pd.DataFrame:
    """
    Create rolling-mean and -std features.
    """
    for w in windows:
        df[f"ds_roll_mean_{w}m"] = df[base_series].rolling(w).mean().shift(1)
        df[f"ds_roll_std_{w}m"]  = df[base_series].rolling(w).std().shift(1)
    return df


def engineer_interactions(df: pd.DataFrame,  lags: list[int]) -> pd.DataFrame:
    year_lags = [lag for lag in lags if lag % 12 == 0]
    for lag in year_lags:
        col = f"ds_lag{lag}"
        if col in df and df[col].notna().any():
            df[f"{col}_x_sin"] = df[col] * df["month_sin"]
            df[f"{col}_x_cos"] = df[col] * df["month_cos"]
    return df



def assemble_features(
    df: pd.DataFrame,
    lags: list[int],
    windows: list[int]
) -> tuple[pd.DataFrame, list[str]]:
    # 1) lag columns
    lag_cols = [f"ds_lag{lag}" for lag in lags]

    # 2) rolling statistics
    roll_cols = []
    for w in windows:
        roll_cols += [f"ds_roll_mean_{w}m", f"ds_roll_std_{w}m"]

    # 3) calendar features
    calendar_cols = ["month_sin", "month_cos", "is_feb"]

    # 4) dynamic yearly-lag interactions
    yearly_lags = [lag for lag in lags if lag % 12 == 0]
    interaction_cols = [f"ds_lag{lag}_x_sin" for lag in yearly_lags]

    # 5) assemble everything
    feature_cols = lag_cols + roll_cols + calendar_cols + interaction_cols
    feature_cols = [c for c in feature_cols if c in df.columns and df[c].notna().any()]
    # 6) drop any rows missing *any* of these features or the target
    df_clean = df.dropna(subset=feature_cols + ["deseasoned"])

    return df_clean, feature_cols



def split_train_test(df: pd.DataFrame, test_months: int = 3) -> (pd.DataFrame, pd.DataFrame):
    """
    Split df into train and test based on last N months.
    """
    hist_end = df.index.max()
    train = df.loc[: hist_end - pd.offsets.MonthBegin(test_months)]
    test  = df.loc[ hist_end - pd.offsets.MonthBegin(test_months-1):]
    return train, test


def train_rf(X: pd.DataFrame, y: pd.Series, rf_params_tuple) -> RandomForestRegressor:
    """
    Fit RandomForestRegressor with given parameters.
    """
    (
        n_estimators,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features,
        bootstrap_flag
    ) = rf_params_tuple

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth if max_depth > 0 else None,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap_flag,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y)
    return rf

def train_xgb(X: pd.DataFrame, y: pd.Series, xgb_params_tuple) -> XGBRegressor:
    """
    Fit XGBRegressor with given parameters.
    """
    (n_estimators, max_depth, learning_rate, subsample, colsample_bytree) = xgb_params_tuple

    xgb = XGBRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth) if int(max_depth) > 0 else None,
        learning_rate=float(learning_rate),
        subsample=float(subsample),
        colsample_bytree=float(colsample_bytree),
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror"
    )
    xgb.fit(X, y)
    return xgb


def recursive_forecast(
    history_ds: pd.Series,
    stl_obj: STL,
    rf: Union[RandomForestRegressor, XGBRegressor],
    features: list,
    start: pd.Timestamp,
    end: pd.Timestamp,
    lags: list,
    windows: list
) -> pd.DataFrame:
    """
    Generate multi-step forecasts by feeding back predictions.
    """
    # 1) build map of average seasonal component by calendar month
    seasonal_months = stl_obj.seasonal.index.month
    season_map = stl_obj.seasonal.groupby(seasonal_months).mean()

    # 2) keep a running deseasoned series for recursive lags
    history_ds = history_ds.copy()

    # 3) precompute which of your lags are "yearly" (multiples of 12)
    yearly_lags = [lag for lag in lags if lag % 12 == 0]

    # 4) forecast index
    forecast_index = pd.date_range(start=start, end=end, freq="MS")
    results = []
    for date in forecast_index:
        m = date.month
        seasonal = season_map.get(m, 0.0)
        # construct features
        row = {}
        for lag in lags:
            lag_date = date - pd.DateOffset(months=lag)
            if lag_date in history_ds.index:
                val = history_ds.loc[lag_date]
                row[f"ds_lag{lag}"] = float(val.iloc[0]) if isinstance(val, pd.Series) else float(val)
            else:
                row[f"ds_lag{lag}"] = np.nan
        for w in windows:
            window = history_ds.loc[(date - pd.DateOffset(months=w)) : (date - pd.DateOffset(months=1))]
            row[f"ds_roll_mean_{w}m"] = window.mean()
            row[f"ds_roll_std_{w}m"]  = window.std()
        row.update({
            "month_sin": np.sin(2*np.pi*m/12),
            "month_cos": np.cos(2*np.pi*m/12),
            "is_feb":   int(m==2),
        })
        for lag in yearly_lags:
            row[f"ds_lag{lag}_x_sin"] = row[f"ds_lag{lag}"] * np.sin(2 * np.pi * m / 12)
        feature_row = pd.DataFrame([row], index=[date])
        X_pred = feature_row[features]
        ds_pred = rf.predict(X_pred)[0]
        history_ds.loc[date] = ds_pred
        if not history_ds.index.is_monotonic_increasing:
            history_ds = history_ds.sort_index()
        results.append((date, ds_pred + seasonal))
    forecast_frame = pd.DataFrame(results, columns=["date", "forecast"])
    return forecast_frame.set_index("date")


def forecast_pod_with_tree(
    df: pd.DataFrame,
    customer_id: str,
    pod_id: str,
    consumption_types: list,
    ufm_config,
    *,
    method_key: str,
    train_fn,
    report_validation_error,
    base_lags: list = [1, 2, 3, 6],
    base_windows: list = [3, 6],
    test_months: int = 3,
) -> PodIDPerformanceData:
    """Forecast one pod with a tree model: engineer features, fit, backtest, and
    forecast the horizon, per consumption type.

    Shared by RandomForest and XGBoost, which differ only in ``method_key`` (the
    hyperparameter lookup key) and ``train_fn`` (:func:`train_rf` / :func:`train_xgb`).
    ``report_validation_error`` is passed in rather than imported here so each caller's
    own module-level reference stays the one tests patch (``patch.object(rf, ...)`` /
    ``patch.object(xgb, ...)``).
    """
    data = []
    forecast_horizon = get_forecast_range(ufm_config)
    start_fc = ufm_config.start_date
    end_fc = ufm_config.end_date

    for consumption_type in consumption_types:
        pod_df = df[df["PodID"] == pod_id].sort_index()
        pod_df = process_reporting_months(pod_df)

        if invalid_series(pod_id, pod_df[consumption_type], consumption_type):
            forecast = pd.Series([0] * len(forecast_horizon), index=forecast_horizon)
            row = _collect_metrics(pod_id, customer_id, consumption_type, forecast,
                                   validation_reason="all-zero / flat series")
            data.append(row)
            continue

        if invalid_length(pod_df[consumption_type], consumption_type):
            forecast = pd.Series([0] * len(forecast_horizon), index=forecast_horizon)
            row = _collect_metrics(pod_id, customer_id, consumption_type, forecast,
                                   validation_reason="series too short")
            data.append(row)
            continue

        if invalid_forecast_horizon(pod_id, pod_df[consumption_type], consumption_type, forecast_horizon):
            forecast = pd.Series([0] * len(forecast_horizon), index=forecast_horizon)
            row = _collect_metrics(pod_id, customer_id, consumption_type, forecast,
                                   validation_reason="gap too large")
            data.append(row)
            continue

        forecast_horizon_months = ((end_fc.year - pod_df.index.max().year) * 12 +
                                   (end_fc.month - pod_df.index.max().month))
        year_lags = [12 * i for i in range(1, forecast_horizon_months // 12 + 1)]
        lags = base_lags + year_lags

        year_windows = [12 * i for i in range(1, forecast_horizon_months // 12 + 1)]
        windows = base_windows + year_windows

        pod_df, feature_cols, stl_obj, history_series = engineer_data(pod_df, consumption_type, lags, windows)

        train_df, test_df = split_train_test(pod_df, test_months)
        if train_df.empty or test_df.empty:
            logger.warning(
                f"🚫 Not enough data after split for {consumption_type} @ Pod {pod_id}. Skipping."
            )
            forecast = pd.Series([0] * len(forecast_horizon), index=forecast_horizon)
            row = _collect_metrics(pod_id, customer_id, consumption_type, forecast,
                                   validation_reason="insufficient data after split")
            data.append(row)
            continue
        X_train, y_train = train_df[feature_cols], train_df["deseasoned"]
        X_test = test_df[feature_cols]

        params_tuple = get_model_hyperparameters(method_key, ufm_config.model_parameters)
        try:
            model = train_fn(X_train, y_train, params_tuple)
        except Exception as model_fit_exception:
            meta = get_error_metadata("ModelFitFailure", {"exception": str(model_fit_exception)})
            report_validation_error(log_id=None, error=meta["message"], traceback="", error_type="ModelFitFailure",
                                   severity=meta["severity"], component=meta["component"],
                                   entity_id=pod_id)
            forecast = pd.Series([0] * len(forecast_horizon), index=forecast_horizon)
            row = _collect_metrics(pod_id, customer_id, consumption_type, forecast,
                                   validation_reason="model fit failed")
            data.append(row)
            continue

        train_pred_ds = model.predict(X_train)
        test_pred_ds = model.predict(X_test)

        # re-add seasonality
        train_pred = train_pred_ds + train_df["seasonal"]
        test_pred = test_pred_ds + test_df["seasonal"]

        y_test_original = test_df["deseasoned"] + test_df["seasonal"]
        metrics, baseline_metrics = evaluate_predictions(y_test_original, test_pred)

        # Forecast from a model refit on ALL data (train+test) so the future uses
        # the most recent months; the train-only model above is kept solely for
        # the held-out backtest metrics. Mirrors the ARIMA full-data forecast.
        full_model = train_fn(pod_df[feature_cols], pod_df["deseasoned"], params_tuple)
        fc_df = recursive_forecast(history_series, stl_obj, full_model,
                                   feature_cols, start_fc, end_fc,
                                   lags, windows)
        future_forecast = fc_df['forecast']
        # Full-history fitted line (train fit + held-out test) for the solid
        # "predicted historical" overlay; metrics stay on the held-out test above.
        in_sample_fit = pd.concat([train_pred, test_pred]).sort_index()
        row = _collect_metrics(
            pod_id, customer_id, consumption_type,
            future_forecast, metrics, baseline_metrics, in_sample=in_sample_fit
        )
        data.append(row)
    performance_frame = pd.DataFrame(data)
    return PodIDPerformanceData(
        pod_id=pod_id,
        forecast_method_name=ufm_config.forecast_method_name,
        customer_id=customer_id,
        user_forecast_method_id=ufm_config.user_forecast_method_id,
        performance_data_frame=performance_frame
    )
