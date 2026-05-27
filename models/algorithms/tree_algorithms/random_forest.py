# random_forest.py

"""
Enhanced Random Forest global→local pipeline with dynamic lag/rolling, encoding,
model persistence, and consistent evaluation. Fixed key‐lookup for ID encoding
in future forecast by using precomputed encoding maps.
"""

import logging
import os
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from db.queries import ForecastConfig, insert_profiling_error
from docstring.utilities import profiled_function
from evaluation.performance import ModelPodPerformance
from hyperparameters import get_model_hyperparameters
from models.algorithms.helper import ensure_numeric_consumption_types
from models.base import ForecastModel
from models.forecast_validation import run_forecast_sanity_checks
from profiler.errors.utils import get_error_metadata
from profiler.profiler_switch import profiling_switch
from utils.exit_handler import safe_exit

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

performance_metrics_table = "dbo.StatisticalPerformanceMetrics"
target_table_name = "dbo.ForecastFact"
write_url = "jdbc:sqlserver://fortrack-maz-sdb-san-dev-01.database.windows.net:1433;database=FortrackDB;Authentication=ActiveDirectoryMSI;trustServerCertificate=true"
write_properties = {
    "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver"
}

def frequency_encode(series: pd.Series) -> pd.Series:
    freqs = series.value_counts(normalize=True)
    return series.map(freqs).fillna(0.0)


def compute_time_index(df: pd.DataFrame, date_col: str) -> pd.Series:
    earliest = df[date_col].min()
    return ((df[date_col] - earliest).dt.days // 30).astype(int)


def train_test_split_time(
    df: pd.DataFrame, date_col: str, test_fraction: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    n_total = len(df_sorted)
    test_size = max(int(n_total * test_fraction), 1)
    return df_sorted.iloc[:-test_size], df_sorted.iloc[-test_size:]


def evaluate_regression(
    y_true: pd.Series, y_pred: np.ndarray
) -> Dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred))
    }

def prepare_global_training_data(
    model: ForecastModel
) -> Tuple[pd.DataFrame, List[str], List[str], pd.DataFrame]:
    """
    Function: Perform feature engineering for random forest
    """
    # 1) Get raw data and reset index since ReportingMonth is index
    df_raw = model.dataset.processed_df.copy().reset_index()
    ufm_config: ForecastConfig = model.dataset.ufm_config
    
    df_raw = ensure_numeric_consumption_types(df_raw, model)

    # 2) Determine forecast horizon in months
    end = pd.to_datetime(ufm_config.end_date)
    last_hist = df_raw["ReportingMonth"].max()
    horizon = ((end.year - last_hist.year) * 12 + (end.month - last_hist.month))
    horizon = max(horizon, 0)

    # 3) Build dynamic lag_list and rolling_windows
    base_lags = [1, 2, 3, 6]
    year_lags = [12 * i for i in range(1, horizon // 12 + 1)]
    lag_list = sorted(set(base_lags + year_lags))

    base_rolls = [3, 12]
    multi_rolls = [12 * i for i in range(2, horizon // 12 + 1)]
    rolling_windows = sorted(set(base_rolls + multi_rolls))

    # 4) Compute dynamic min_history
    min_lag = max(lag_list) if lag_list else 0
    min_roll = (max(rolling_windows) + 1) if rolling_windows else 0
    min_history = max(min_lag, min_roll, 1)

    # 5) Attach dynamic config to model.config
    cfg = model.config
    cfg.lag_list = lag_list
    cfg.rolling_windows = rolling_windows
    cfg.min_history = min_history
    cfg.encoder_method = getattr(cfg, "encoder_method", "frequency")
    cfg.test_fraction = getattr(cfg, "test_fraction", 0.2)

    # 6) Identify zero-consumption types
    consumption_types = cfg.consumption_types[:]
    zero_types: List[str] = [c for c in consumption_types if df_raw[c].sum() == 0]
    cfg.zero_consumptions = zero_types
    cfg.consumption_types = [c for c in consumption_types if c not in zero_types]

    # 7) Add time index and seasonal features
    df = df_raw.copy()
    df["TimeIndex"] = compute_time_index(df, "ReportingMonth")
    df["month"] = df["ReportingMonth"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # 8) Dynamic lag features for each non-zero consumption type
    df = df.sort_values(["CustomerID", "PodID", "ReportingMonth"]).reset_index(drop=True)
    for cons in cfg.consumption_types:
        for lag in lag_list:
            df[f"{cons}_lag{lag}"] = (
                df.groupby(["CustomerID", "PodID"])[cons]
                  .shift(lag)
            )

    # 9) Dynamic rolling statistics
    for cons in cfg.consumption_types:
        for window in rolling_windows:
            df[f"{cons}_roll{window}_mean"] = (
                df.groupby(["CustomerID", "PodID"])[cons]
                  .shift(1)
                  .rolling(window=window)
                  .mean()
            )
            df[f"{cons}_roll{window}_std"] = (
                df.groupby(["CustomerID", "PodID"])[cons]
                  .shift(1)
                  .rolling(window=window)
                  .std()
                  .fillna(0.0)
            )

    # 10) Drop rows missing required lags
    required_cols = []
    for cons in cfg.consumption_types:
        required_cols += [f"{cons}_lag{lag}" for lag in lag_list]
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    # 11) Frequency-encode IDs and TariffID if present
    df["CustomerID_enc"] = frequency_encode(df["CustomerID"].astype(str))
    df["PodID_enc"] = frequency_encode(df["PodID"].astype(str))
    id_features = ["CustomerID_enc", "PodID_enc"]
    if "TariffID" in df.columns:
        df["TariffID_enc"] = frequency_encode(df["TariffID"].astype(str))
        id_features.append("TariffID_enc")

    # 12) Build feature column list
    feature_cols = ["TimeIndex", "month_sin", "month_cos"] + id_features
    for cons in cfg.consumption_types:
        feature_cols += [f"{cons}_lag{lag}" for lag in lag_list]
        for window in rolling_windows:
            feature_cols += [f"{cons}_roll{window}_mean", f"{cons}_roll{window}_std"]
    
    # 13) Ensure all feature columns are numeric floats (cast object/Decimal → float)
    df[feature_cols] = (
        df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype(float)
    )
    return df, feature_cols, zero_types, df_raw


def train_global_rf_models(
    df: pd.DataFrame,
    feature_columns: List[str],
    model: ForecastModel
) -> Tuple[Dict[str, RandomForestRegressor], Dict[str, Tuple[pd.DataFrame, pd.Series]]]:
    """
    Function: Train a global random forest model
    """
    ufm_config: ForecastConfig = model.dataset.ufm_config
    cfg = model.config
    global_models: Dict[str, RandomForestRegressor] = {}
    test_sets: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {}

    # Base directory for saving/loading models
    base_dir = os.path.join("model_configuration", "rf", ufm_config.forecast_method_name, "global")
    os.makedirs(base_dir, exist_ok=True)

    for cons in cfg.consumption_types:
        model_path = os.path.join(base_dir, f"{cons}.pkl")

        # If model file exists, load and skip training
        if os.path.isfile(model_path):
            rf = joblib.load(model_path)
            # logger.info(f"Loaded existing model for {cons} from {model_path}")
            # Re-split for X_test, y_test
            train_df, test_df = train_test_split_time(df, "ReportingMonth", cfg.test_fraction)
            X_test, y_test = test_df[feature_columns], test_df[cons]
            test_sets[cons] = (X_test, y_test)
            global_models[cons] = rf
            continue

        y = df[cons]
        if len(y) < cfg.min_history or y.nunique() <= 1:
            meta = get_error_metadata("InsufficientObservations", {"consumption_type": cons, "series_length": len(y)})
            insert_profiling_error(
                log_id=None,
                error=meta["message"],
                traceback="",  # or traceback.format_exc()
                error_type="InsufficientObservations",
                severity=meta["severity"],
                component=meta["component"]
            )
            # logger.info(f"⚠️ Skipping global training for {cons} insufficient data..")
            continue

        train_df, test_df = train_test_split_time(df, "ReportingMonth", cfg.test_fraction)
        X_train, y_train = train_df[feature_columns], train_df[cons]
        X_test, y_test = test_df[feature_columns], test_df[cons]

        # Load hyperparameters from config
        rf_params_tuple = get_model_hyperparameters("randomforest", ufm_config.model_parameters)
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
        rf.fit(X_train, y_train)
        global_models[cons] = rf
        test_sets[cons] = (X_test, y_test)

    return global_models, test_sets


def forecast_locally_with_global_models(
    df_feat: pd.DataFrame,
    feature_columns: List[str],
    global_models: Dict[str, RandomForestRegressor],
    test_sets: Dict[str, Tuple[pd.DataFrame, pd.Series]],
    model: ForecastModel,
    zero_types: List[str],
    df_raw: pd.DataFrame,
    gap_handling: str = "skip"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function: Forecast for every customer-pod pair
    """
    ufm_config: ForecastConfig = model.dataset.ufm_config
    full_forecast_dates = pd.date_range(
        start=pd.to_datetime(ufm_config.start_date),
        end=pd.to_datetime(ufm_config.end_date),
        freq="MS"
    )
    # Build future monthly dates from last historical to end_date
    last_hist = df_raw["ReportingMonth"].max()
    forecast_start = last_hist + pd.DateOffset(months=1)
    forecast_end = pd.to_datetime(ufm_config.end_date)
    forecast_dates = pd.date_range(start=forecast_start, end=forecast_end, freq="MS")

    cfg = model.config
    consumption_types = cfg.consumption_types


    # Will hold ModelPodPerformance objects
    rf_rows: List[ModelPodPerformance] = []
    all_forecasts = []

    # Precompute encoding maps from df_feat
    cust_enc_map = (
        df_feat[["CustomerID", "CustomerID_enc"]]
        .drop_duplicates()
        .set_index("CustomerID")["CustomerID_enc"]
        .to_dict()
    )
    pod_enc_map = (
        df_feat[["PodID", "PodID_enc"]]
        .drop_duplicates()
        .set_index("PodID")["PodID_enc"]
        .to_dict()
    )
    if "TariffID_enc" in df_feat.columns:
        tariff_enc_map = (
            df_feat[["TariffID", "TariffID_enc"]]
            .drop_duplicates()
            .set_index("TariffID")["TariffID_enc"]
            .to_dict()
        )
    else:
        tariff_enc_map = {}

    # Last TariffID per (CustomerID, PodID) for carrying forward
    if "TariffID" in df_raw.columns:
        last_tariff = (
            df_raw.groupby(["CustomerID", "PodID"])["TariffID"]
            .apply(lambda x: x.iloc[-1])
            .to_dict()
        )
    else:
        last_tariff = {}

    for (cust_id, pod_id), pod_df in df_raw.groupby(["CustomerID", "PodID"]):
        # Prepare a dict {consumption_type: {"RMSE": ..., "R2": ...}, ...}
        metrics: Dict[str, Dict[str, float]] = {}

        # --- In‐sample performance ---
        if len(pod_df) < cfg.min_history:
            # Insufficient history: record None for each type
            for cons in consumption_types:
                metrics[cons] = {"RMSE": None, "R2": None}
            for z in zero_types:
                metrics[z] = {"RMSE": 0.0, "R2": 1.0}
            # Create and append ModelPodPerformance
            mpp = ModelPodPerformance(
                ModelName=ufm_config.forecast_method_name,
                CustomerID=cust_id,
                PodID=pod_id,
                DataBrickID=ufm_config.databrick_task_id,
                UserForecastMethodID=ufm_config.user_forecast_method_id,
                metrics=metrics
            )
            rf_rows.append(mpp)
        else:
            # Enough history: evaluate each non-zero consumption type
            for cons in consumption_types:
                if cons not in global_models or cons not in test_sets:
                    metrics[cons] = {"RMSE": None, "R2": None}
                else:
                    X_test, y_test = test_sets[cons]
                    y_pred = global_models[cons].predict(X_test)
                    perf = evaluate_regression(y_test, y_pred)
                    metrics[cons] = {"RMSE": perf["RMSE"], "R2": perf["R2"]}
            # Zero-type performance: perfect zeros
            for z in zero_types:
                metrics[z] = {"RMSE": 0.0, "R2": 1.0}

            # Create and append ModelPodPerformance
            mpp = ModelPodPerformance(
                ModelName=ufm_config.forecast_method_name,
                CustomerID=cust_id,
                PodID=pod_id,
                DataBrickID=ufm_config.databrick_task_id,
                UserForecastMethodID=ufm_config.user_forecast_method_id,
                metrics=metrics
            )
            rf_rows.append(mpp)
        # --- Check for large gap between last observed and requested forecast start ---
        last_date = pod_df["ReportingMonth"].max()
        forecast_start = full_forecast_dates[0]
        forecast_end = full_forecast_dates[-1]

        # If last_date is strictly before one month prior to requested horizon start:
        if last_date < (forecast_start - pd.DateOffset(months=1)):
            if gap_handling == "skip":
                meta = get_error_metadata("ForecastGapTooLarge", {
                    "pod_id": pod_id,
                    "last_observed": str(last_date.date()),
                    "requested_start": str(forecast_start.date())
                })
                insert_profiling_error(
                    log_id=None,
                    error=meta["message"],
                    traceback="",
                    error_type="ForecastGapTooLarge",
                    severity=meta["severity"],
                    component=meta["component"]
                )
                # logger.info(f"⛔ Forecast gap too large for Pod {pod_id}. Skipping forecast.")
                # Produce all-zero forecasts for each future month and each consumption type
                zero_records = []
                for dt in full_forecast_dates:
                    base_row = {
                        "CustomerID": cust_id,
                        "PodID": pod_id,
                        "UserForecastMethodID": ufm_config.user_forecast_method_id,
                        "ReportingMonth": dt
                    }
                    for cons in consumption_types + zero_types:
                        base_row[cons] = 0.0
                    zero_records.append(base_row)
                all_forecasts.append(pd.DataFrame(zero_records))
                continue


        # --- Build future feature DataFrame for forecasting ---
        # Only build future horizon from the month after last_date, up to forecast_end
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            end=forecast_end,
            freq="MS"
        )
        future_rows = []
        for dt in future_dates :
            row = {
                "CustomerID": cust_id,
                "PodID": pod_id,
                "ReportingMonth": dt,
                "TimeIndex": ((dt - pod_df["ReportingMonth"].min()).days // 30),
                "month": dt.month,
                "month_sin": np.sin(2 * np.pi * dt.month / 12),
                "month_cos": np.cos(2 * np.pi * dt.month / 12),
                "CustomerID_enc": cust_enc_map.get(cust_id, 0.0),
                "PodID_enc": pod_enc_map.get(pod_id, 0.0),
            }
            if "TariffID" in pod_df.columns:
                tar = last_tariff.get((cust_id, pod_id), np.nan)
                row["TariffID"] = tar
                row["TariffID_enc"] = tariff_enc_map.get(tar, 0.0)
            future_rows.append(row)

        future_df = pd.DataFrame(future_rows)
        combined = pd.concat([pod_df, future_df], ignore_index=True).sort_values("ReportingMonth")
        combined = combined.reset_index(drop=True)

        # Recompute dynamic lags and rolling stats on combined
        for cons in consumption_types:
            for lag in cfg.lag_list:
                combined[f"{cons}_lag{lag}"] = (
                    combined.groupby(["CustomerID", "PodID"])[cons]
                            .shift(lag)
                )
        for cons in consumption_types:
            for window in cfg.rolling_windows:
                combined[f"{cons}_roll{window}_mean"] = (
                    combined.groupby(["CustomerID", "PodID"])[cons]
                            .shift(1)
                            .rolling(window=window)
                            .mean()
                )
                combined[f"{cons}_roll{window}_std"] = (
                    combined.groupby(["CustomerID", "PodID"])[cons]
                            .shift(1)
                            .rolling(window=window)
                            .std()
                            .fillna(0.0)
                )

        # Extract only future rows’ features
        mask_fut = combined["ReportingMonth"].isin(forecast_dates)
        X_future = combined.loc[mask_fut, feature_columns].reset_index(drop=True)
        X_future = X_future.fillna(0.0)

        # Build forecast records: one row per dt, with all consumption types
        forecast_records = []
        for idx, dt in enumerate(forecast_dates):
            forecast_row = {
                "CustomerID": cust_id,
                "PodID": pod_id,
                "UserForecastMethodID": ufm_config.user_forecast_method_id,
                "ReportingMonth": dt
            }

            # Predict each non‐zero consumption type
            for cons in consumption_types:
                rf_model = global_models.get(cons)
                if rf_model is not None and not X_future.empty:
                    try:
                        raw_val = float(rf_model.predict(X_future.iloc[[idx]])[0])
                        val = max(raw_val, 0.0)
                    except Exception as exception:
                        val = 0.0
                        meta = get_error_metadata("ModelFitFailure",{"exception": exception})
                        insert_profiling_error(
                            log_id=None,
                            error=meta["message"],
                            traceback="",  # or traceback.format_exc()
                            error_type="ModelFitFailure",
                            severity=meta["severity"],
                            component=meta["component"]
                        )
                        # logger.warning(f"⚠️ Model training failed: {exception}")
                else:
                    val = 0.0
                forecast_row[cons] = val

            # Zero types are always 0.0
            for z in zero_types:
                forecast_row[z] = 0.0

            forecast_records.append(forecast_row)

        all_forecasts.append(pd.DataFrame(forecast_records))
    rf_performance_df = pd.DataFrame([m.to_row() for m in rf_rows])
    forecast_combined_df = (
        pd.concat(all_forecasts, ignore_index=True)
        if all_forecasts else pd.DataFrame()
    )
    rf_performance_df.fillna(0,inplace=True)
    run_forecast_sanity_checks(forecast_combined_df, rf_performance_df, consumption_types, model)
    return rf_performance_df, forecast_combined_df


def write_to_spark(spark, dataframe, target_table):
    try:
        spark.createDataFrame(dataframe).write.jdbc(url=write_url, table=target_table, mode="append",properties=write_properties)
        logger.info(f"✅ Successfully wrote {len(dataframe)} rows to x{target_table}")
    except Exception as e:
        logger.info(f"🚫 Failed to write to {target_table}")
        safe_exit("E0000", f"Failed to write to {target_table}")


@profiled_function(category="model_training",enabled=profiling_switch.enabled)
def train_random_forest_globally_forecast_locally_with_aggregation(
    model: ForecastModel, spark
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function: Perform Random Forest forecasting for a particular data selection
    """
    forecast_method_id = getattr(model.dataset.ufm_config, "forecast_method_id", None)
    if model.dataset is None or model.dataset.processed_df is None:
        meta = get_error_metadata("ModelConfigMissing", {"field": "dataset.df"})
        insert_profiling_error(
            log_id=None,
            error=meta["message"],
            traceback="",  # or traceback.format_exc()
            error_type="ModelConfigMissing",
            severity=meta["severity"],
            component=meta["component"]
        )
        safe_exit(meta["code"], meta["message"])

    if model.dataset.processed_df.empty:
        meta = get_error_metadata("EmptySeries", {"forecast_method_id": forecast_method_id})
        insert_profiling_error(
            log_id=None,
            error=meta["message"],
            traceback="",  # or traceback.format_exc()
            error_type="EmptySeries",
            severity=meta["severity"],
            component=meta["component"]
        )
        safe_exit(meta["code"], meta["message"])
    df_feat, feature_columns, zero_types, df_raw = prepare_global_training_data(model)
    global_models, test_sets = train_global_rf_models(df_feat, feature_columns, model)
    perf_df, forecast_df = forecast_locally_with_global_models(
        df_feat, feature_columns, global_models, test_sets, model, zero_types, df_raw
    )
    # return perf_df, forecast_df
    write_to_spark(spark, forecast_df, target_table_name)
    write_to_spark(spark, perf_df, performance_metrics_table)
