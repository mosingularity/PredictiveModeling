"""Load, clean, and parse forecast data and model parameters."""

import logging
import os
import re
from typing import Optional, Tuple, Union

import pandas as pd
from pyspark.sql import DataFrame, SparkSession

from db.queries import ForecastConfig, get_actual_data, get_predictive_data

logging.basicConfig(level=logging.INFO)


def convert_to_pandas(df: DataFrame) -> pd.DataFrame:
    """Return a pandas frame, converting Spark input when needed."""
    if isinstance(df, pd.DataFrame):
        return df
    row_count = df.count()
    logging.info(f"📊 Converting Spark DF with {row_count} rows to Pandas.")
    try:
        pandas_df = df.toPandas()
        logging.info(f"✅ Conversion successful. Pandas DF has shape {pandas_df.shape}")
        return pandas_df
    except Exception as exc:
        logging.error(f"❌ Failed during toPandas(): {exc}")
        raise


def load_and_prepare_data(
    ufm_config: ForecastConfig,
    rows: int = 5000,
    actual: bool = False,
    spark: Optional[SparkSession] = None,
    save: bool = False,
) -> DataFrame:
    """Load predictive or actual data, then clean and sort it."""
    environment = os.getenv("ENV", "DEV")
    logging.info(f"📂 Fetching Spark DataFrame For: {ufm_config.forecast_method_name}")
    data_type = "ActualData" if actual else "PredictiveInputData"
    cache_directory = os.path.join(os.getcwd(), "dataset", environment)
    os.makedirs(cache_directory, exist_ok=True)
    cache_path = os.path.join(
        cache_directory, f"{data_type}{ufm_config.forecast_method_name}.parquet"
    )
    logging.info(f"📂Looking for dataset: {cache_path}")

    if spark is None:
        spark = SparkSession.builder.getOrCreate()

    if os.path.isfile(cache_path):
        parquet_path = f"file:{cache_path}"
        logging.info(f"📂 Loading cached Spark DataFrame from: {parquet_path}")
        df = spark.read.parquet(parquet_path)
    else:
        logging.info(f"📥 Querying source via JDBC and saving to: {cache_path}")
        if actual:
            df = get_actual_data(spark, rows)
        else:
            df = get_predictive_data(spark, ufm_config.user_forecast_method_id)
    is_empty = df.empty if isinstance(df, pd.DataFrame) else df.isEmpty()
    if is_empty:
        logging.warning("🚫 No data retrieved — returning None")
        return None

    df = convert_to_pandas(df)
    logging.info("✅ Converted dataset to pandas for easier use")
    df = clean_dataframe(df)
    logging.info("✅ Cleaned the dataset")
    df = df.sort_values(by=["PodID", "ReportingMonth"])

    df["CustomerID"] = df["CustomerID"].astype(str)
    logging.info("🔄 Converted 'CustomerID' to string.")

    df["PodID"] = df["PodID"].astype(str)
    logging.info("🔄 Converted 'PodID' to string.")

    logging.info("🧹 Data cleaned using 'clean_dataframe'.")
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the loaded monthly frame."""
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    if (
        "UserForecastMethodID" in df.columns
        and df["UserForecastMethodID"].nunique() == 1
    ):
        df.drop(columns=["UserForecastMethodID"], inplace=True)

    df["ReportingMonth"] = (
        pd.to_datetime(df["ReportingMonth"], format="%Y-%m-%d")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    df.set_index("ReportingMonth", inplace=True)

    logging.info("✅ Raw dataset cleaned.")
    return df


def get_forecast_range(ufm_config: ForecastConfig) -> pd.DatetimeIndex:
    try:
        forecast_dates = pd.date_range(
            start=ufm_config.start_date, end=ufm_config.end_date, freq="MS"
        )
        logging.info(f"📅 Forecast period: {forecast_dates[0]} to {forecast_dates[-1]}")
        return forecast_dates
    except Exception as exc:
        logging.error(f"Failed to generate forecast range: {exc}")
        return pd.DatetimeIndex([])


def extract_xgboost_params(param_str: str) -> Tuple[float, float, float, float, float]:
    try:
        matches = re.findall(r"\(.*?\)", param_str)
        values = matches[0].strip("()").split(",")
        return tuple(float(value) for value in values)
    except Exception as exc:
        logging.error(f"❌ Failed to parse XGBoost parameters: {exc}")
        return (0, 0, 0, 0, 0)


def extract_random_forest_params(
    param_str: str,
) -> Tuple[
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
    Union[int, float, str, None],
    bool,
]:
    """Parse Random Forest parameters, falling back to canonical defaults."""
    defaults: Tuple[
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Union[int, float, str, None],
        bool,
    ] = (100, 10, 2, 1, 5, True)

    try:
        match = re.search(r"\((.*?)\)", param_str)
        if not match:
            raise ValueError("No parentheses found")

        parts = [part.strip() for part in match.group(1).split(",")]

        if len(parts) == 1:
            logging.info(
                f"ℹ️ Only one parameter '{parts[0]}' provided; "
                "falling back to all default RF params"
            )
            return defaults

        if len(parts) != 6:
            raise ValueError(f"Expected 6 values, got {len(parts)}")

        n_estimators = int(parts[0])
        max_depth = int(parts[1])
        min_samples_split = int(parts[2])
        min_samples_leaf = int(parts[3])

        max_features_value = parts[4].lower()
        if max_features_value in ("sqrt", "log2"):
            max_features: Union[int, float, str, None] = max_features_value
        elif max_features_value in ("none", ""):
            max_features = None
        else:
            try:
                max_features = int(parts[4])
            except ValueError:
                max_features = float(parts[4])

        bootstrap = parts[5].lower() in ("true", "tru", "1", "yes")

        return (
            n_estimators,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features,
            bootstrap,
        )
    except Exception as exc:
        logging.error(f"❌ Failed to parse RF params '{param_str}': {exc}")
        return defaults


def extract_sarimax_params(
    param_str: str,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
    try:
        matches = re.findall(r"\(.*?\)", param_str)
        order_values = matches[0].strip("()").split(",")
        order = tuple(int(value) for value in order_values)
        seasonal_order = None
        if len(matches) > 1:
            seasonal_values = matches[1].strip("()").split(",")
            seasonal_order = tuple(int(value) for value in seasonal_values)
        logging.info(f"📌 Parsed ARIMA Order: {order}, Seasonal Order: {seasonal_order}")
        return order, seasonal_order
    except Exception as exc:
        logging.error(f"❌ Failed to parse SARIMAX parameters: {exc}")
        return (0, 0, 0), (0, 0, 0, 0)

