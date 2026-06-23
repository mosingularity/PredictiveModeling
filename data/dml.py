# dml_timeseries.py

import logging
from typing import Tuple, List, Optional, Any, Union
from pyspark.sql import DataFrame, SparkSession
from db.queries import ForecastConfig, get_actual_data, get_predictive_data
from pyspark.sql.functions import col, to_date, expr, explode, array, lit, when, to_timestamp, month,year
from pyspark.sql.window import Window
from datetime import datetime
from dateutil.relativedelta import relativedelta
from profiler.errors.utils import get_error_metadata
from pyspark.sql import DataFrame as SparkDataFrame
import pandas as pd

logging.basicConfig(level=logging.INFO)

import os 


def convert_to_pandas(df: SparkDataFrame) -> pd.DataFrame:
    """
    Function: Convert the dataframe to pandas 
    Returns: A pandas dataframe that contains all the filtered data
    Raises: ValueError: If the data loaded from the database is empty.
    """
    row_count = df.count()
    logging.info(f"📊 Converting Spark DF with {row_count} rows to Pandas.")
    try:
        pdf = df.toPandas()
        logging.info(f"✅ Conversion successful. Pandas DF has shape {pdf.shape}")
        return pdf
    except Exception as e:
        logging.error(f"❌ Failed during toPandas(): {e}")
        raise


def load_and_prepare_data(
    ufm_config: ForecastConfig,
    rows: int = 5000,
    actual: bool = False,
    spark: Optional[SparkSession] = None,
    path: str = None,
    save: bool = False
) -> DataFrame:
    """
    Load data from the DB or cache using Spark, clean, and sort it.
    Saves a .parquet snapshot on first fetch to avoid repeated DB calls.
    """
    
    environment = os.getenv("ENV", "DEV")
    logging.info(f"📂 Fetching Spark DataFrame For: {ufm_config.forecast_method_name}")
    data_type = "ActualData" if actual else "PredictiveInputData"
    path = os.path.join(os.getcwd(), "dataset",environment)
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, f"{data_type}{ufm_config.forecast_method_name}.parquet")
    logging.info(f"📂Looking for dataset: {path}")
    
    if spark is None:
        spark = SparkSession.builder.getOrCreate()

    if os.path.isfile(path):
        parquet_path = f"file:{path}"
        logging.info(f"📂 Loading cached Spark DataFrame from: {parquet_path}")
        df = spark.read.parquet(parquet_path)
    else:
        logging.info(f"📥 Querying source via JDBC and saving to: {path}")
        df = get_actual_data(spark, rows) if actual else get_predictive_data(spark, ufm_config.user_forecast_method_id)
    if df.isEmpty():
        logging.warning("🚫 No data retrieved — returning None")
        return None  # ✅ Let the caller handle logging + exit
        
    # df.write.mode("overwrite").parquet(path)
    # logging.info("💾 Data written to cache.")

    df = convert_to_pandas(df)
    logging.info("✅ Converted dataset to pandas for easier use")
    df = clean_dataframe(df)
    logging.info("✅ Cleaned the dataset")
    df = df.sort_values(by=["PodID", "ReportingMonth"])

    df['CustomerID'] = df['CustomerID'].astype(str)
    logging.info("🔄 Converted 'CustomerID' to string.")

    df['PodID'] = df['PodID'].astype(str)
    logging.info("🔄 Converted 'PodID' to string.")
    
    logging.info("🧹 Data cleaned using 'clean_dataframe'.")
    return df

def file_exists(spark: SparkSession, path: str) -> bool:
    """
    Check if a given DBFS or external path exists.
    """
    try:
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
        return fs.exists(spark._jvm.org.apache.hadoop.fs.Path(path))
    except Exception as e:
        logging.warning(f"⚠️ Unable to check file existence at {path}: {e}")
        return False

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Remove CSV index column
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Drop constant columns like UserForecastMethodID
    if 'UserForecastMethodID' in df.columns and df['UserForecastMethodID'].nunique() == 1:
        df.drop(columns=['UserForecastMethodID'], inplace=True)

    # logging.info(f"Dataset: {df}")
    # Convert ReportingMonth to datetime and set as index
    df['ReportingMonth'] = pd.to_datetime(df['ReportingMonth'], format='%Y-%m-%d') \
        .dt.to_period('M').dt.to_timestamp()
    df.set_index('ReportingMonth', inplace=True)

    logging.info("✅ Raw dataset cleaned.")
    return df

def create_month_and_year_columns(df: pd.DataFrame, column: str = 'ReportingMonth') -> pd.DataFrame:
    df.reset_index(inplace=True)
    df['ReportingMonth'] = pd.to_datetime(df['ReportingMonth'])
    df['Month'] = df['ReportingMonth'].dt.month
    df['Year'] = df['ReportingMonth'].dt.year
    return df

def get_forecast_range(ufm_config: ForecastConfig) -> pd.DatetimeIndex:
    try:
        forecast_dates = pd.date_range(start=ufm_config.start_date, end=ufm_config.end_date, freq='MS')
        logging.info(f"📅 Forecast period: {forecast_dates[0]} to {forecast_dates[-1]}")
        return forecast_dates
    except Exception as e:
        logging.error(f"Failed to generate forecast range: {e}")
        return pd.DatetimeIndex([])


def time_series_train_test_split(
    series: pd.Series,
    test_ratio: float = 0.2
) -> Tuple[pd.Series, pd.Series, int]:
    """
    Splits a time series into train and test sets, using a fraction of the dataset
    for the test set. The function ensures the test set is at least 1 record
    if the series is very small.

    Parameters
    ----------
    series : pd.Series
        The time series dataset.
    test_ratio : float, optional
        The fraction of the dataset to allocate to the test set. Default is 0.2 (20%).

    Returns
    -------
    train : pd.Series
        The training subset of the series.
    test : pd.Series
        The test (hold-out) subset of the series.
    """
    n = len(series)
    test_size = max(int(n * test_ratio), 1)
    test_size = min(test_size, n - 1)  # ensure at least 1 train point

    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]

    return train, test, test_size


def train_test_split(X, Y, test_size=0.2):
    """
    Splits time series dataset into training and test sets while preserving the temporal order.

    Parameters:
    X (array-like): Feature dataset.
    Y (array-like): Target dataset.
    test_size (float): Proportion of the dataset to reserve for testing. Default is 0.2 (20%).

    Returns:
    tuple: A tuple containing:
           - (X_train, Y_train): Training dataset.
           - (X_test, Y_test): Testing dataset.
    """
    n_samples = len(X)
    # Determine the index at which to split the dataset.
    split_idx = int(n_samples * (1 - test_size))

    # Create the train/test split.
    X_train = X[:split_idx]
    Y_train = Y[:split_idx]
    X_test = X[split_idx:]
    Y_test = Y[split_idx:]

    return (X_train, Y_train), (X_test, Y_test)


def split_time_series_three_way(
    series: pd.Series,
    final_test_ratio: float = 0.2,
    validation_ratio: float = 0.25
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Splits the series into:
      - final_test (the last `final_test_ratio` portion)
      - validation (the last `validation_ratio` portion of the remaining dataset)
      - sub_train (the rest)
    in chronological order.
    """
    n = len(series)
    final_test_size = max(1, int(n * final_test_ratio))
    final_test_size = min(final_test_size, n - 1)

    # Final test is the tail
    final_test = series.iloc[-final_test_size:]
    train_val = series.iloc[:-final_test_size]

    val_size = max(1, int(len(train_val) * validation_ratio))
    val_size = min(val_size, len(train_val) - 1)

    validation = train_val.iloc[-val_size:]
    sub_train = train_val.iloc[:-val_size]

    return sub_train, validation, final_test

def split_time_series_three_ways(index: pd.Index,
                                val_ratio: float = 0.25,
                                test_ratio: float = 0.2) -> Tuple[pd.Index, pd.Index, pd.Index]:
    """
    Splits the index into sub_train, validation, final_test.
    E.g. if you have 100 points, test_ratio=0.2 => 20 test points,
    then val_ratio=0.25 => 20 of the remaining 80 => 20 val, 60 sub_train.
    """
    n = len(index)
    test_size = int(n * test_ratio)
    test_size = min(max(test_size, 0), n)
    remain_after_test = n - test_size

    val_size = int(remain_after_test * val_ratio)
    val_size = min(max(val_size, 0), remain_after_test)

    sub_train_idx = index[:remain_after_test - val_size]
    val_idx = index[remain_after_test - val_size: remain_after_test]
    final_test_idx = index[remain_after_test:]
    return sub_train_idx, val_idx, final_test_idx


def prepare_lag_features(df, lag_columns = None, lags=3, base_features=None):
    """
    Create lag features for the specified columns, convert them to numeric,
    and construct a final list of feature columns.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        lag_columns (list of str): List of column names for which to create lag features.
        lags (int): Number of lag periods to generate.
        base_features (list of str, optional): List of base feature column names.
            Defaults to ["Month", "Year"].

    Returns:
        tuple: A tuple containing:
            - The modified DataFrame with lag features.
            - A list of feature column names including base features and lag features.
    """
    # Use default base_features if none provided
    if base_features is None:
        base_features = ["Month", "Year"]
    if lag_columns is None:
        lag_columns = ['StandardConsumption']
    # Create lag features using the provided function (assumed to be defined elsewhere)
    df = create_lag_features(df, lag_columns, lags)

    # Generate list of lag feature column names
    lag_feature_cols = [f"{col}_lag{lag}" for col in lag_columns for lag in range(1, lags + 1)]

    # Ensure lag feature columns are numeric and fill missing values with 0
    for col in lag_feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Construct final list of feature columns
    feature_columns = base_features + lag_feature_cols

    return df, feature_columns

def get_melted_df(df: pd.DataFrame, customer_col: str= 'CustomerID',id_vars=None, value_vars=None, consumption_type: str = "ConsumptionType", value_name: str = 'kWh'):
    if id_vars is None:
        id_vars = ['CustomerID', 'ReportingMonth']
    if value_vars is None:
        value_vars = [
            'PeakConsumption', 'StandardConsumption', 'OffPeakConsumption',
            'Block1Consumption', 'Block2Consumption', 'Block3Consumption',
            'Block4Consumption', 'NonTOUConsumption'
        ]
    top_customers = df[customer_col].value_counts().head(4).index.tolist()
    sample_df = df.reset_index()
    sample_df = sample_df[sample_df[customer_col].isin(top_customers)]
    # Melt the dataframe for easier time series plotting
    melted_df = pd.melt(sample_df,id_vars=id_vars,value_vars=value_vars,var_name=consumption_type,value_name=value_name)
    return melted_df

def prepare_features_and_target(
    df: pd.DataFrame,
    feature_columns: List[str],
    consumption_type: str
) -> (pd.DataFrame, pd.Series):
    """
    Extracts X and y from df for the given consumption_type.
    Example: X = df[feature_columns], y = df[consumption_type].
    You might handle missing or frequency alignment here too.
    """
    sub_df = df.dropna(subset=[consumption_type])
    if sub_df.empty:
        return pd.DataFrame(), pd.Series(dtype=float)
    X = sub_df[feature_columns]
    y = sub_df[consumption_type]
    return X, y


def convert_column(df, col: str = 'PodID', to_type: type(str) = str):
    df = df.astype({col: to_type})
    return df


def create_lag_features(df, lag_columns, lags):
    for col in lag_columns:
            for lag in range(1,lags+1):
                df[f"{col}_lag{lag}"]=df[col].shift(lag)
    return df

import re
from typing import Tuple

def extract_xgboost_params(param_str: str) -> Tuple[float, float, float, float, float]:
    try:
        matches = re.findall(r'\(.*?\)', param_str)
        params = tuple(map(float, matches[0].strip('()').split(',')))
        return params
    except Exception as e:
        logging.error(f"❌ Failed to parse XGBoost parameters: {e}")
        return (0, 0, 0, 0, 0)
    
def extract_random_forest_params(
    param_str: str
) -> Tuple[
    Optional[int],       # n_estimators
    Optional[int],       # max_depth
    Optional[int],       # min_samples_split
    Optional[int],       # min_samples_leaf
    Union[int, float, str, None],  # max_features
    bool                 # bootstrap
]:
    """
    Parse a string like "(50,10,2,1,0.75,true)" into a 6-tuple of RF params:
      (n_estimators, max_depth, min_samples_split,
       min_samples_leaf, max_features, bootstrap)

    max_features : {"sqrt", "log2", None}, int or float, default="sqrt"
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `max(1, int(max_features * n_features_in_))` features are considered at each
          split.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

    If the string contains exactly one value—e.g. "(100)"—we return the
    full set of defaults instead. On any other parse error, we also fall
    back to defaults.
    """
    # Define the “canonical” defaults
    DEFAULT_PARAMS: Tuple[
        Optional[int], Optional[int], Optional[int],
        Optional[int], Union[int, float, str, None], bool
    ] = (
        100,     # n_estimators
        10,    # max_depth
        2,       # min_samples_split
        1,       # min_samples_leaf
        5,  # max_features
        True     # bootstrap
    )

    try:
        # 1) Extract inside of first parentheses
        match = re.search(r'\((.*?)\)', param_str)
        if not match:
            raise ValueError("No parentheses found")

        parts = [p.strip() for p in match.group(1).split(',')]

        # 2) Edge case: single value → return full defaults
        if len(parts) == 1:
            logging.info(
                f"ℹ️ Only one parameter '{parts[0]}' provided; "
                "falling back to all default RF params"
            )
            return DEFAULT_PARAMS

        # 3) Must have exactly six
        if len(parts) != 6:
            raise ValueError(f"Expected 6 values, got {len(parts)}")

        # 4) Convert first four to ints
        n_estimators    = int(parts[0])
        max_depth       = int(parts[1])
        min_samples_split = int(parts[2])
        min_samples_leaf  = int(parts[3])

        # 5) Parse max_features
        mf = parts[4].lower()
        if mf in ("sqrt", "log2"):
            max_features: Union[int, float, str, None] = mf
        elif mf in ("none", ""):
            max_features = None
        else:
            # try int, then float
            try:
                max_features = int(parts[4])
            except ValueError:
                max_features = float(parts[4])

        # 6) Last: boolean
        bool_str = parts[5].lower()
        bootstrap = bool_str in ("true", "tru", "1", "yes")

        return (
            n_estimators,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features,
            bootstrap
        )

    except Exception as e:
        logging.error(f"❌ Failed to parse RF params '{param_str}': {e}")
        return DEFAULT_PARAMS

def extract_sarimax_params(param_str: str) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
    try:
        matches = re.findall(r'\(.*?\)', param_str)
        order = tuple(map(int, matches[0].strip('()').split(',')))
        seasonal_order = tuple(map(int, matches[1].strip('()').split(','))) if len(matches) > 1 else None
        logging.info(f"📌 Parsed ARIMA Order: {order}, Seasonal Order: {seasonal_order}")
        return order, seasonal_order
    except Exception as e:
        logging.error(f"❌ Failed to parse SARIMAX parameters: {e}")
        return (0, 0, 0), (0, 0, 0, 0)

from typing import List, Tuple

def get_unique_list_of_customer_and_pod(df: pd.DataFrame) -> Tuple[list, list]:
    customers = df['CustomerID'].unique().tolist()
    pod_ids = df['PodID'].unique().tolist()

    if customers:
        logging.info(f"🧮 Forecasting for {len(customers)}")
    else:
        logging.warning("⚠️ No Customer IDs found for forecasting.")

    return customers, pod_ids

