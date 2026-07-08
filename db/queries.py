from pyspark.sql import DataFrame, SparkSession
from .utilities import read_sql_query,get_jdbc_options
from dataclasses import dataclass
import pandas as pd 
import logging
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, TimestampType, FloatType, LongType

logger = logging.getLogger(__name__)

def get_predictive_data(spark: SparkSession, UFMID=64) -> DataFrame:
    import os
    fixture_path = os.getenv("PREDICTIVE_FIXTURE_PATH")
    if fixture_path:
        return pd.read_parquet(fixture_path)
    query = f"SELECT * FROM dbo.PredictiveInputData({UFMID})"
    return read_sql_query(query, spark)


def get_unbundled_predictive_data(spark: SparkSession, UFMID=64) -> DataFrame:
    """PodID-keyed predictive data — the unbundled analogue of
    :func:`get_predictive_data`, backed by the same ``dbo.PredictiveInputData``
    function.

    Honors ``UNBUNDLED_FIXTURE_PATH`` (unbundled-only override) then
    ``PREDICTIVE_FIXTURE_PATH`` for local testing, loading ``.csv`` fixtures
    (e.g. ``data/fixtures/Results.csv`` = PredictiveInputData(421)) or parquet
    as pandas. Otherwise issues a single ``PredictiveInputData(UFMID)`` call.
    """
    import os
    # The dedicated UNBUNDLED_FIXTURE_PATH exists so the UseErmeloFixture toggle
    # can redirect this loader without also redirecting the bundled load_data(),
    # which reads PREDICTIVE_FIXTURE_PATH and expects the bundled parquet sample.
    fixture_path = os.getenv("UNBUNDLED_FIXTURE_PATH") or os.getenv("PREDICTIVE_FIXTURE_PATH")
    if fixture_path:
        if fixture_path.endswith(".csv"):
            # utf-8-sig strips the BOM Results.csv carries on its header.
            return pd.read_csv(fixture_path, encoding="utf-8-sig")
        return pd.read_parquet(fixture_path)
    query = f"SELECT * FROM dbo.PredictiveInputData({UFMID})"
    return read_sql_query(query, spark)

def get_actual_data(spark: SparkSession, rows=5000) -> DataFrame:
    query = f"""
    SELECT TOP {rows}
        ReportingMonth, CustomerID, PodID,
        SUM(OffpeakConsumption) AS OffpeakConsumption,
        SUM(StandardConsumption) AS StandardConsumption,
        SUM(PeakConsumption) AS PeakConsumption,
        SUM(Block1Consumption) AS Block1Consumption,
        SUM(Block2Consumption) AS Block2Consumption,
        SUM(Block3Consumption) AS Block3Consumption,
        SUM(Block4Consumption) AS Block4Consumption,
        SUM(NonTOUConsumption) AS NonTOUConsumption
    FROM ActualData
    GROUP BY ReportingMonth, CustomerID, PodID"""
    return read_sql_query(query, spark)

def get_user_forecast_data(spark: SparkSession, databrick_task_id=39) -> DataFrame:
    query = f"""
    SELECT TOP 1
        ufm.StartDate, ufm.EndDate, ufm.Parameters, ufm.Region, ufm.Status,
        ufm.ForecastMethodID, ufm.UserForecastMethodID,
        ufm.JSONCustomer AS CustomerJSON, ufm.varJSON,
        dfm.Method, dbt.DatabrickID
    FROM [dbo].[DataBrickTasks] AS dbt
    INNER JOIN [dbo].[UserForecastMethod] AS ufm 
        ON dbt.UserForecastMethodID = ufm.UserForecastMethodID
    INNER JOIN [dbo].[DimForecastMethod] AS dfm 
        ON ufm.ForecastMethodID = dfm.ForecastMethodID
    WHERE dbt.DatabrickID = {databrick_task_id}
    ORDER BY dbt.CreationDate"""
    return read_sql_query(query, spark)


@dataclass
class ForecastConfig:
    forecast_method_id: int
    forecast_method_name: str
    model_parameters: str
    region: str
    status: str
    user_forecast_method_id: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    databrick_task_id: int

def row_to_config(row: pd.Series) -> ForecastConfig:
    return ForecastConfig(
        forecast_method_id=row["ForecastMethodID"],
        forecast_method_name=row["Method"],
        model_parameters=row["Parameters"],
        region=row["Region"],
        status=row["Status"],
        user_forecast_method_id=row["UserForecastMethodID"],
        start_date=row["StartDate"],
        end_date=row["EndDate"],
        databrick_task_id=row["DatabrickID"],
    )

