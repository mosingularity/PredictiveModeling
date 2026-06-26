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
    """Entity-keyed (LPU/SPU/PPU) predictive data — the unbundled analogue of
    :func:`get_predictive_data`.

    Honors ``PREDICTIVE_FIXTURE_PATH`` for local testing (returns the parquet
    fixture as pandas). On a cluster / DEV it runs the shared unbundled query once
    per tariff-scope filter (SPU, LPU, …) and appends the results — the same
    Ermelo-shaped contract the unbundled loop groups on.
    """
    import os
    from db.unbundled_query import build_query, to_contract, UNBUNDLED_FILTERS
    fixture_path = os.getenv("PREDICTIVE_FIXTURE_PATH")
    if fixture_path:
        return pd.read_parquet(fixture_path)
    frames = []
    for label, where_clause in UNBUNDLED_FILTERS:
        logger.info(f"📥 Unbundled query — {label} scope.")
        raw = read_sql_query(build_query(where_clause), spark).toPandas()
        frames.append(to_contract(raw))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

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

