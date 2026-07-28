"""Named forecast-data queries and their configuration record."""

import os
from dataclasses import dataclass

import pandas as pd
from pyspark.sql import DataFrame, SparkSession

from .utilities import read_sql_query


def get_predictive_data(spark: SparkSession, UFMID=64) -> DataFrame:
    fixture_path = os.getenv("PREDICTIVE_FIXTURE_PATH")
    if fixture_path:
        return pd.read_parquet(fixture_path)
    query = f"SELECT * FROM dbo.PredictiveInputData({UFMID})"
    return read_sql_query(query, spark)


def get_unbundled_predictive_data(spark: SparkSession, UFMID=64) -> DataFrame:
    """Load PodID-keyed predictive data from a fixture or SQL."""
    fixture_path = os.getenv("UNBUNDLED_FIXTURE_PATH") or os.getenv(
        "PREDICTIVE_FIXTURE_PATH"
    )
    if fixture_path:
        if fixture_path.endswith(".csv"):
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
        ufm.BundledInd,
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
    # Existing configs default to the original unbundled mode.
    bundled: bool = False


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
        bundled=bool(row.get("BundledInd")),
    )
