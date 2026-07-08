# dataset.py

import logging
from typing import Dict, Any, List, Tuple
from pyspark.sql import SparkSession, DataFrame
import traceback
from db.queries import row_to_config, get_user_forecast_data, ForecastConfig
from etl.etl import parse_json_column, generate_combinations, extract_metadata
from .dml import (
    load_and_prepare_data,
    get_unique_list_of_customer_and_pod,
    get_forecast_range
)
from pyspark.sql.functions import col

from IPython.display import display
import logging
from utils.exit_handler import safe_exit
from validation.metadata import get_error_metadata
from db.error_logger import report_validation_error

class DatabricksNotebookHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        display(log_entry)  # Forces output into notebook cells

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j.clientserver").setLevel(logging.INFO)

# Attach notebook handler
if not logger.hasHandlers():
    notebook_handler = DatabricksNotebookHandler()
    notebook_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(notebook_handler)


class ForecastDataset:
    def __init__(self, databrick_task_id: int, spark: SparkSession, save: bool = False):
        self.databrick_task_id = databrick_task_id
        self.save = save
        self.spark = spark
        self.ufm_config: ForecastConfig = self.load_ufm_config()
        self.user_forecast_data = None
        self.raw_df: DataFrame = None
        self.processed_df: DataFrame = None
        self.metadata: dict = {}
        self.customer_ids: list = []
        self.variable_ids: list = []
        self.column_combinations: dict = {}
        self.unique_customers: list = []
        self.unique_pod_ids: list = []
        self.forecast_dates = []
    
    def load_ufm_config(self) -> ForecastConfig:
        """
        Function: Retrieve user forecast configuration from database and store configuration results in an object variable.
        Returns: ForecastConfig: A configuration object created from the first row of the query.
        Raises: ValueError: If the data loaded from the database is empty.
        """
        self.user_forecast_data = get_user_forecast_data(self.spark, self.databrick_task_id)
        if self.user_forecast_data is None or self.user_forecast_data.isEmpty():
            logger.error(f"🚫 User forecast data is empty")
            meta = get_error_metadata("EmptyConfigResult", {"databrick_task_id": self.databrick_task_id})
            logger.info(f"💈 This is what is inside the meta: {meta}")
            report_validation_error(
                log_id=None,
                error=meta["message"],
                traceback="",  # or traceback.format_exc()
                error_type="EmptyConfigResult",
                severity=meta["severity"],
                component=meta["component"]
            )
            safe_exit(meta["code"], meta["message"])
        try:
            first_row = self.user_forecast_data.limit(1).collect()[0]
            ufm_config: ForecastConfig = row_to_config(first_row.asDict())
            logger.info(f"✅ Loaded UFM config: {ufm_config}")
            return ufm_config
        except Exception as e:
            logger.error(f"🚫 Failed to convert row to ForecastConfig: {e}")
            meta = get_error_metadata("EmptyConfigResult", {"databrick_task_id": self.databrick_task_id})
            logger.info(f"💈 This is what is inside the meta: {meta}")
            report_validation_error(
                log_id=None,
                error=meta["message"],
                traceback="",  # or traceback.format_exc()
                error_type="EmptyConfigResult",
                severity=meta["severity"],
                component=meta["component"]
            )
            safe_exit(meta["code"], meta["message"])
        

    def load_data(self) -> None:
        """
        Function: Perform dbo.PredictiveInputData() to fetch and filter data from the database
        Returns: A dataframe that contains all the filtered data
        Raises: ValueError: If the data loaded from the database is empty.
        """
        try:
            logger.info(f"📥 Loading data for forecast_method_id={self.ufm_config.forecast_method_id}")
            self.raw_df = load_and_prepare_data(
                ufm_config=self.ufm_config,
                save=self.save,
                spark=self.spark
            )

            if self.raw_df is None or self.raw_df.empty:
                logger.error("🚫 Raw dataset is empty.")
                meta = get_error_metadata("EmptyQueryResult", {
                    "forecast_method_id": self.ufm_config.forecast_method_id
                })
                report_validation_error(
                    log_id=None,
                    error=meta["message"],
                    traceback="",  # or traceback.format_exc()
                    error_type="EmptySeries",
                    severity=meta["severity"],
                    component=meta["component"]
                )
                safe_exit(meta["code"], meta["message"])

            # Defensive logging
            self.processed_df = self.raw_df  # Spark DataFrames are immutable
            logger.info(f"✅ Data loaded — {len(self.raw_df)} rows assigned to processed_df.")

        except Exception as e:
            logger.exception("❌ Exception in load_data()")
            meta = get_error_metadata("LoadFailure", {
                "forecast_method_id": self.ufm_config.forecast_method_id,
                "exception": str(e)
            })
            report_validation_error(
                log_id=None,
                error=meta["message"],
                traceback=traceback.format_exc(),
                error_type="LoadFailure",
                severity=meta["severity"],
                component="load_data"
            )
            # safe_exit(meta["code"], meta["message"])


    def extract_metadata(self) -> Dict[str, Any]:
        self.metadata = extract_metadata(self.user_forecast_data)
        return self.metadata

    def parse_identifiers(self) -> Tuple[List[str], List[str]]:
        # Spark JSON fields must be parsed outside of Spark DataFrame context (e.g., after collect)
        pdf = self.user_forecast_data.toPandas()
        self.customer_ids = parse_json_column(pdf, "CustomerJSON")
        self.variable_ids = parse_json_column(pdf, "varJSON", key="VariableID")
        logger.info(f"🔍 Parsed customer_ids and variable_ids.")
        return self.customer_ids, self.variable_ids

    def generate_column_combinations(self) -> Dict[Any, List[str]]:
        columns = ["PeakConsumption", "StandardConsumption", "OffPeakConsumption"]
        self.column_combinations = generate_combinations(columns)
        return self.column_combinations

    def extract_unique_customers_and_pods(self) -> Tuple[List[str], List[str]]:
        if self.processed_df is None:
            raise ValueError("❌ Processed DataFrame not found. Call load_data() first.")
        self.unique_customers, self.unique_pod_ids = get_unique_list_of_customer_and_pod(self.processed_df)
        return self.unique_customers, self.unique_pod_ids

    def preprocess(self) -> Dict[str, Any]:
        if self.raw_df is None:
            raise ValueError("❌ Raw DataFrame is not loaded.")
        
        self.metadata = extract_metadata(self.user_forecast_data)
        pdf = self.user_forecast_data.toPandas()
        self.customer_ids = parse_json_column(pdf, "CustomerJSON")
        self.variable_ids = parse_json_column(pdf, "varJSON", key="VariableID")
        self.column_combinations = self.generate_column_combinations()
        self.unique_customers, self.unique_pod_ids = get_unique_list_of_customer_and_pod(self.raw_df)

        return {
            "metadata": self.metadata,
            "customer_ids": self.customer_ids,
            "variable_ids": self.variable_ids,
            "column_combinations": self.column_combinations,
            "unique_customers": self.unique_customers,
            "unique_pod_ids": self.unique_pod_ids
        }

    def define_forecast_range(self) -> List[str]:
        """
        Generate forecast month range using Spark-safe logic (list of date strings).

        Returns:
            List[str]: Forecast period from start_date to end_date.
        """
        try:
            self.forecast_dates = get_forecast_range(self.ufm_config)
            logging.info(f"📅 Forecast range defined: {self.forecast_dates[0]} to {self.forecast_dates[-1]}")
        except Exception as e:
            logging.error(f"🚫 Failed to define forecast range: {e}")
            self.forecast_dates = []

        return self.forecast_dates
