"""Forecast run data and configuration loading."""

import logging
import traceback
from typing import List

from IPython.display import display
from pyspark.sql import DataFrame, SparkSession

from db.error_logger import report_validation_error
from db.queries import ForecastConfig, get_user_forecast_data, row_to_config
from utils.exit_handler import safe_exit
from validation.metadata import get_error_metadata

from .dml import get_forecast_range, load_and_prepare_data


class DatabricksNotebookHandler(logging.Handler):
    """Display log records in Databricks notebook cells."""

    def emit(self, record):
        log_entry = self.format(record)
        display(log_entry)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j.clientserver").setLevel(logging.WARNING)

if not logger.hasHandlers():
    notebook_handler = DatabricksNotebookHandler()
    notebook_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(notebook_handler)


class ForecastDataset:
    """Load one forecast task's configuration, data, and forecast range."""

    def __init__(self, databrick_task_id: int, spark: SparkSession, save: bool = False):
        self.databrick_task_id = databrick_task_id
        self.save = save
        self.spark = spark
        self.ufm_config: ForecastConfig = self.load_ufm_config()
        self.user_forecast_data = None
        self.raw_df: DataFrame = None
        self.processed_df: DataFrame = None
        self.forecast_dates = []

    def load_ufm_config(self) -> ForecastConfig:
        """Load the task's first user forecast configuration row."""
        self.user_forecast_data = get_user_forecast_data(self.spark, self.databrick_task_id)
        if self.user_forecast_data is None or self.user_forecast_data.isEmpty():
            logger.error("🚫 User forecast data is empty")
            error_metadata = get_error_metadata(
                "EmptyConfigResult", {"databrick_task_id": self.databrick_task_id}
            )
            report_validation_error(
                log_id=None,
                error=error_metadata["message"],
                traceback="",
                error_type="EmptyConfigResult",
                severity=error_metadata["severity"],
                component=error_metadata["component"],
            )
            safe_exit(error_metadata["code"], error_metadata["message"])
        try:
            first_row = self.user_forecast_data.limit(1).collect()[0]
            ufm_config: ForecastConfig = row_to_config(first_row.asDict())
            logger.info(f"✅ Loaded UFM config: {ufm_config}")
            return ufm_config
        except Exception as exc:
            logger.error(f"🚫 Failed to convert row to ForecastConfig: {exc}")
            error_metadata = get_error_metadata(
                "EmptyConfigResult", {"databrick_task_id": self.databrick_task_id}
            )
            report_validation_error(
                log_id=None,
                error=error_metadata["message"],
                traceback="",
                error_type="EmptyConfigResult",
                severity=error_metadata["severity"],
                component=error_metadata["component"],
            )
            safe_exit(error_metadata["code"], error_metadata["message"])

    def load_data(self) -> None:
        """Load and prepare the task's predictive input data."""
        try:
            logger.info(
                f"📥 Loading data for "
                f"forecast_method_id={self.ufm_config.forecast_method_id}"
            )
            self.raw_df = load_and_prepare_data(
                ufm_config=self.ufm_config,
                save=self.save,
                spark=self.spark,
            )

            if self.raw_df is None or self.raw_df.empty:
                logger.error("🚫 Raw dataset is empty.")
                error_metadata = get_error_metadata(
                    "EmptyQueryResult",
                    {"forecast_method_id": self.ufm_config.forecast_method_id},
                )
                report_validation_error(
                    log_id=None,
                    error=error_metadata["message"],
                    traceback="",
                    error_type="EmptySeries",
                    severity=error_metadata["severity"],
                    component=error_metadata["component"],
                )
                safe_exit(error_metadata["code"], error_metadata["message"])

            self.processed_df = self.raw_df
            logger.info(
                f"✅ Data loaded — {len(self.raw_df)} rows assigned to processed_df."
            )

        except Exception as exc:
            logger.exception("❌ Exception in load_data()")
            error_metadata = get_error_metadata(
                "LoadFailure",
                {
                    "forecast_method_id": self.ufm_config.forecast_method_id,
                    "exception": str(exc),
                },
            )
            report_validation_error(
                log_id=None,
                error=error_metadata["message"],
                traceback=traceback.format_exc(),
                error_type="LoadFailure",
                severity=error_metadata["severity"],
                component="load_data",
            )

    def define_forecast_range(self) -> List[str]:
        """Set and return the configured monthly forecast range."""
        try:
            self.forecast_dates = get_forecast_range(self.ufm_config)
            logging.info(
                f"📅 Forecast range defined: {self.forecast_dates[0]} "
                f"to {self.forecast_dates[-1]}"
            )
        except Exception as exc:
            logging.error(f"🚫 Failed to define forecast range: {exc}")
            self.forecast_dates = []

        return self.forecast_dates
