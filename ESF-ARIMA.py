# Databricks notebook source

# COMMAND ----------

import sys
import logging
from py4j.protocol import Py4JNetworkError
from socket import error as SocketError, timeout as SocketTimeout
from config_loader import load_config
sys.path.append("/Workspace/Shared")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
import os
from notebook_bootstrap import bootstrap_run, run_forecast, render_forecast

# COMMAND ----------

import mlflow
mlflow.autolog(disable=True)
mlflow.statsmodels.autolog(disable=True)


# COMMAND ----------

try:
    from pyspark.sql import SparkSession
    from pyspark.dbutils import DBUtils
    from utils.dbutils_singleton import set_dbutils
    from config_loader import load_config
    from data.dataset import ForecastDataset
    logger.info("✅ Core imports OK (pyspark, ForecastDataset).")
except (ConnectionResetError, SocketError, SocketTimeout) as e:
    logger.error(f"❌ Connection error during import: {type(e).__name__} — {e}")
    raise
except Py4JNetworkError as e:
    logger.error(f"🔥 Py4JNetworkError during import: {type(e).__name__} — {e}")
    raise
except Exception as e:
    logger.error(f"🚨 Import failed — pipeline/dataset unavailable: {type(e).__name__} — {e}")
    raise

# COMMAND ----------

config = load_config("config.yaml")

# COMMAND ----------

from models.algorithms.autoarima import forecast_arima_unbundled
from models.algorithms.bundled import run_bundled
forecasters = {"unbundled": forecast_arima_unbundled, "bundled": run_bundled}
spark, dataset = bootstrap_run(1, "ARIMA", forecasters, config)

# COMMAND ----------

dataset.ufm_config

# COMMAND ----------

result = run_forecast(dataset, spark, config, forecasters)

# COMMAND ----------

render_forecast(result)
