# Databricks notebook source
import yaml

# COMMAND ----------

import sys
import logging
from py4j.protocol import Py4JNetworkError
from socket import error as SocketError, timeout as SocketTimeout
from config_loader import load_config 
sys.path.append("/Workspace/Shared")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j.clientserver").setLevel(logging.WARNING)
import os
from notebook_bootstrap import (resolve_env, bootstrap_run, run_forecast,
                                 render_unbundled)
# ENV picks the config.yaml section and DB host. An explicit ENV always wins;
# otherwise DEV locally (databricks-connect), PROD on a Databricks cluster.
resolve_env()


# COMMAND ----------

import mlflow
mlflow.autolog(disable=True)
mlflow.statsmodels.autolog(disable=True)


# COMMAND ----------

try:
    from pyspark.sql import SparkSession
    from pyspark.dbutils import DBUtils
    from utils.dbutils_singleton import set_dbutils
    from config_loader import load_config  # your existing config loader module
    from data.dataset import ForecastDataset
    from programs.pipeline import ForecastPipeline
    logger.info("✅ Core imports OK (pyspark, ForecastDataset, ForecastPipeline).")
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

from programs.pipeline import ForecastPipeline

# COMMAND ----------

config = load_config("config.yaml") # Forecast config

# COMMAND ----------

# Resolve the whole run environment in one call: local offline gate → Spark →
# widgets → dataset (+ bundled load_data guard) → (spark, dataset, unbundled).
from models.algorithms.tree_algorithms.xgb import forecast_xgb_unbundled
spark, dataset, unbundled = bootstrap_run(2, "XGBoost", forecast_xgb_unbundled, config)

# COMMAND ----------

# What the run resolved to (task id, UFMID, model, forecast dates) — display-only QA.
dataset.ufm_config

# COMMAND ----------

# Mode dispatch — bundled by default; the Unbundled widget/env selects the entity-keyed path.
result = run_forecast(dataset, spark, config, forecast_xgb_unbundled, unbundled)

# COMMAND ----------



# Unbundled path is display-only — render forecast output inline (no DB writes).
# The bundled path persists to ForecastFact / StatisticalPerformanceMetrics instead.
render_unbundled(result, unbundled)
