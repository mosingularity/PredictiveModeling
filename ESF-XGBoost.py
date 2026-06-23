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
from notebook_bootstrap import (resolve_env, init_spark, assert_local_workspace,
                                 resolve_task_id, save_fixture, run_unbundled_fixture)
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

# Unbundled local-fixture escape hatch: `--mode unbundled` + PREDICTIVE_FIXTURE_PATH
# runs the per-entity forecaster on a parquet fixture and exits; otherwise no-op.
from models.algorithms.tree_algorithms.xgb import forecast_xgb_unbundled
run_unbundled_fixture(2, "XGBoost", forecast_xgb_unbundled, config)

# COMMAND ----------

spark, dbutils = init_spark()
set_dbutils(dbutils)
assert_local_workspace(spark)
databrick_task_id = resolve_task_id(dbutils)


# COMMAND ----------

from data.dataset import ForecastDataset

# COMMAND ----------

dataset = ForecastDataset(databrick_task_id, spark)

# COMMAND ----------

dataset.ufm_config

# COMMAND ----------

dataset.load_data()

save_fixture(dataset)
# COMMAND ----------

forecast_range = dataset.define_forecast_range()

# COMMAND ----------

from programs.pipeline import ForecastPipeline

# COMMAND ----------

pipeline = ForecastPipeline(dataset=dataset,config=config)

# COMMAND ----------

pipeline.run(spark)

# COMMAND ----------


