# Databricks notebook source
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
except (ConnectionResetError, SocketError, SocketTimeout) as e:
        logger.error(
            f"❌ Connection error during Spark operation: {type(e).__name__} — {str(e)}"
        )
except Py4JNetworkError as e:
    logger.error(
        f"🔥 Py4JNetworkError during Spark operation:\n"
    )
except Exception as e:
    # Catch absolutely everything else
    logger.error(
        f"🚨 Unexpected error during Spark operation:\n"
    )

# COMMAND ----------

config = load_config("config.yaml") # Forecast config

# COMMAND ----------

# Unbundled local-fixture escape hatch: `--mode unbundled` + PREDICTIVE_FIXTURE_PATH
# runs the per-entity forecaster on a parquet fixture and exits; otherwise no-op.
from models.algorithms.tree_algorithms.rf import forecast_rf_unbundled
run_unbundled_fixture(3, "RandomForest", forecast_rf_unbundled, config)

# COMMAND ----------

spark, dbutils = init_spark()
set_dbutils(dbutils)
assert_local_workspace(spark)
databrick_task_id = resolve_task_id(dbutils)


# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

from data.dataset import ForecastDataset

# COMMAND ----------

dataset = ForecastDataset(databrick_task_id, spark)

# COMMAND ----------

# !python --version  # diagnostic-only Databricks magic; commented so the file parses under plain `python`

# COMMAND ----------

dataset.load_data()

save_fixture(dataset)
# COMMAND ----------

dataset.processed_df

# COMMAND ----------

forecast_range = dataset.define_forecast_range()

# COMMAND ----------

from programs.pipeline import ForecastPipeline

# COMMAND ----------

dataset

# COMMAND ----------

pipeline = ForecastPipeline(dataset=dataset,config=config)

# COMMAND ----------

pipeline.run(spark)

# COMMAND ----------

