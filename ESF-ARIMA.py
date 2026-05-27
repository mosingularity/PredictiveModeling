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
os.environ["ENV"] = "DEV"


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
    from profiler.profiler_run import run_context
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
    print(e)
    # Catch absolutely everything else
    logger.error(
        f"🚨 Unexpected error during Spark operation:\n"
    )

# COMMAND ----------

from programs.pipeline import ForecastPipeline

# COMMAND ----------

config = load_config("config.yaml") # Forecast config

# COMMAND ----------

def init_spark():
    spark = SparkSession.builder.appName("Energy Consumption Prediction").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    dbutils = DBUtils(spark)
    return spark, dbutils


# COMMAND ----------

spark, dbutils = init_spark()
set_dbutils(dbutils) 
databrick_task_id = int(dbutils.widgets.get("DatabrickTaskID"))
databrick_task_id


# COMMAND ----------

from data.dataset import ForecastDataset

# COMMAND ----------

dataset = ForecastDataset(databrick_task_id, spark)

# COMMAND ----------

dataset.ufm_config

# COMMAND ----------

dataset.load_data()

# COMMAND ----------

forecast_range = dataset.define_forecast_range()

# COMMAND ----------

from programs.pipeline import ForecastPipeline

# COMMAND ----------

pipeline = ForecastPipeline(dataset=dataset,config=config)

# COMMAND ----------

pipeline.run(spark)

# COMMAND ----------


