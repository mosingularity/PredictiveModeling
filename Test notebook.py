# Databricks notebook source
# from pyspark.sql import SparkSession
# from pyspark.dbutils import DBUtils
# from utils.dbutils_singleton import set_dbutils
# from config_loader import load_config  # your existing config loader module
# from data.dataset import ForecastDataset
# from profiler.profiler_run import run_context
# from programs.pipeline import ForecastPipeline

# def init_spark():
#     spark = SparkSession.builder.appName("Energy Consumption Prediction").getOrCreate()
#     spark.sparkContext.setLogLevel("ERROR")
#     dbutils = DBUtils(spark)
#     return spark, dbutils

# COMMAND ----------

# spark, dbutils = init_spark()
# set_dbutils(dbutils) 
# databrick_task_id = int(dbutils.widgets.get("DatabrickTaskID"))
# print(databrick_task_id)
# print(spark)


# COMMAND ----------

#dataset = ForecastDataset(databrick_task_id, spark)

# COMMAND ----------



# COMMAND ----------

#import db.utilities

query = "select getdate() as testdate"

df = (
            spark.read.format("jdbc")
            # .option("url", jdbc_url)
            .option("url", "jdbc:sqlserver://fortrack-maz-sdb-san-prod-01.database.windows.net:1433;database=FortrackDB;Authentication=ActiveDirectoryMSI;trustServerCertificate=true")
            .option("query", query)
            .load()
        )

#df = db.utilities.read_sql_query(query,spark)

display(df)

# COMMAND ----------

# MAGIC %%sh
# MAGIC telnet fortrack-maz-sdb-san-prod-01.database.windows.net 1433

# COMMAND ----------

# MAGIC %%sh
# MAGIC ping fortrack-maz-sdb-san-prod-01.database.windows.net

# COMMAND ----------

query = "SELECT GETDATE() AS testdate"

jdbc_url = (
    "jdbc:sqlserver://fortrack-maz-sdb-san-prod-01.database.windows.net:1433;"
    "database=FortrackDB;"
    "encrypt=true;"
    "trustServerCertificate=false;"
    "hostNameInCertificate=*.database.windows.net;"
    "Authentication=ActiveDirectoryMSI"
)

df = (
    spark.read.format("jdbc")
    .option("url", jdbc_url)
    .option("driver", "com.microsoft.sqlserver.jdbc.SQLServerDriver")
    .option("query", query)
    .load()
)

display(df)