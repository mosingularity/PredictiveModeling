import os
import logging
from pyspark.sql import SparkSession


def init_spark():
    spark = SparkSession.builder.appName("Energy Consumption Forecasting").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    print("This is where I start logging")
    logging.getLogger().setLevel(logging.WARNING)
    return spark

def init_secrets():
    return {
        "DB_USER": os.getenv("DB_USER", "fortrackSQL"),
        "DB_PASSWORD": os.getenv("DB_PASSWORD", "your-default-password")
    }
