# utilities_spark.py

import os
import yaml
import logging
from typing import Tuple, Dict
from pyspark.sql import SparkSession, DataFrame

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_yaml_config(config_path: str = "config.yaml") -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_environment_config(config_path: str = "config.yaml") -> Tuple[str, Dict]:
    env = os.getenv("ENV", "QA")
    config = load_yaml_config(config_path)
    if env not in config:
        raise ValueError(f"Environment '{env}' not found in {config_path}. Available: {list(config)}")
    return env, config[env]


def get_jdbc_options(config_path: str = "config.yaml") -> Tuple[str, str, str]:
    _, env_cfg = get_environment_config(config_path)
    user = os.getenv("DB_USER", "fortrackSQL")
    password = os.getenv("DB_PASSWORD", "")
    # The Spark JDBC read executes on the Databricks cluster and authenticates via
    # its managed identity (MSI), so the only thing that varies per ENV is the host.
    # ENV (DEV/UAT/QA/PROD) selects the section in config.yaml — no code change needed.
    url = (
        f"{env_cfg['server']}:1433;"
        f"database={env_cfg['database']};"
        "Authentication=ActiveDirectoryMSI"
    )
    return url, user, password


def jdbc_write(spark, dataframe, table, mode="append", config_path="config.yaml"):
    """Write a pandas/Spark DataFrame to a SQL Server table over JDBC.

    The host is resolved through ``get_jdbc_options`` — the SAME ENV-driven source
    as the reads — so a write can never target a different environment than the
    read (previously each model hardcoded its own ``write_url``, and they had
    drifted: RF→prod while ARIMA/XGB→dev).
    """
    from utils.exit_handler import safe_exit
    url, _, _ = get_jdbc_options(config_path)
    properties = {"driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver"}
    try:
        sdf = dataframe if isinstance(dataframe, DataFrame) else spark.createDataFrame(dataframe)
        sdf.write.jdbc(url=f"{url};trustServerCertificate=true", table=table,
                       mode=mode, properties=properties)
        logger.info(f"✅ Wrote {len(dataframe)} rows to {table}")
    except Exception:
        logger.info(f"🚫 Failed to write to {table}")
        safe_exit("E0000", f"Failed to write to {table}")


def read_sql_query(query: str, spark: SparkSession, config_path: str = "config.yaml") -> DataFrame:
    jdbc_url, user, password = get_jdbc_options(config_path)

    try:
        df = (
            spark.read.format("jdbc")
            .option("url", jdbc_url)
            .option("query", query)
            .load()
        )
        logging.info("✅ Query executed using Spark JDBC.")
        return df
    except Exception as e:
        logging.error(f"❌ Query failed: {e}")
        raise
