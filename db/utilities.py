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


def get_jdbc_options(config_path: str = "config.yaml") -> Tuple[str, Dict[str, str]]:
    _, env_cfg = get_environment_config(config_path)
    user = os.getenv("DB_USER", "fortrackSQL")
    password = os.getenv("DB_PASSWORD", "")
    url = f"{env_cfg['server']};databaseName={env_cfg['database']}"
    return url, user, password


def read_sql_query(query: str, spark: SparkSession, config_path: str = "config.yaml") -> DataFrame:
    jdbc_url, user, password = get_jdbc_options(config_path)

    try:
        df = (
            spark.read.format("jdbc")
            # .option("url", jdbc_url)
            .option("url", "jdbc:sqlserver://fortrack-maz-sdb-san-dev-01.database.windows.net:1433;database=FortrackDB;Authentication=ActiveDirectoryMSI")
            .option("query", query)
            .load()
        )
        logging.info("✅ Query executed using Spark JDBC.")
        return df
    except Exception as e:
        logging.error(f"❌ Query failed: {e}")
        raise
