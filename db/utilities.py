"""Configuration and Spark JDBC helpers."""

import logging
import os
from typing import Dict, Tuple

import yaml
from pyspark.sql import DataFrame, SparkSession

logging.basicConfig(level=logging.INFO)


def load_yaml_config(config_path: str = "config.yaml") -> Dict:
    with open(config_path) as config_file:
        return yaml.safe_load(config_file)


def get_environment_config(config_path: str = "config.yaml") -> Tuple[str, Dict]:
    environment = os.getenv("ENV", "QA")
    config = load_yaml_config(config_path)
    if environment not in config:
        raise ValueError(
            f"Environment '{environment}' not found in {config_path}. "
            f"Available: {list(config)}"
        )
    return environment, config[environment]


def get_jdbc_options(config_path: str = "config.yaml") -> Tuple[str, str, str]:
    _, environment_config = get_environment_config(config_path)
    user = os.getenv("DB_USER", "fortrackSQL")
    password = os.getenv("DB_PASSWORD", "")
    url = (
        f"{environment_config['server']}:1433;"
        f"database={environment_config['database']};"
        "Authentication=ActiveDirectoryMSI"
    )
    return url, user, password


def read_sql_query(
    query: str, spark: SparkSession, config_path: str = "config.yaml"
) -> DataFrame:
    jdbc_url, _, _ = get_jdbc_options(config_path)

    try:
        df = (
            spark.read.format("jdbc")
            .option("url", jdbc_url)
            .option("query", query)
            .load()
        )
        logging.info("✅ Query executed using Spark JDBC.")
        return df
    except Exception as exc:
        logging.error(f"❌ Query failed: {exc}")
        raise
