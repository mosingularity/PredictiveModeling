from typing import List, Dict, Optional, Any
from pyspark.sql import DataFrame, SparkSession
import pandas as pd
import json
import logging
from itertools import chain, combinations

logging.basicConfig(level=logging.INFO)


def extract_metadata(df: DataFrame) -> Dict[str, Any]:
    """
    Extracts forecasting metadata from Spark DataFrame.
    Assumes a single-row DataFrame from a config source.
    """
    try:
        first_row = df.limit(1).collect()[0]
        metadata = {
            "forecast_method": first_row["Method"],
            "parameters": first_row["Parameters"],
            "ufm_id": first_row["UserForecastMethodID"],
            "start_date": first_row["StartDate"],
            "end_date": first_row["EndDate"],
            "databrick_id": first_row["DatabrickID"],
        }
        logging.info(f"🔍 Metadata extracted: {metadata}")
        return metadata
    except Exception as e:
        logging.error(f"❌ Failed to extract metadata: {e}")
        return {}


def parse_json_column(df, column_name, key=None):
    if column_name not in df.columns:
        logging.warning(f"Column '{column_name}' does not exist.")
        return []

    extracted = []
    for row in df[column_name].dropna():
        try:
            parsed = json.loads(row)
            if key:
                value = parsed.get(key, [])
            else:
                value = [v for entry in parsed.items() for v in entry[1]] if isinstance(parsed, dict) else []
            extracted.extend(value)
        except json.JSONDecodeError as e:
            logging.warning(f"Skipping invalid JSON: {e}")
            continue

    return list(set([v for v in extracted if v]))

def generate_combinations(columns=None) -> Dict[frozenset, List[str]]:
    """
    Generate all non-empty combinations of prediction columns, mapping each to a reporting structure.
    """
    if columns is None:
        columns = ["PeakConsumption", "StandardConsumption", "OffPeakConsumption", "Block1Consumption",
                   "Block2Consumption", "Block3Consumption", "Block4Consumption", "NonTOUConsumption"]
    combo_map = {
        frozenset(c): ['ReportingMonth', 'CustomerID'] + list(c)
        for r in range(1, len(columns) + 1)
        for c in combinations(columns, r)
    }
    logging.info(f"Generated {len(combo_map)} column combinations.")
    # type help(generate_combinations)
    return combo_map


def find_matching_combination(combos: Dict[frozenset, List[str]], target_columns = None ) -> Optional[List[str]]:
    """
    Find the best matching key from combinations mapping.
    Tries for exact/full set match first.
    """
    if target_columns is None:
        target_columns = ["PeakConsumption", "StandardConsumption", "OffPeakConsumption", "Block1Consumption",
                   "Block2Consumption", "Block3Consumption", "Block4Consumption", "NonTOUConsumption"]


    target_set = frozenset(target_columns)

    for key in combos.keys():
        if key == target_set:
            logging.info(f"Exact match found for: {target_set}")
            return combos[key]

    logging.warning(f"No exact match found for: {target_set}")
    return None