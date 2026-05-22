# config_service.py
"""
Singleton accessor for environment + profiling configuration.
Avoids repeated YAML reads or environment detection.
"""

import os
import traceback

import yaml
from typing import Tuple, Dict
from threading import Lock

from db.error_logger import insert_profiling_error
from profiler.errors.utils import get_error_metadata
from utils.exit_handler import safe_exit

# Internal cache
_cached_env_config: Tuple[str, Dict] = None
_full_config: Dict = None
_config_lock = Lock()

def load_yaml_config(path="config.yaml") -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_cached_config(config_path="config.yaml") -> Tuple[str, Dict]:
    global _cached_env_config, _full_config
    if _cached_env_config is None or _full_config is None:
        with _config_lock:
            if _cached_env_config is None or _full_config is None:
                try:
                    _full_config = load_yaml_config(config_path)
                    env = os.getenv("ENV", "LOCAL").upper()
                    if env not in _full_config:
                        raise KeyError(f"Missing environment: {env}")
                    base_config = _full_config[env]
                    if env == "LOCAL" and "datasource" in base_config.get("profiling_cfg", {}):
                        datasource = base_config["profiling_cfg"]["datasource"]
                        base_config["datasource_config"] = _full_config.get(datasource, {})
                    _cached_env_config = (env, base_config)
                except Exception as e:
                    meta = get_error_metadata("ConfigMissing", {"field": str(e)})
                    insert_profiling_error(
                        log_id=None,
                        error=meta["message"],
                        traceback=traceback.format_exc(),
                        error_type="ConfigMissing",
                        severity=meta["severity"],
                        component=meta["component"]
                    )
                    safe_exit(meta["code"], meta["message"])
    return _cached_env_config


def get_active_env() -> str:
    return get_cached_config()[0]

def get_env_config() -> Dict:
    return get_cached_config()[1]

def get_profiling_engine() -> str:
    return get_env_config().get("profiling_cfg", {}).get("engine", "mariadb")

def get_global_config() -> Dict:
    if _full_config is None:
        get_cached_config()  # trigger cache
    return _full_config

def get_global_value(key: str, default=None):
    return get_global_config().get(key, default)

def is_feature_engineering_enabled() -> bool:
    return bool(get_global_value("use_feature_engineering", False))

def is_profiling_enabled() -> bool:
    return bool(get_global_value("profiling", {}).get("enabled", False))

def get_all_categories() -> list:
    return get_global_value("categories", [])



def get_datasource_config() -> Dict:
    env, cfg = get_cached_config()

    # Only LOCAL delegates to a datasource block like DEV/QA
    if env == "LOCAL":
        return cfg.get("datasource_config", {})

    # DEV and QA return their own config blocks
    return cfg
