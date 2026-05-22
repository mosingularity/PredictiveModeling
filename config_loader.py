import yaml
from dataclasses import dataclass
from typing import List

@dataclass
class HyperParameterConfig:
    selected_columns: List[str]
    consumption_types: List[str]
    mode: str
    debug: bool
    log: bool
    use_feature_engineering: bool
    train_mode: str
    fail_on_invalid_lags: bool
    use_extended_calendar_features: bool
    add_calendar_features: bool
    lag: List[int]
    visualize: bool
    tables: dict
    write_url: str
    save: bool
    user: str
    password: str

    def get(self, key: str, default=None):
        return getattr(self, key, default)

def load_config(path: str = 'config.yaml') -> HyperParameterConfig:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    # Filter out keys used for the forecasting pipeline (ignore environment definitions)
    config_keys = ["selected_columns", "consumption_types", "mode", "debug", "log",
                   "use_feature_engineering","train_mode", "fail_on_invalid_lags",
                   "use_extended_calendar_features", "add_calendar_features", "lag",
                   "visualize", "tables", "write_url", "save",  "user", "password"]
    filtered_data = { key: data[key] for key in config_keys if key in data }
    return HyperParameterConfig(**filtered_data)
