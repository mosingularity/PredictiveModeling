# hyperparameters.py

import logging
from typing import Any
from db.utilities import load_yaml_config
from data.dml import extract_sarimax_params, extract_random_forest_params, extract_xgboost_params

logger = logging.getLogger(__name__)


def get_model_hyperparameters(model_name: str, model_parameters: str) -> Any:
    """
    Load model hyperparameters with robust fallback for missing or incomplete parameters.
    - If model_parameters is None, empty, or incomplete (e.g., "(100)"), use defaults from config.yaml.
    - Applies only to Random Forest, XGBoost, and ARIMA/SARIMA.
    """
    import re
    config = load_yaml_config("config.yaml")
    defaults = config.get("model_defaults", {})
    default_value = defaults.get(model_name.lower())

    # Check if we should fallback for Random Forest
    if model_name.lower() == "randomforest":
        needs_fallback = (
            not model_parameters or model_parameters.strip() == "" or
            # Check if parsed parameters are incomplete (<6 values)
            len(re.findall(r'\d+|true|false|tru', model_parameters.lower())) < 6
        )
        if needs_fallback:
            logger.info(f"💡 Fallback to default Random Forest parameters: {default_value}")
            model_parameters = default_value

    # General fallback for other models
    if not model_parameters or model_parameters.strip() == "":
        if default_value is None:
            logger.error(f"🚫 No default hyperparameters provided for model {model_name}")
            raise ValueError(f"Default hyperparameters for {model_name} not found in config.yaml")
        logger.info(f"💡 Using default hyperparameters for {model_name}: {default_value}")
        model_parameters = default_value

    # Parse based on model type
    if model_name.lower() in ("sarima", "arima"):
        return extract_sarimax_params(model_parameters)
    elif model_name.lower() == "randomforest":
        return extract_random_forest_params(model_parameters)
    elif model_name.lower() == "xgboost":
        return extract_xgboost_params(model_parameters)
    else:
        raise ValueError(f"Unknown model {model_name}")
