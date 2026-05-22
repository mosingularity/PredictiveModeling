# hyperparameters.py

import logging
from typing import Any, Dict
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


def load_hyperparameter_grid(model_name: str, cfg: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Load the hyperparameter grid for the specified model from config.yaml.

    Args:
        model_name (str): The model name ("xgboost" or "randomforest").
        cfg (Dict[str, Any], optional): An optional configuration dictionary.

    Returns:
        Dict[str, Any]: The hyperparameter grid.
    """
    if cfg is None:
        cfg = load_yaml_config("config.yaml")
    key = f"{model_name.lower()}_grid"
    grid = cfg.get(key)
    if grid is None:
        # Fallback to default values
        if model_name.lower() == "xgboost":
            grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 1.0],
                'random_state': [42]
            }
        elif model_name.lower() == "randomforest":
            grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 4, 6],
                'min_samples_leaf': [1, 2, 4],
                'random_state': [42]
            }
        else:
            raise ValueError(f"No grid defined for model {model_name}")
        logger.info(f"💡 Using fallback grid for {model_name}: {grid}")
    else:
        logger.info(f"💡 Loaded {model_name} hyperparameter grid from config: {grid}")
    return grid

def get_cv_config() -> Dict[str, Any]:
    """
    Load the cross-validation configuration from config.yaml.

    Returns:
        Dict[str, Any]: A dictionary with keys: n_splits, shuffle, random_state.
    """
    config = load_yaml_config("config.yaml")
    cv_config = config.get("cv_config", {"n_splits": 3, "shuffle": True, "random_state": 42})
    logger.info(f"💡 Loaded CV configuration: {cv_config}")
    return cv_config


def get_pipeline_config(model_name: str) -> Dict[str, Any]:
    """
    Load the pipeline configuration for the given model from config.yaml.

    Args:
        model_name (str): The model name (e.g., "randomforest" or "xgboost").

    Returns:
        Dict[str, Any]: The pipeline configuration dictionary.
    """
    config = load_yaml_config("config.yaml")
    key = f"pipeline_config_{model_name.lower()}"
    pipeline_config = config.get(key)
    if pipeline_config is None:
        # Fallback defaults
        if model_name.lower() == "randomforest":
            pipeline_config = {
                "imputer_strategy": "mean",
                "rfecv": {"step": 0.1, "cv": 3, "scoring": "neg_mean_absolute_error"}
            }
        elif model_name.lower() == "xgboost":
            pipeline_config = {
                "imputer_strategy": "mean",
                "rfecv": {"step": 0.1, "cv": 3, "scoring": "neg_mean_absolute_error"}
            }
        else:
            pipeline_config = {"imputer_strategy": "mean"}
        logger.info(f"💡 Using fallback pipeline configuration for {model_name}: {pipeline_config}")
    else:
        logger.info(f"💡 Loaded {model_name} pipeline configuration: {pipeline_config}")
    return pipeline_config
