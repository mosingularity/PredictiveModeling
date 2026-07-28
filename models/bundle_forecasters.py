"""Real estimators for the bundle aggregate series.

``forecast_for_bundle`` fits one model on the bundle's aggregate through an
injected ``fit_fn(bundle_series, horizon) -> forecast_series``. :func:`bundle_forecaster`
builds that ``fit_fn`` for each model family (ARIMA, SARIMA, RandomForest, XGBoost),
reusing the same fitting primitives the unbundled path uses, so there is one fitting
implementation, not two. Imports stay in this module, not ``models/bundle.py``, so the
aggregate/disaggregate math stays free of the statsmodels/sklearn dependencies.
"""
from typing import Callable

import numpy as np
import pandas as pd

from hyperparameters import get_model_hyperparameters
from models.algorithms.autoarima import fit_time_series_model
from models.algorithms.tree_algorithms.helper import (
    engineer_data, train_rf, train_xgb, recursive_forecast,
)

# Base lag/window recipe mirrors forecast_for_pod_id (the per-entity tree fitter) so
# the bundle's engineered features match the unbundled path's; yearly terms are added
# out to the horizon below.
_BASE_LAGS = [1, 2, 3, 6]
_BASE_WINDOWS = [3, 6]


def _future_months(series: pd.Series, horizon: int) -> pd.DatetimeIndex:
    """``horizon`` month-start timestamps following the series' last observed month.

    Matches ``models/bundle._future_index`` so a real forecaster and the naive default
    index their forecasts identically.
    """
    return pd.date_range(series.index[-1], periods=horizon + 1, freq="MS")[1:]


def _arima_forecast(series: pd.Series, horizon: int, *, order, seasonal_order, log: bool) -> pd.Series:
    """Fit ARIMA/SARIMA on the aggregate and forecast ``horizon`` months ahead."""
    model = fit_time_series_model(series, order, seasonal_order, log=log)
    forecast = model.get_forecast(steps=horizon).predicted_mean
    if log:
        forecast = np.exp(forecast)
    forecast.index = _future_months(series, horizon)
    return forecast


def _tree_forecast(series: pd.Series, horizon: int, *, params, train: Callable) -> pd.Series:
    """Fit a tree model on the aggregate via the shared engineer→train→recurse path.

    ``train`` is the trainer (:func:`train_rf` or :func:`train_xgb`) and ``params`` its
    hyperparameter tuple. The lag/window recipe (base + yearly out to the horizon)
    mirrors ``forecast_for_pod_id`` so the bundle's features match the unbundled path.
    """
    future = _future_months(series, horizon)
    df = series.to_frame(name="value")
    year_terms = [12 * i for i in range(1, (horizon // 12) + 2)]
    lags = _BASE_LAGS + year_terms
    windows = _BASE_WINDOWS + year_terms
    df, feature_cols, stl_obj, history = engineer_data(df, "value", lags, windows)
    model = train(df[feature_cols], df["deseasoned"], params)
    fc_df = recursive_forecast(history, stl_obj, model, feature_cols,
                               future[0], future[-1], lags, windows)
    forecast = fc_df["forecast"]
    forecast.index = future
    return forecast


def bundle_forecaster(method: str, *, model_parameters: str = "", log: bool = False
                      ) -> Callable[[pd.Series, int], pd.Series]:
    """Build the ``fit_fn(bundle_series, horizon)`` that fits ``method`` on the aggregate.

    ``method`` is the UFM's model name (ARIMA / SARIMA / RandomForest / XGBoost).
    Hyperparameters come from ``model_parameters``, falling back to config.yaml
    defaults when empty, the same resolution the unbundled path uses.

    Production wires this from the UFM config::

        bundle_forecaster(ufm_config.forecast_method_name,
                          model_parameters=ufm_config.model_parameters,
                          log=forecast_model.config.log)
    """
    key = method.lower().replace(" ", "")
    if key in ("arima", "sarima"):
        order, seasonal_order = get_model_hyperparameters(key, model_parameters)
        return lambda s, h: _arima_forecast(s, h, order=order,
                                            seasonal_order=seasonal_order, log=log)
    trainers = {"randomforest": train_rf, "xgboost": train_xgb}
    if key in trainers:
        params = get_model_hyperparameters(key, model_parameters)
        train = trainers[key]
        return lambda s, h: _tree_forecast(s, h, params=params, train=train)
    raise ValueError(f"Unknown bundle forecast method: {method!r}")
