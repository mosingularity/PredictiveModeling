"""Shared driver for the unbundled (entity-keyed) forecasting path.

The per-model ``forecast_<model>_unbundled`` functions were byte-identical apart
from the per-entity dispatch call, so the loop lives here once and each model
passes its own ``forecast_for_entity``.
"""
import logging
import os

import pandas as pd

from data.dml import convert_to_pandas
from db.queries import get_unbundled_predictive_data
from evaluation.performance import PredictionUnit, ForecastResults
from validation.run_summary import RunSummary
from validation.series import validate_series

logger = logging.getLogger(__name__)


def _save_unbundled_input(data: pd.DataFrame, ufmid, bundled: bool = False) -> None:
    """Optionally save the fetched input frame to disk, for gradio/QA re-use.

    Gated on ``SAVE_UNBUNDLED_FIXTURE``: a truthy flag writes
    ``data/fixtures/Results_<UFMID>.csv``; a path-like value is used verbatim. This
    saves the run's *input* rows, not a forecast, so the no-DB-write contract holds.
    ``BundledInd`` is stamped in because the fetched frame doesn't carry it, and a
    saved fixture with no mode would silently default to unbundled on replay.
    """
    save = os.getenv("SAVE_UNBUNDLED_FIXTURE")
    if not save:
        return
    data = data.copy()
    data.insert(1, "BundledInd", int(bool(bundled)))
    path = save if ("/" in save or save.endswith((".csv", ".parquet"))) \
        else f"data/fixtures/Results_{ufmid}.csv"
    if path.endswith(".parquet"):
        data.to_parquet(path, index=False)
    else:
        data.to_csv(path, index=False)
    logger.info(f"💾 Saved unbundled input frame ({len(data)} rows, UFMID {ufmid}) "
                f"→ {path} [input fixture, not a forecast write].")


def run_unbundled(model, spark, forecast_for_entity) -> ForecastResults:
    """Group the predictive data by PodID, validate each pod's series, and
    dispatch to the model's ``forecast_for_entity``.

    Resilient by pod: a series that fails validation is skipped, and a single
    pod whose forecast raises is logged and skipped without aborting the run.
    Outcomes are tallied into one summary line (see :class:`RunSummary`). The
    returned results contain only successfully-forecast pods — failed/skipped
    pods produce no output rows.
    """
    ufm_config = model.dataset.ufm_config
    data = get_unbundled_predictive_data(spark, ufm_config.user_forecast_method_id)
    data = convert_to_pandas(data)
    bundled = getattr(ufm_config, "bundled", False)
    _save_unbundled_input(data, ufm_config.user_forecast_method_id, bundled)
    results = ForecastResults(forecast_method_name=ufm_config.forecast_method_name)
    summary = RunSummary(f"unbundled {ufm_config.forecast_method_name}")
    for pod_id, group in data.groupby("PodID"):
        series = group.set_index("ReportingMonth").sort_index()
        series.index = pd.to_datetime(series.index)
        tariff_type = group["TariffType"].iloc[0] if "TariffType" in group.columns else "Consumption"
        customer_id = str(group["CustomerID"].iloc[0]) if "CustomerID" in group.columns else ""
        tariff_id = group["TariffID"].iloc[0] if "TariffID" in group.columns else 0
        unit = PredictionUnit(
            entity_id=pod_id,
            entity_type="",
            tariff_type=tariff_type,
            customer_id=customer_id,
            tariff_id=tariff_id,
            series=series,
        )
        ok, reason = validate_series(unit, ufm_config.forecast_method_name)
        if not ok:
            summary.record_skip(reason, pod_id)
            continue
        try:
            entity = forecast_for_entity(unit, ufm_config, model)
            results.entity_performance.append(entity)
            summary.record_ok(reason)
        except Exception as exc:  # backstop: one pod must not sink the whole run
            summary.record_failure(exc, pod_id)
    summary.log()
    return results
