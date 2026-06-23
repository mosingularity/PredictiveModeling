"""Shared driver for the unbundled (entity-keyed) forecasting path.

The per-model ``forecast_<model>_unbundled`` functions were byte-identical apart
from the per-entity dispatch call, so the loop lives here once and each model
passes its own ``forecast_for_entity``.
"""
import logging

from db.error_logger import report_validation_error
from db.queries import get_predictive_data
from evaluation.performance import PredictionUnit, UnbundledResults
from models.series_validator import validate_series
from profiler.errors.utils import get_error_metadata

logger = logging.getLogger(__name__)


def run_unbundled(model, spark, forecast_for_entity) -> UnbundledResults:
    """Group the predictive data by (TariffType, EntityID), validate each
    entity's series, and dispatch to the model's ``forecast_for_entity``.
    """
    ufm_config = model.dataset.ufm_config
    data = get_predictive_data(spark, ufm_config.user_forecast_method_id)
    results = UnbundledResults(forecast_method_name=ufm_config.forecast_method_name)
    for (tariff_type, entity_id), group in data.groupby(["TariffType", "EntityID"]):
        series = group.set_index("ReportingMonth").sort_index()
        if "PodID" not in series.columns:
            series = series.copy()
            series["PodID"] = entity_id
        unit = PredictionUnit(
            entity_id=entity_id,
            entity_type=group["EntityType"].iloc[0],
            tariff_type=tariff_type,
            # Real unbundled exports (Ermelo) are entity-keyed and carry no
            # CustomerID; it is reporting metadata only, so default to "" when absent.
            customer_id=str(group["CustomerID"].iloc[0]) if "CustomerID" in group.columns else "",
            tariff_id=group["TariffID"].iloc[0],
            series=series,
        )
        ok, reason = validate_series(unit, ufm_config.forecast_method_name)
        if not ok:
            meta = get_error_metadata("SeriesValidationFailed", {"entity_id": entity_id, "reason": reason})
            report_validation_error(
                log_id=None,
                error=meta["message"],
                traceback="",
                error_type="SeriesValidationFailed",
                severity=meta["severity"],
                component=meta["component"],
            )
            continue
        if reason != "ok":
            logger.warning(f"⚠️ Series validation warning for entity {entity_id}: {reason}")
        results.entity_performance.append(forecast_for_entity(unit, ufm_config, model))
    return results
