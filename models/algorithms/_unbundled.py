"""Shared driver for the unbundled (entity-keyed) forecasting path.

The per-model ``forecast_<model>_unbundled`` functions were byte-identical apart
from the per-entity dispatch call, so the loop lives here once and each model
passes its own ``forecast_for_entity``.
"""
import logging

from data.dml import convert_to_pandas
from db.queries import get_unbundled_predictive_data
from evaluation.performance import PredictionUnit, UnbundledResults
from validation.run_summary import RunSummary
from validation.series import validate_series

logger = logging.getLogger(__name__)


def run_unbundled(model, spark, forecast_for_entity) -> UnbundledResults:
    """Group the predictive data by (TariffType, EntityID), validate each
    entity's series, and dispatch to the model's ``forecast_for_entity``.

    Resilient by entity: a series that fails validation is skipped, and a single
    entity whose forecast raises is logged and skipped without aborting the run.
    Outcomes are tallied into one summary line (see :class:`RunSummary`). The
    returned results contain only successfully-forecast entities — failed/skipped
    entities produce no output rows.
    """
    ufm_config = model.dataset.ufm_config
    data = get_unbundled_predictive_data(spark, ufm_config.user_forecast_method_id)
    # get_unbundled_predictive_data yields Spark from the DB and pandas from the
    # fixture path; the per-entity groupby below is pandas-only, so normalise first.
    data = convert_to_pandas(data)
    results = UnbundledResults(forecast_method_name=ufm_config.forecast_method_name)
    summary = RunSummary(f"unbundled {ufm_config.forecast_method_name}")
    for (tariff_type, entity_id), group in data.groupby(["TariffType", "EntityID"]):
        series = group.set_index("ReportingMonth").sort_index()
        if "PodID" not in series.columns:
            series = series.copy()
            series["PodID"] = entity_id
        # Entity-keyed exports carry EntityType/TariffID/CustomerID as reporting
        # metadata only; default each when a source omits it rather than KeyError.
        unit = PredictionUnit(
            entity_id=entity_id,
            entity_type=group["EntityType"].iloc[0] if "EntityType" in group.columns else "",
            tariff_type=tariff_type,
            customer_id=str(group["CustomerID"].iloc[0]) if "CustomerID" in group.columns else "",
            tariff_id=group["TariffID"].iloc[0] if "TariffID" in group.columns else 0,
            series=series,
        )
        ok, reason = validate_series(unit, ufm_config.forecast_method_name)
        if not ok:
            summary.record_skip(reason, entity_id)
            continue
        try:
            results.entity_performance.append(forecast_for_entity(unit, ufm_config, model))
            summary.record_ok(reason)
        except Exception as exc:  # backstop: one entity must not sink the whole run
            summary.record_failure(exc, entity_id)
    summary.log()
    return results
