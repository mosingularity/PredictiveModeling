"""Run-level driver for the bundle forecast path: the counterpart of
:func:`models.algorithms.unbundled.run_unbundled`.

Where ``run_unbundled`` forecasts each pod separately, ``run_bundled`` aggregates the
UFM's pods into one bundle per consumption channel, fits one model on each aggregate,
and disaggregates the forecast back to members by historical share. The members are
the UFM's own entity set (the PodID contract); there is no separate membership table.

Performs no DB writes. The result is a :class:`ForecastResults`, the same shape
``run_unbundled`` returns, so the existing preview machinery works unchanged.
"""
import pandas as pd

from data.dml import convert_to_pandas, get_forecast_range
from db.error_logger import report_validation_error
from db.queries import get_unbundled_predictive_data
from evaluation.performance import EntityPerformanceData, ForecastResults
from models.algorithms.helper import _collect_metrics
from models.algorithms.utilities import evaluate_predictions
from models.bundle import build_member_series, forecast_for_bundle
from models.bundle_forecasters import bundle_forecaster
from validation.run_summary import RunSummary
from validation.series import consumption_columns


def forecast_bundle_guarded(members, method, horizon, *, fit_fn=None,
                            summary=None, label=""):
    """Run one bundle's forecast; on failure, log it and every member id and
    return None instead of raising, so one bad bundle can't sink the run.

    A bundle cannot be partially disaggregated, so any failure (validation or fit)
    loses ALL of its members at once — unlike the unbundled path, where one failure
    loses one entity. Naming the members in the log is how that loss stays visible.
    """
    member_ids = [str(c) for c in members.columns]
    try:
        parts = forecast_for_bundle(members, method=method, horizon=horizon, fit_fn=fit_fn)
        if summary is not None:
            summary.record_ok()
        return parts
    except Exception as exc:
        reason = (f"{label}: " if label else "") + \
                 f"{exc} (members lost: {', '.join(member_ids)})"
        report_validation_error(
            log_id=None, error=reason, traceback="",
            error_type="BundleForecastFailure", severity="high", component="bundle")
        if summary is not None:
            if isinstance(exc, ValueError):
                summary.record_skip(reason, label or "bundle")
            else:
                summary.record_failure(exc, label or "bundle")
        return None


def _backtest_member_metrics(members, method, horizon, fit_fn):
    """Per-member RMSE/MAE/R2 from a held-out backtest of the bundle.

    Holds out the last window, refits the aggregate on the training portion, and
    scores each member's disaggregated forecast against its own held-out actuals.
    The fit is on the aggregate, so a member's metric measures the bundle's fit
    scaled by that member's share, not an independent per-member fit.

    Returns ``{member_id: {'RMSE':.., 'MAE':.., 'R2':..}}``, or an empty dict when
    the series is too short to backtest or the fit fails (metrics stay NaN, the
    same "unscored" signal the unbundled path uses).
    """
    bt = min(horizon, len(members) // 3)
    if bt < 2:
        return {}
    train, test = members.iloc[:-bt], members.iloc[-bt:]
    try:
        bt_parts = forecast_for_bundle(train, method=method, horizon=bt, fit_fn=fit_fn)
    except Exception:
        return {}
    scored = {}
    for member in members.columns:
        pred = bt_parts[member].copy()
        pred.index = test.index
        metrics, _ = evaluate_predictions(test[member], pred)
        scored[member] = metrics
    return scored


def run_bundled(model, spark) -> ForecastResults:
    """Aggregate the UFM's pods into one bundle per channel, fit once, disaggregate.

    Reads the same PodID-keyed frame the unbundled path reads, and for each channel
    aggregates, fits one model, and disaggregates back to members. Coherent by
    construction: members sum to the bundle every month, per channel.
    """
    ufm_config = model.dataset.ufm_config
    raw = get_unbundled_predictive_data(spark, ufm_config.user_forecast_method_id)
    data = convert_to_pandas(raw)
    def _passthrough(group, col, default):
        return group[col].iloc[0] if col in group.columns else default

    identity = {
        str(pod): {
            "customer_id": str(_passthrough(g, "CustomerID", "")),
            "tariff_type": _passthrough(g, "TariffType", "Consumption"),
            "tariff_id": _passthrough(g, "TariffID", 0),
        }
        for pod, g in data.groupby("PodID")
    }

    def _identity(pod_id):
        return identity.get(str(pod_id),
                            {"customer_id": "", "tariff_type": "Consumption", "tariff_id": 0})

    channels = consumption_columns(data)
    horizon = len(get_forecast_range(ufm_config))
    fit_fn = bundle_forecaster(
        ufm_config.forecast_method_name,
        model_parameters=ufm_config.model_parameters,
        log=model.config.log,
    )

    summary = RunSummary(f"bundled {ufm_config.forecast_method_name}")
    member_rows = {}  # pod_id -> [per-channel metric rows, _collect_metrics-shaped]
    for channel in channels:
        members = build_member_series(data, channel)
        parts = forecast_bundle_guarded(
            members, ufm_config.forecast_method_name, horizon,
            fit_fn=fit_fn, summary=summary, label=channel)
        if parts is None:
            continue
        metrics = _backtest_member_metrics(
            members, ufm_config.forecast_method_name, horizon, fit_fn)
        for pod_id in parts.columns:
            ident = _identity(pod_id)
            row = _collect_metrics(pod_id, ident["customer_id"], channel, parts[pod_id],
                                   metrics=metrics.get(pod_id))
            member_rows.setdefault(pod_id, []).append(row)
    summary.log()

    results = ForecastResults(forecast_method_name=ufm_config.forecast_method_name)
    for pod_id, rows in member_rows.items():
        ident = _identity(pod_id)
        member_frame = pd.DataFrame(rows)
        entity = EntityPerformanceData(
            entity_id=pod_id,
            entity_type="",
            tariff_type=ident["tariff_type"],
            customer_id=ident["customer_id"],
            forecast_method_name=ufm_config.forecast_method_name,
            user_forecast_method_id=ufm_config.user_forecast_method_id,
            performance_data_frame=member_frame,
            tariff_id=ident["tariff_id"],
        )
        results.entity_performance.append(entity)
    return results
