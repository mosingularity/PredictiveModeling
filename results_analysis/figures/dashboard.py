"""Dashboard figures — sleek, and every number sourced from production code.

Two builders for the two tabs, plus the entity-picker scenario index. None of
them recompute anything: forecasts and the real ``RMSE/MAE/R²`` arrive via
:func:`results_analysis.tidy.to_tidy` (fed by ``forecasting.engine``, which calls
the real ``forecast_for_entity``), actuals come from the raw input frame, and the
scenario marks come from the real ``validation.series.validate_series``.

* :func:`build_forecast_figure` — Tab 1. Actual (solid) + each model's in-sample
  fit (solid) and future forecast (dotted), legend carrying the model's real
  metrics, an optional CI ribbon on the forecast.
* :func:`build_metrics_figure` — Tab 2. The runners' ``RMSE`` as a per-entity
  distribution, faceted by TariffType (within-scale), with the median marked.
* :func:`entity_scenarios` — the per-entity data-condition marks that filter the
  picker (a PodID *has* a gap; you navigate to it).
"""
from types import SimpleNamespace
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go

from results_analysis.grammar import X_COL, model_colour
from results_analysis.tidy import validate_tidy
from validation.series import CONSUMPTION_COLUMNS, classify_reason

_ACTUAL_COLOUR = "#222222"


# ── Tab 1 — forecast ─────────────────────────────────────────────────────────────


def _carried_metrics(rows: pd.DataFrame) -> Optional[Dict[str, float]]:
    """The model's real RMSE/MAE/R² (carried from the run), or None if absent."""
    out = {}
    for m in ("RMSE", "MAE", "R2"):
        if m in rows.columns:
            vals = rows[m].dropna()
            if len(vals):
                out[m] = float(vals.iloc[0])
    return out if len(out) == 3 else None


def _carried_reason(rows: pd.DataFrame) -> Optional[str]:
    """Why an unscored model got the zero-fallback (gap / too short / all-zero /
    fit failed), carried from the runner via validation_reason, or None."""
    if "validation_reason" in rows.columns:
        vals = rows["validation_reason"].dropna()
        if len(vals):
            return str(vals.iloc[0])
    return None


def _model_label(model: str, metrics: Optional[Dict[str, float]]) -> str:
    if not metrics:
        return f"{model} · (unscored)"
    return f"{model} · RMSE {metrics['RMSE']:.1f} · MAE {metrics['MAE']:.1f} · R² {metrics['R2']:.2f}"


def build_forecast_figure(
    tidy: pd.DataFrame,
    actuals: Optional[pd.Series] = None,
    *,
    title: Optional[str] = None,
    show_ci: bool = False,
    models: Optional[List[str]] = None,
) -> go.Figure:
    """One subject's forecast view: actual + each model (solid fit, dotted forecast).

    ``tidy`` is already sliced to one subject + consumption channel (rows carry
    ``model``/``ds``/``y_hat``/``is_forecast`` and the real metric columns).
    ``actuals`` is the observed series (from the raw input), drawn as the solid
    reference line. Solid = in-sample (``is_forecast=False``); dotted = forecast.
    """
    tidy = validate_tidy(tidy)
    fig = go.Figure()

    if actuals is not None and len(actuals.dropna()):
        a = actuals.dropna().sort_index()
        # Reindex to the full month span so a missing month is a NaN → a visible
        # hole (connectgaps=False), not a line silently bridged across the gap.
        full = pd.date_range(a.index.min(), a.index.max(), freq="MS")
        a = a.reindex(full)
        fig.add_trace(go.Scatter(
            x=a.index, y=a.to_numpy(), mode="lines+markers", name="actual",
            line=dict(color=_ACTUAL_COLOUR, width=2), connectgaps=False,
            hovertemplate="actual: %{y:.1f}<extra></extra>",
        ))

    all_models = models or sorted(tidy["model"].dropna().unique())
    colour = {m: model_colour(m, i) for i, m in enumerate(all_models)}
    is_fc = tidy["is_forecast"].astype("boolean").fillna(True)

    unscored = []  # (model, reason) for fits with NaN metrics = the zero-fallback
    for model in all_models:
        rows = tidy[tidy["model"] == model]
        if rows.empty:
            continue
        metrics = _carried_metrics(rows)
        if metrics is None:
            # A failed/skipped fit returns a degenerate zero series with no score —
            # don't draw it as a confident forecast; flag it (with the why) instead.
            unscored.append((model, _carried_reason(rows)))
            continue
        clr = colour[model]
        label = _model_label(model, metrics)
        in_sample = rows[~is_fc.loc[rows.index]].sort_values(X_COL)
        forecast = rows[is_fc.loc[rows.index]].sort_values(X_COL)
        # In-sample fit — solid, legend entry carries the metrics.
        if not in_sample.empty:
            fig.add_trace(go.Scatter(
                x=in_sample[X_COL], y=in_sample["y_hat"], mode="lines",
                name=label, legendgroup=model, line=dict(color=clr, width=2),
                showlegend=True, connectgaps=False,
                hovertemplate=f"{model} fit: %{{y:.1f}}<extra></extra>",
            ))
        # Forecast — dotted; share the legend group so the legend toggles both.
        if not forecast.empty:
            fig.add_trace(go.Scatter(
                x=forecast[X_COL], y=forecast["y_hat"], mode="lines",
                name=label, legendgroup=model, line=dict(color=clr, width=2, dash="dot"),
                showlegend=in_sample.empty, connectgaps=False,
                hovertemplate=f"{model} forecast: %{{y:.1f}}<extra></extra>",
            ))
            if show_ci:
                _add_ci(fig, forecast, clr, model)

    if unscored:
        parts = [f"{m}: {reason}" if reason else f"{m}: unscored" for m, reason in unscored]
        fig.add_annotation(
            x=0.0, xref="paper", xanchor="left", y=1.06, yref="paper", yanchor="bottom",
            showarrow=False, text="⚠ no forecast — " + " · ".join(parts),
            font=dict(size=12, color="#b00020"),
        )

    fig.update_layout(
        title=dict(text=title or "Forecast", x=0.5, xanchor="center"),
        template="plotly_white", hovermode="x unified",
        xaxis_title="ReportingMonth", yaxis_title="Consumption",
        legend=dict(yanchor="top", y=1.0, x=1.02),
        height=520, margin=dict(t=70, r=240, l=70, b=60),
    )
    return fig


def _add_ci(fig, forecast, clr, model):
    """Shaded confidence ribbon on the forecast, where bounds are present."""
    band = forecast.dropna(subset=["y_hat_lower", "y_hat_upper"])
    if band.empty:
        return
    h = clr.lstrip("#")
    rgba = f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},0.15)"
    fig.add_trace(go.Scatter(
        x=band[X_COL], y=band["y_hat_lower"], mode="lines", line=dict(width=0),
        legendgroup=model, showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=band[X_COL], y=band["y_hat_upper"], mode="lines", fill="tonexty",
        fillcolor=rgba, line=dict(width=0), legendgroup=model, showlegend=False, hoverinfo="skip",
    ))


# ── Tab 2 — evaluation ───────────────────────────────────────────────────────────


def build_metrics_figure(tidy: pd.DataFrame, metric: str = "RMSE") -> go.Figure:
    """Median bar per model + the individual entity points, faceted by TariffType.

    Uses the carried ``RMSE``/``MAE``/``R²`` (one value per entity·model — the
    numbers your runners scored). With only a few entities per group a box implies
    a distribution that isn't there, so this shows a **median bar** for the
    at-a-glance comparison with the **per-entity dots** overlaid for honesty about
    n. Faceted by TariffType so aggregation stays within a comparable scale. No
    recomputation, no MASE/RMSSE.

    Note: R² is unbounded below — a few catastrophic entities (R² ≪ 0 on the short
    backtest) will dominate the axis. Prefer RMSE/MAE for comparison; read R² for
    its sign (< 0 = worse than the mean baseline).
    """
    tidy = validate_tidy(tidy)
    if metric not in tidy.columns:
        raise ValueError(f"unknown metric {metric!r}")
    # One metric value per (entity, tariff, consumption, model).
    per_entity = (tidy.dropna(subset=[metric])
                  .groupby(["TariffType", "consumption_type", "EntityID", "model"], dropna=False)[metric]
                  .first().reset_index())
    if per_entity.empty:
        raise ValueError(f"no {metric} values to plot")

    tariffs = sorted(per_entity["TariffType"].dropna().unique())
    models = sorted(per_entity["model"].dropna().unique())
    colour = {m: model_colour(m, i) for i, m in enumerate(models)}

    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=len(tariffs), subplot_titles=tariffs, shared_yaxes=False)
    for col, tariff in enumerate(tariffs, start=1):
        sub = per_entity[per_entity["TariffType"] == tariff]
        medians = sub.groupby("model")[metric].median().reindex(models)
        # Median bar per model (the at-a-glance comparison).
        fig.add_trace(go.Bar(
            x=models, y=medians.to_numpy(),
            marker_color=[colour[m] for m in models], opacity=0.55,
            showlegend=False, name="median",
            hovertemplate=f"%{{x}} · median {metric} %{{y:.1f}}<extra></extra>",
        ), row=1, col=col)
        # Per-entity dots overlaid (honest about how few entities there are).
        for model in models:
            vals = sub[sub["model"] == model][metric]
            if vals.empty:
                continue
            fig.add_trace(go.Scatter(
                x=[model] * len(vals), y=vals.to_numpy(), mode="markers",
                marker=dict(color="#333", size=7, line=dict(color="#fff", width=1)),
                showlegend=False, name=model,
                hovertemplate=f"{model} · entity {metric} %{{y:.1f}}<extra></extra>",
            ), row=1, col=col)
        fig.update_yaxes(title_text=f"{metric} (median bar · entity dots)", row=1, col=col)

    fig.update_layout(
        title=dict(text=f"{metric} by model — median bar + per-entity points", x=0.5, xanchor="center"),
        template="plotly_white", height=480, margin=dict(t=80, r=120, l=70, b=50),
    )
    return fig


# ── entity-picker scenario index (real validator) ───────────────────────────────


def entity_scenarios(raw_df: pd.DataFrame, method: str = "ARIMA") -> pd.DataFrame:
    """Per-entity scenario marks from the real validator — the picker's filter.

    Runs ``series_validator.validate_series`` (the production check) on each
    entity's series for ``method`` and buckets the reason into a scenario slug.
    The scenario is a property of the PodID's data; the dashboard filters the
    picker to entities carrying a chosen condition rather than toggling a mode.
    """
    from evaluation.performance import PredictionUnit
    from validation.series import validate_series

    cols = [c for c in CONSUMPTION_COLUMNS if c in raw_df.columns]
    rows = []
    for entity_id, g in raw_df.groupby("EntityID"):
        series = (g[["ReportingMonth", *cols]]
                  .groupby("ReportingMonth", as_index=True).sum(min_count=1).sort_index())
        series.index = pd.DatetimeIndex(series.index)
        unit = PredictionUnit(
            entity_id=str(entity_id), entity_type=str(g.get("EntityType", pd.Series(["POD"])).iloc[0]),
            tariff_type=str(g["TariffType"].iloc[0]), customer_id="", tariff_id=0, series=series,
        )
        ok, reason = validate_series(unit, method)
        rows.append({"EntityID": str(entity_id), "TariffType": str(g["TariffType"].iloc[0]),
                     "scenario": classify_reason(reason), "ok": ok, "validation_reason": reason})
    return pd.DataFrame(rows, columns=["EntityID", "TariffType", "scenario", "ok", "validation_reason"])
