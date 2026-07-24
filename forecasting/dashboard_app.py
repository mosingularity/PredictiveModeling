"""Serve forecast and evaluation views over cached production-model results."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go

from forecasting.engine import ALL_MODELS, DEFAULT_PARAMS, run_forecasts
from results_analysis.figures.dashboard import (
    build_bundle_figure,
    build_forecast_figure,
    build_metrics_figure,
    entity_scenarios,
)
from validation.series import LEGACY_CONSUMPTION_COLUMNS


# ── state + cache (gradio-free, testable) ────────────────────────────────────────


@dataclass
class DashboardState:
    """Dashboard data, scenario index, and forecast caches."""

    raw_df: pd.DataFrame
    horizon_months: int = 12
    scenarios_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    # (model, parameters) -> tidy frame for every entity
    cache: Dict[Tuple[str, str], pd.DataFrame] = field(default_factory=dict)
    # (model, parameters, channel) -> disaggregated member forecasts
    bundle_cache: Dict[Tuple[str, str, str], pd.DataFrame] = field(default_factory=dict)

    def __post_init__(self):
        if self.scenarios_df.empty:
            self.scenarios_df = entity_scenarios(self.raw_df)


def live_channels(state: DashboardState) -> List[str]:
    """Return consumption columns populated by any entity."""
    return [
        column
        for column in LEGACY_CONSUMPTION_COLUMNS
        if column in state.raw_df.columns
        and (state.raw_df[column].fillna(0) != 0).any()
    ]


def channels_for(state: DashboardState, pod: str) -> List[str]:
    """Return one entity's populated channels, most-used first."""
    entity_rows = state.raw_df[state.raw_df["EntityID"] == pod]
    if entity_rows.empty:
        return live_channels(state)
    counts = {
        column: int((entity_rows[column].fillna(0) != 0).sum())
        for column in LEGACY_CONSUMPTION_COLUMNS
        if column in entity_rows.columns
    }
    live = sorted(
        (column for column, count in counts.items() if count > 0),
        key=lambda column: -counts[column],
    )
    return live or live_channels(state)


def tariffs(state: DashboardState) -> List[str]:
    return sorted(state.raw_df["TariffType"].dropna().unique())


def scenarios_for(state: DashboardState, tariff: str) -> List[str]:
    """Return scenarios that occur within a tariff."""
    scenario_rows = state.scenarios_df[state.scenarios_df["TariffType"] == tariff]
    return ["(any)"] + sorted(scenario_rows["scenario"].dropna().unique())


def pods_for(state: DashboardState, tariff: str, scenario: str = "(any)") -> List[str]:
    """Return PodIDs in a tariff and optional scenario."""
    scenario_rows = state.scenarios_df
    scenario_rows = scenario_rows[scenario_rows["TariffType"] == tariff]
    if scenario and scenario != "(any)":
        scenario_rows = scenario_rows[scenario_rows["scenario"] == scenario]
    return sorted(scenario_rows["EntityID"].unique())


def is_trained(state: DashboardState, model: str, param: str) -> bool:
    return (model, param) in state.cache


def config_choices(state: DashboardState, model: str) -> List[str]:
    """Return parameter strings already trained for a model."""
    return sorted(p for (m, p) in state.cache if m == model)


def train_config(state: DashboardState, model: str, param: str) -> pd.DataFrame:
    """Run and cache one model configuration across all entities."""
    from db.error_logger import collect_validation_errors

    # Collapse per-entity validation warnings into one training summary.
    with collect_validation_errors(f"train {model} {param}"):
        tidy = run_forecasts(
            state.raw_df, {model: param}, horizon_months=state.horizon_months
        )
    state.cache[(model, param)] = tidy
    return tidy


def actuals_for(state: DashboardState, pod: str, channel: str) -> pd.Series:
    """Return the observed series for one PodID and channel."""
    entity_rows = state.raw_df[state.raw_df["EntityID"] == pod]
    if entity_rows.empty or channel not in entity_rows.columns:
        return pd.Series(dtype="float64")
    actuals = (
        entity_rows.groupby("ReportingMonth")[channel]
        .sum(min_count=1)
        .sort_index()
    )
    actuals.index = pd.DatetimeIndex(actuals.index)
    return actuals


def subject_tidy(state: DashboardState, pod: str, channel: str,
                 model_params: Dict[str, str]) -> pd.DataFrame:
    """Return cached forecast rows for one PodID and channel."""
    frames = []
    for model, param in model_params.items():
        tidy = state.cache.get((model, param))
        if tidy is None:
            continue
        matching = tidy[
            (tidy["EntityID"] == pod)
            & (tidy["consumption_type"] == channel)
        ]
        frames.append(matching)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def forecast_view(state: DashboardState, pod: str, channel: str, ci: bool,
                  model_params: Dict[str, str], title: Optional[str] = None,
                  bundled: bool = False) -> go.Figure:
    """Build the selected entity's cached forecast comparison."""
    tidy = subject_tidy(state, pod, channel, model_params)
    bundled_by_model = (
        bundled_for_pod(state, pod, channel, model_params)
        if bundled and can_bundle(state)
        else {}
    )
    if tidy.empty and not bundled_by_model:
        fig = go.Figure()
        fig.update_layout(template="plotly_white",
                          title="No trained config for this selection — click Train",
                          height=520)
        return fig
    trained = [m for m, p in model_params.items() if is_trained(state, m, p)]
    models = [m for m in ALL_MODELS if m in trained or m in bundled_by_model]
    actuals = actuals_for(state, pod, channel)
    return build_forecast_figure(
        tidy,
        actuals,
        title=title or f"{pod} · {channel}",
        show_ci=ci,
        models=models,
        bundled=bundled_by_model,
    )


def bundle_view(state: DashboardState, channel: str,
                model_params: Dict[str, str]) -> go.Figure:
    """Build the aggregate bundle and summed per-entity comparison."""
    actual, bundled, unbundled = bundle_totals(state, channel, model_params)
    member_ids = state.raw_df["PodID"].unique()
    title = f"Bundle total · {channel} ({len(member_ids)} members)"
    return build_bundle_figure(actual, bundled, unbundled, title=title)


def metrics_view(state: DashboardState, metric: str = "RMSE") -> go.Figure:
    """Build the metric figure over all trained configurations."""
    if not state.cache:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title="Train a config to see metrics", height=480)
        return fig
    cached_frames = list(state.cache.values())
    all_tidy = pd.concat(cached_frames, ignore_index=True)
    try:
        return build_metrics_figure(all_tidy, metric=metric)
    except ValueError as exc:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title=str(exc), height=480)
        return fig


# ── bundled vs unbundled comparison (gradio-free, testable) ──────────────────────


def can_bundle(state: DashboardState) -> bool:
    """Return whether the loaded frame carries the bundle path's PodID contract."""
    return "PodID" in state.raw_df.columns


def bundled_members(state: DashboardState, model: str, param: str,
                    channel: str) -> pd.DataFrame:
    """Return one cached bundle forecast disaggregated to its member PodIDs."""
    key = (model, param, channel)
    if key in state.bundle_cache:
        return state.bundle_cache[key]

    from db.error_logger import collect_validation_errors
    from models.algorithms.bundled import forecast_bundle_guarded
    from models.bundle import build_member_series
    from models.bundle_forecasters import bundle_forecaster

    with collect_validation_errors(f"bundle {model} · {channel}"):
        member_series = build_member_series(state.raw_df, channel)
        fit_fn = bundle_forecaster(model, model_parameters=param)
        member_forecasts = forecast_bundle_guarded(
            member_series,
            method=model,
            horizon=state.horizon_months,
            fit_fn=fit_fn,
            label=f"{model} · {channel}",
        )
    result = pd.DataFrame() if member_forecasts is None else member_forecasts
    state.bundle_cache[key] = result
    return result


def bundled_for_pod(state: DashboardState, pod: str, channel: str,
                    model_params: Dict[str, str]) -> Dict[str, pd.Series]:
    """Return one PodID's disaggregated bundle forecast for each model."""
    forecasts = {}
    for model, param in model_params.items():
        member_forecasts = bundled_members(state, model, param, channel)
        if not member_forecasts.empty and pod in member_forecasts.columns:
            forecasts[model] = member_forecasts[pod]
    return forecasts


def bundle_totals(state: DashboardState, channel: str,
                  model_params: Dict[str, str]) -> Tuple[pd.Series, Dict, Dict]:
    """Return actual, bundled, and summed-unbundled series for the aggregate plot."""
    from models.bundle import BUNDLE_CONSUMPTION_COLUMN, aggregate_members, build_member_series

    member_series = build_member_series(state.raw_df, channel)
    bundle_frame = aggregate_members(member_series)
    bundle_actual = bundle_frame[BUNDLE_CONSUMPTION_COLUMN]

    bundled, unbundled = {}, {}
    for model, param in model_params.items():
        member_forecasts = bundled_members(state, model, param, channel)
        if not member_forecasts.empty:
            bundled[model] = member_forecasts.sum(axis=1)
        tidy = state.cache.get((model, param))
        if tidy is not None:
            future = tidy[
                (tidy["consumption_type"] == channel)
                & tidy["is_forecast"].astype(bool)
            ]
            if not future.empty:
                unbundled[model] = (
                    future.groupby("ds")["y_hat"].sum().sort_index()
                )
    return bundle_actual, bundled, unbundled


def bundle_note(state: DashboardState, channel: str, model_params: Dict[str, str]) -> str:
    """Summarize bundled and unbundled horizon totals for each model."""
    _, bundled, unbundled = bundle_totals(state, channel, model_params)
    if not bundled:
        return ("_No bundle forecast yet — the bundle either failed validation (see the "
                "run summary in the console) or its model has not been trained._")
    notes = []
    for model, bundled_series in bundled.items():
        bundled_total = float(bundled_series.sum())
        unbundled_series = unbundled.get(model)
        if unbundled_series is None or unbundled_series.empty:
            notes.append(
                f"**{model}** bundled {bundled_total:,.0f} kWh "
                f"(train {model} to compare)"
            )
            continue
        unbundled_total = float(unbundled_series.sum())
        gap = (
            (bundled_total - unbundled_total) / unbundled_total * 100
            if unbundled_total
            else float("nan")
        )
        notes.append(
            f"**{model}** — bundled {bundled_total:,.0f} vs unbundled "
            f"{unbundled_total:,.0f} kWh (**{gap:+.2f}%**)"
        )
    return f"Horizon totals over the bundle · {channel} — " + " · ".join(notes)


# ── gradio shell (lazy import) ───────────────────────────────────────────────────


def _outputs(state, pod, channel, ci, bundled, model_params):
    """Refresh the pod plot, bundle plot, and bundle note together."""
    import gradio as gr

    fig = forecast_view(state, pod, channel, ci, model_params, bundled=bundled)
    show = bool(bundled) and can_bundle(state)
    if not show:
        return fig, gr.update(visible=False), gr.update(visible=False)
    bundle_fig = bundle_view(state, channel, model_params)
    note = bundle_note(state, channel, model_params)
    return (
        fig,
        gr.update(value=bundle_fig, visible=True),
        gr.update(value=note, visible=True),
    )


def _render_fn(state):
    """Build the widget change handler."""

    def fn(_tariff, _scenario, pod, channel, ci, bundled, *params):
        model_params = dict(zip(ALL_MODELS, params))
        return _outputs(state, pod, channel, ci, bundled, model_params)

    return fn


def _train_fn(state, model):
    """Build a train-button handler for one model."""
    import gradio as gr

    def fn(_tariff, _scenario, pod, channel, ci, bundled, *params):
        model_params = dict(zip(ALL_MODELS, params))
        param = model_params[model]
        if not is_trained(state, model, param):
            gr.Info(f"Training {model} {param} … (up to a few minutes)")
            train_config(state, model, param)
            gr.Info(f"{model} {param} ready")
        outputs = _outputs(state, pod, channel, ci, bundled, model_params)
        choices = config_choices(state, model)
        return (*outputs, gr.update(choices=choices, value=param))

    return fn


def build_app(state: DashboardState):
    """Build the Gradio Blocks app, importing Gradio only when needed."""
    import gradio as gr

    tlist = tariffs(state)
    init_pods = pods_for(state, tlist[0], "(any)") if tlist else []
    chans = channels_for(state, init_pods[0]) if init_pods else live_channels(state)

    with gr.Blocks(title="Forecast dashboard") as app:
        gr.Markdown("## Forecast dashboard — over the real model code")
        with gr.Tab("Forecast"):
            with gr.Row():
                tariff = gr.Dropdown(tlist, value=tlist[0] if tlist else None, label="TariffType")
                init_scen = scenarios_for(state, tlist[0]) if tlist else ["(any)"]
                scenario = gr.Dropdown(init_scen, value="(any)", label="Scenario")
                pod = gr.Dropdown(init_pods, value=init_pods[0] if init_pods else None, label="PodID")
                channel = gr.Dropdown(chans, value=chans[0] if chans else None, label="Consumption")
                ci = gr.Checkbox(value=False, label="CI ribbon")
                bundled = gr.Checkbox(
                    value=False, label="Bundled", interactive=can_bundle(state),
                    info=("overlay each pod's disaggregated share of the bundle forecast"
                          if can_bundle(state) else "needs a PodID dataset (Results_<UFMID>.csv)"))

            param_inputs, train_btns = {}, {}
            for model in ALL_MODELS:
                with gr.Row():
                    param_inputs[model] = gr.Dropdown(
                        config_choices(state, model) or [DEFAULT_PARAMS[model]],
                        value=DEFAULT_PARAMS[model], allow_custom_value=True,
                        label=f"{model} params", scale=4,
                    )
                    train_btns[model] = gr.Button(f"Train {model}", scale=1)

            plot = gr.Plot(label="Forecast")
            bundle_plot = gr.Plot(label="Bundle total", visible=False)
            bundle_md = gr.Markdown("", visible=False)

            inputs = ([tariff, scenario, pod, channel, ci, bundled]
                      + [param_inputs[m] for m in ALL_MODELS])
            outputs = [plot, bundle_plot, bundle_md]
            render = _render_fn(state)

            def _chan_update(pod):
                chans = channels_for(state, pod) if pod else []
                return gr.update(choices=chans, value=chans[0] if chans else None)

            def _on_tariff(tariff):
                scen = scenarios_for(state, tariff)
                pods = pods_for(state, tariff, "(any)")
                pod0 = pods[0] if pods else None
                return (gr.update(choices=scen, value="(any)"),
                        gr.update(choices=pods, value=pod0), _chan_update(pod0))

            def _on_scenario(tariff, scenario):
                pods = pods_for(state, tariff, scenario)
                pod0 = pods[0] if pods else None
                return gr.update(choices=pods, value=pod0), _chan_update(pod0)

            tariff.change(_on_tariff, tariff, [scenario, pod, channel]).then(render, inputs, outputs)
            scenario.change(_on_scenario, [tariff, scenario], [pod, channel]).then(render, inputs, outputs)
            pod.change(_chan_update, pod, channel).then(render, inputs, outputs)
            for ctrl in [channel, ci, bundled] + [param_inputs[m] for m in ALL_MODELS]:
                ctrl.change(render, inputs, outputs)
            for model in ALL_MODELS:
                train_btns[model].click(_train_fn(state, model), inputs,
                                        outputs + [param_inputs[model]])
            app.load(render, inputs, outputs)

        with gr.Tab("Evaluation"):
            metric = gr.Radio(["RMSE", "MAE", "R2"], value="RMSE", label="Metric")
            mplot = gr.Plot(label="Per-entity distribution")
            metric.change(lambda m: metrics_view(state, m), metric, mplot)
            app.load(lambda: metrics_view(state, "RMSE"), None, mplot)

    return app


def load_input(data_path: str) -> pd.DataFrame:
    """Load parquet or BOM-tolerant CSV, normalizing dates and the entity key."""
    if data_path.endswith(".csv"):
        raw = pd.read_csv(data_path, encoding="utf-8-sig")
    else:
        raw = pd.read_parquet(data_path)
    raw["ReportingMonth"] = pd.to_datetime(raw["ReportingMonth"])
    if "EntityID" not in raw.columns and "PodID" in raw.columns:
        raw = raw.assign(EntityID=raw["PodID"].astype(str))
    return raw


def main(data_path: str = "data/fixtures/unbundled_real_ermelo.parquet",
         horizon: int = 12, port: int = 7860, share: bool = False):
    import logging

    from utils.quiet_warnings import quiet_third_party_warnings
    quiet_third_party_warnings()
    # Suppress per-fit root-logger chatter while retaining warnings.
    logging.getLogger().setLevel(logging.WARNING)
    raw_df = load_input(data_path)
    state = DashboardState(raw_df=raw_df, horizon_months=horizon)
    app = build_app(state)
    app.queue().launch(server_port=port, share=share)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch the local forecast dashboard.")
    parser.add_argument("--data", default="data/fixtures/unbundled_real_ermelo.parquet",
                        help="input parquet or CSV — legacy entity-keyed or PodID contract "
                             "(e.g. data/fixtures/Results.csv)")
    parser.add_argument("--horizon", type=int, default=12, help="months to forecast ahead")
    parser.add_argument("--port", type=int, default=7860, help="localhost port")
    parser.add_argument("--share", action="store_true", help="create a public Gradio share link")
    args = parser.parse_args()
    main(args.data, args.horizon, args.port, args.share)
