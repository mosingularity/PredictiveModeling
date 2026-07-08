"""Forecast dashboard — a local Gradio shell over the real production code.

The interaction layer (the only legitimately-new code): two tabs over the engine
and the figure builders. Nothing forecasts or scores here — it caches and
displays what ``forecasting.engine`` (real ``forecast_for_entity``) and
``results_analysis.figures.dashboard`` produce.

* **Forecast tab** — pick TariffType → scenario (filters the PodID picker, via the
  real validator) → PodID → channel. Each model has a config control and a Train
  button: a cached config draws instantly; a new one trains in the background
  (≤~5 min) with a toast when ready. CI ribbon is a toggle. Solid = in-sample
  fit, dotted = forecast.
* **Evaluation tab** — the runners' real RMSE/MAE/R², per-entity distribution.

The state/cache/train logic below is gradio-free and unit-tested; ``build_app``
imports gradio lazily so this module loads (and its logic runs) without it.
``python -m forecasting.dashboard_app`` launches the app (needs ``pip install gradio``).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go

from forecasting.engine import ALL_MODELS, DEFAULT_PARAMS, run_forecasts
from results_analysis.figures.dashboard import (
    build_forecast_figure,
    build_metrics_figure,
    entity_scenarios,
)
from validation.series import LEGACY_CONSUMPTION_COLUMNS


# ── state + cache (gradio-free, testable) ────────────────────────────────────────


@dataclass
class DashboardState:
    """Loaded data, the per-entity scenario index, and the retrieve-or-train cache."""
    raw_df: pd.DataFrame
    horizon_months: int = 12
    scenarios_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    # cache: (model, param_str) -> tidy frame (all entities for that config)
    cache: Dict[Tuple[str, str], pd.DataFrame] = field(default_factory=dict)

    def __post_init__(self):
        if self.scenarios_df.empty:
            self.scenarios_df = entity_scenarios(self.raw_df)


def live_channels(state: DashboardState) -> List[str]:
    """Consumption columns the dataset actually uses (any entity)."""
    return [c for c in LEGACY_CONSUMPTION_COLUMNS
            if c in state.raw_df.columns and (state.raw_df[c].fillna(0) != 0).any()]


def channels_for(state: DashboardState, pod: str) -> List[str]:
    """Channels populated for THIS entity, most-used first.

    SPU entities barely use Peak/OffPeak (their consumption is in Standard/NonTOU),
    so a global channel list would default the view to an all-zero series. Offer
    only what this PodID actually has, busiest channel first.
    """
    g = state.raw_df[state.raw_df["EntityID"] == pod]
    if g.empty:
        return live_channels(state)
    counts = {c: int((g[c].fillna(0) != 0).sum())
              for c in LEGACY_CONSUMPTION_COLUMNS if c in g.columns}
    live = sorted((c for c, n in counts.items() if n > 0), key=lambda c: -counts[c])
    return live or live_channels(state)


def tariffs(state: DashboardState) -> List[str]:
    return sorted(state.raw_df["TariffType"].dropna().unique())


def scenarios(state: DashboardState) -> List[str]:
    return ["(any)"] + sorted(state.scenarios_df["scenario"].dropna().unique())


def scenarios_for(state: DashboardState, tariff: str) -> List[str]:
    """Scenarios that actually occur within a tariff — so the picker never offers
    a condition that yields zero PodIDs (e.g. SPU has no happy_path entities)."""
    s = state.scenarios_df[state.scenarios_df["TariffType"] == tariff]
    return ["(any)"] + sorted(s["scenario"].dropna().unique())


def pods_for(state: DashboardState, tariff: str, scenario: str = "(any)") -> List[str]:
    """PodIDs in a tariff, optionally filtered to those whose data is in a scenario."""
    s = state.scenarios_df
    s = s[s["TariffType"] == tariff]
    if scenario and scenario != "(any)":
        s = s[s["scenario"] == scenario]
    return sorted(s["EntityID"].unique())


def is_trained(state: DashboardState, model: str, param: str) -> bool:
    return (model, param) in state.cache


def config_choices(state: DashboardState, model: str) -> List[str]:
    """Param strings already trained for a model (the browseable configs)."""
    return sorted(p for (m, p) in state.cache if m == model)


def train_config(state: DashboardState, model: str, param: str) -> pd.DataFrame:
    """Run the real engine for one (model, param) across all entities; cache it."""
    tidy = run_forecasts(state.raw_df, {model: param}, horizon_months=state.horizon_months)
    state.cache[(model, param)] = tidy
    return tidy


def actuals_for(state: DashboardState, pod: str, channel: str) -> pd.Series:
    """The observed series for one PodID·channel, from the raw input."""
    g = state.raw_df[state.raw_df["EntityID"] == pod]
    if g.empty or channel not in g.columns:
        return pd.Series(dtype="float64")
    s = g.groupby("ReportingMonth")[channel].sum(min_count=1).sort_index()
    s.index = pd.DatetimeIndex(s.index)
    return s


def subject_tidy(state: DashboardState, pod: str, channel: str,
                 model_params: Dict[str, str]) -> pd.DataFrame:
    """Cached forecast rows for one PodID·channel across the chosen per-model configs."""
    frames = []
    for model, param in model_params.items():
        t = state.cache.get((model, param))
        if t is None:
            continue
        frames.append(t[(t["EntityID"] == pod) & (t["consumption_type"] == channel)])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def forecast_view(state: DashboardState, pod: str, channel: str, ci: bool,
                  model_params: Dict[str, str], title: Optional[str] = None) -> go.Figure:
    """Assemble the Tab-1 figure for the current selection (instant from cache)."""
    tidy = subject_tidy(state, pod, channel, model_params)
    if tidy.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white",
                          title="No trained config for this selection — click Train",
                          height=520)
        return fig
    trained = [m for m, p in model_params.items() if is_trained(state, m, p)]
    return build_forecast_figure(tidy, actuals_for(state, pod, channel),
                                 title=title or f"{pod} · {channel}", show_ci=ci,
                                 models=[m for m in ALL_MODELS if m in trained])


def metrics_view(state: DashboardState, metric: str = "RMSE") -> go.Figure:
    """Assemble the Tab-2 figure over everything trained so far."""
    if not state.cache:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title="Train a config to see metrics", height=480)
        return fig
    allt = pd.concat(list(state.cache.values()), ignore_index=True)
    try:
        return build_metrics_figure(allt, metric=metric)
    except ValueError as e:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title=str(e), height=480)
        return fig


# ── gradio shell (lazy import) ───────────────────────────────────────────────────


def _render_fn(state):
    """Change-handler that redraws the forecast plot from current widget values."""
    def fn(tariff, scenario, pod, channel, ci, *params):
        return forecast_view(state, pod, channel, ci, dict(zip(ALL_MODELS, params)))
    return fn


def _train_fn(state, model):
    """Train-button handler: background train (toast) + redraw + refresh this model's configs."""
    import gradio as gr

    def fn(tariff, scenario, pod, channel, ci, *params):
        model_params = dict(zip(ALL_MODELS, params))
        param = model_params[model]
        if not is_trained(state, model, param):
            gr.Info(f"Training {model} {param} … (up to a few minutes)")
            train_config(state, model, param)
            gr.Info(f"{model} {param} ready")
        fig = forecast_view(state, pod, channel, ci, model_params)
        return fig, gr.update(choices=config_choices(state, model), value=param)
    return fn


def build_app(state: DashboardState):
    """Build the Gradio Blocks app over ``state`` (imports gradio lazily).

    All components are created first, then wired — so Train buttons can target the
    plot (which is defined after them in the layout).
    """
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
            inputs = [tariff, scenario, pod, channel, ci] + [param_inputs[m] for m in ALL_MODELS]
            render = _render_fn(state)

            def _chan_update(pod):
                chans = channels_for(state, pod) if pod else []
                return gr.update(choices=chans, value=chans[0] if chans else None)

            def _on_tariff(tariff):
                # New tariff: its scenarios (reset to any), its pods, the pod's channels.
                scen = scenarios_for(state, tariff)
                pods = pods_for(state, tariff, "(any)")
                pod0 = pods[0] if pods else None
                return (gr.update(choices=scen, value="(any)"),
                        gr.update(choices=pods, value=pod0), _chan_update(pod0))

            def _on_scenario(tariff, scenario):
                pods = pods_for(state, tariff, scenario)
                pod0 = pods[0] if pods else None
                return gr.update(choices=pods, value=pod0), _chan_update(pod0)

            tariff.change(_on_tariff, tariff, [scenario, pod, channel]).then(render, inputs, plot)
            scenario.change(_on_scenario, [tariff, scenario], [pod, channel]).then(render, inputs, plot)
            pod.change(_chan_update, pod, channel).then(render, inputs, plot)
            for ctrl in [channel, ci] + [param_inputs[m] for m in ALL_MODELS]:
                ctrl.change(render, inputs, plot)
            for model in ALL_MODELS:
                train_btns[model].click(_train_fn(state, model), inputs, [plot, param_inputs[model]])
            app.load(render, inputs, plot)

        with gr.Tab("Evaluation"):
            metric = gr.Radio(["RMSE", "MAE", "R2"], value="RMSE", label="Metric")
            mplot = gr.Plot(label="Per-entity distribution")
            metric.change(lambda m: metrics_view(state, m), metric, mplot)
            app.load(lambda: metrics_view(state, "RMSE"), None, mplot)

    return app


def load_input(data_path: str) -> pd.DataFrame:
    """Load the dashboard input — parquet or CSV (BOM-tolerant).

    Accepts both the legacy entity-keyed frames and the PodID contract
    (``dbo.PredictiveInputData`` shape, e.g. ``data/fixtures/Results.csv``): the
    dashboard's internal key column is EntityID, so a PodID-only frame gets its
    PodID mirrored in. ReportingMonth is parsed up front (CSV loads it as str).
    """
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
    state = DashboardState(raw_df=load_input(data_path), horizon_months=horizon)
    app = build_app(state)
    # queue() → Train runs as a background job; gr.Info shows the completion toast.
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
