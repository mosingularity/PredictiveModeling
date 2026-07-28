"""Grouping grammar + in-figure pill plumbing — the shared figure skeleton.

Every figure family in this pack stands on the same grammar: the x axis is the
forecast month, facet columns are ``TariffType``, and each forecasting ``model``
is its own coloured trace. Two further dimensions — ``consumption_type`` and the
aggregation *level* (per-entity vs. bottom-up group sum) — are exposed as
in-figure pills.

The layer never fits a model and never reconciles: the only aggregation is a
plain bottom-up sum of ``y`` / ``y_hat`` across entities by month within a group
key (:func:`aggregate_bottom_up`). Figure plans 03–06 only author traces on top
of :func:`build_base`.

Pill semantics
--------------
Every pill-reachable state is *pre-materialized* as hidden traces at build time;
a pill click only flips trace visibility (no on-click compute). Each pill row is
one Plotly ``updatemenu``. Selecting a value on one pill resets the *other* pill
dimensions to their default (first) value — the honest cost of stateless,
pre-built buttons. Tariff on/off is handled separately and natively by the
legend via ``legendgroup=TariffType``.
"""
from typing import Dict, List, Optional, Sequence

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Tidy timestamp column. In the tidy contract this is ``ds``; conceptually it is
# the ReportingMonth the forecast falls on.
X_COL = "ds"

# The group key a bottom-up sum collapses entities within. EntityID is summed
# out; everything else identifies the resulting group series.
GROUP_KEYS = ["TariffType", "consumption_type", "model", "param_set_id", X_COL]

# Aggregation levels exposed as pill state.
LEVEL_ENTITY = "entity"
LEVEL_GROUP = "group"
DEFAULT_LEVELS = (LEVEL_ENTITY, LEVEL_GROUP)

# House colours: stable per model so a model reads the same across every figure.
_MODEL_COLOURS = {
    "ARIMA": "#1f77b4",
    "SARIMA": "#ff7f0e",
    "RandomForest": "#2ca02c",
    "RF": "#2ca02c",
    "XGBoost": "#d62728",
    "XGB": "#d62728",
}
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def model_colour(model: str, index: int = 0) -> str:
    """Stable colour for a model — house map first, palette fallback by index.

    Shared by every figure family so a model reads the same colour across the
    whole pack.
    """
    return _MODEL_COLOURS.get(model, _PALETTE[index % len(_PALETTE)])


# Backwards-compatible private alias.
_model_colour = model_colour


def aggregate_bottom_up(df: pd.DataFrame, level: str = LEVEL_ENTITY) -> pd.DataFrame:
    """Bottom-up sum of a tidy frame to the requested aggregation level.

    ``"entity"`` returns the frame unchanged (one series per entity). ``"group"``
    sums ``y`` / ``y_hat`` / CI bounds across entities by :data:`GROUP_KEYS`
    (month within tariff·consumption·model·param-set). This is a plain groupby
    sum — no fitting, no reconciliation; member forecasts simply add up.

    NaN-preserving: an all-NaN group (e.g. ``y`` actuals absent for the whole
    group) sums to NaN rather than a misleading 0, via ``min_count=1``.
    """
    if level == LEVEL_ENTITY:
        return df.copy()
    if level != LEVEL_GROUP:
        raise ValueError(f"unknown aggregation level: {level!r}")

    def _sum(series: pd.Series) -> float:
        return series.sum(min_count=1)

    grouped = df.groupby(GROUP_KEYS, dropna=False)
    agg = grouped.agg(
        y=("y", _sum),
        y_hat=("y_hat", _sum),
        y_hat_lower=("y_hat_lower", _sum),
        y_hat_upper=("y_hat_upper", _sum),
        n_entities=("EntityID", "nunique"),
    ).reset_index()
    agg["EntityID"] = "ALL"
    agg["EntityType"] = "Group"
    return agg


def _state_frame(df: pd.DataFrame, consumption_type: str, level: str) -> pd.DataFrame:
    """Slice to one consumption_type and aggregate to one level."""
    sliced = df[df["consumption_type"] == consumption_type]
    return aggregate_bottom_up(sliced, level)


def build_base(
    df: pd.DataFrame,
    *,
    value_col: str = "y_hat",
    consumption_types: Optional[Sequence[str]] = None,
    levels: Sequence[str] = DEFAULT_LEVELS,
    models: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """Build the faceted base figure with every pill state pre-materialized.

    x = ReportingMonth (``ds``), facet column = ``TariffType``, one coloured
    trace per ``model``. Every ``(consumption_type, level)`` combination is built
    as hidden traces tagged on ``trace.meta`` so :func:`add_pills` can toggle them
    without recomputing. Only the default state (first consumption_type, first
    level) is visible initially. Tariff on/off rides the legend via
    ``legendgroup=TariffType``.

    Returns a ``go.Figure`` ready for :func:`add_pills` and :func:`render`.
    """
    if df.empty:
        raise ValueError("cannot build a figure from an empty tidy frame")

    consumption_types = list(consumption_types or sorted(df["consumption_type"].dropna().unique()))
    levels = list(levels)
    tariffs = sorted(df["TariffType"].dropna().unique())
    all_models = list(models or sorted(df["model"].dropna().unique()))
    colour = {m: _model_colour(m, i) for i, m in enumerate(all_models)}

    fig = make_subplots(
        rows=1,
        cols=len(tariffs),
        subplot_titles=tariffs,
        shared_yaxes=False,
        horizontal_spacing=0.05,
    )

    default_ct, default_level = consumption_types[0], levels[0]
    seen_legend = set()  # one legend entry per model

    for ct in consumption_types:
        for level in levels:
            state = _state_frame(df, ct, level)
            is_default = (ct == default_ct and level == default_level)
            for col_idx, tariff in enumerate(tariffs, start=1):
                facet = state[state["TariffType"] == tariff]
                for model in all_models:
                    series = (
                        facet[facet["model"] == model]
                        .sort_values(X_COL)
                    )
                    if series.empty:
                        continue
                    show_legend = is_default and model not in seen_legend
                    if show_legend:
                        seen_legend.add(model)
                    fig.add_trace(
                        go.Scatter(
                            x=series[X_COL],
                            y=series[value_col],
                            mode="lines+markers",
                            name=model,
                            legendgroup=tariff,
                            legendgrouptitle_text=tariff,
                            line=dict(color=colour[model]),
                            visible=is_default,
                            showlegend=show_legend,
                            meta={"consumption_type": ct, "level": level, "model": model},
                            hovertemplate=(
                                f"{tariff} · {model} · {ct}<br>"
                                "%{x|%Y-%m}: %{y:.2f}<extra></extra>"
                            ),
                        ),
                        row=1,
                        col=col_idx,
                    )

    fig.update_layout(
        title=title or "Forecasts by TariffType",
        legend=dict(groupclick="toggleitem"),
        template="plotly_white",
    )
    return fig


def add_pills(
    fig: go.Figure,
    *,
    dimensions: Sequence[str] = ("consumption_type", "level"),
    labels: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """Attach in-figure pill rows (``updatemenus``) over pre-materialized traces.

    One ``updatemenu`` per dimension in ``dimensions`` (read from ``trace.meta``).
    Each button flips the full ``visible`` mask — a trace is shown iff its value
    for the button's dimension matches and its *other* pill dimensions sit at
    their default (first) value. No on-click compute; all states already exist.

    Idempotent on layout: replaces any pills previously added by this function.
    """
    labels = labels or {"consumption_type": "Consumption", "level": "Aggregation"}
    metas = [dict(tr.meta) for tr in fig.data]

    # Default (first-seen) value per dimension — the co-dimension a button holds.
    defaults: Dict[str, object] = {}
    ordered_values: Dict[str, List[object]] = {}
    for dim in dimensions:
        seen: List[object] = []
        for m in metas:
            v = m.get(dim)
            if v not in seen:
                seen.append(v)
        ordered_values[dim] = seen
        defaults[dim] = seen[0] if seen else None

    menus = []
    for row, dim in enumerate(dimensions):
        buttons = []
        for value in ordered_values[dim]:
            visible = []
            for m in metas:
                ok = m.get(dim) == value
                for other in dimensions:
                    if other != dim:
                        ok = ok and (m.get(other) == defaults[other])
                visible.append(bool(ok))
            buttons.append(
                dict(
                    label=str(value),
                    method="update",
                    args=[{"visible": visible}],
                )
            )
        menus.append(
            dict(
                type="buttons",
                direction="right",
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.18 + row * 0.10,
                yanchor="top",
                pad={"r": 6, "t": 4},
                buttons=buttons,
                name=labels.get(dim, dim),
            )
        )

    fig.update_layout(updatemenus=menus)
    return fig
