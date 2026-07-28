"""Result containers for a forecast run, and the molders that flatten them.

The forecasting paths build one ``EntityPerformanceData`` per entity (a long
per-channel metrics frame plus identity), collect them in ``ForecastResults``,
and read the run back through two views: ``get_performance_data`` (metrics,
identity stamped on every row) and ``to_forecast_fact`` (the write-shaped
forecast preview). Nothing here touches a database.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

import pandas as pd


def build_forecast_df(
    forecast_map: Dict[str, pd.Series],
    customer_id: str,
    pod_id: str,
    cons_types: List[str],
    user_forecast_method_id: int
) -> pd.DataFrame:
    """Mold one pod's per-channel forecast Series into the ForecastFact shape:
    one row per ReportingMonth, a column per consumption type, identity stamped
    on every row. The date axis comes from the first Series in ``forecast_map``;
    a missing channel becomes zeros."""
    if not forecast_map:
        raise ValueError("forecast_map is empty — nothing to build dates from.")

    first_series = next(iter(forecast_map.values()))
    forecast_dates = list(first_series.index)
    n_periods = len(forecast_dates)

    data: Dict[str, Union[List[Any], pd.Series]] = {
        "PodID": [pod_id] * n_periods,
        "UserForecastMethodID": [user_forecast_method_id] * n_periods,
        "CustomerID": [customer_id] * n_periods,
        "ReportingMonth": forecast_dates,
    }
    for ct in cons_types:
        ser = forecast_map.get(ct)
        data[ct] = list(ser.values) if isinstance(ser, pd.Series) else [None] * n_periods

    df = pd.DataFrame(data)
    for ct in cons_types:
        if ct in df.columns:
            df[ct] = df[ct].fillna(0).round(2)
    return df


@dataclass
class PodIDPerformanceData:
    pod_id: str
    forecast_method_name: str
    customer_id: str
    user_forecast_method_id: int
    performance_data_frame: pd.DataFrame


@dataclass
class PredictionUnit:
    """One entity's input to a forecast call: its identity plus its series."""
    entity_id: str          # the PodID on the PodID contract
    entity_type: str        # legacy label; "" on the PodID contract
    tariff_type: str        # passthrough metadata
    customer_id: str
    tariff_id: int
    series: pd.DataFrame    # indexed by ReportingMonth, sorted ascending


@dataclass
class EntityPerformanceData:
    """One entity's output: its identity plus the long per-channel metrics frame."""
    entity_id: str
    entity_type: str
    tariff_type: str
    customer_id: str
    forecast_method_name: str
    user_forecast_method_id: int
    performance_data_frame: pd.DataFrame
    tariff_id: int = 0


@dataclass
class ForecastResults:
    forecast_method_name: str
    entity_performance: List[EntityPerformanceData] = field(default_factory=list)

    def get_performance_data(self) -> pd.DataFrame:
        """One long frame across all entities, with each row stamped with its
        pod's identity. The per-pod frame is keyed by pod_id/consumption_type
        only; PodID, TariffType and TariffID live on the wrapper, so this is
        the single place they are injected."""
        frames = []
        for e in self.entity_performance:
            pdf = e.performance_data_frame
            if pdf is None or len(pdf) == 0:
                continue
            pdf = pdf.assign(
                PodID=e.entity_id,
                TariffType=e.tariff_type,
                TariffID=e.tariff_id,
            )
            frames.append(pdf)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def to_forecast_fact(self) -> pd.DataFrame:
        """Mold the per-pod forecasts into the ForecastFact write-shape WITHOUT
        writing — a display/validation preview of what a writer would persist.

        UserForecastMethodID comes from the run's config (the UFMID the data was
        fetched with), not from the input rows, so preview rows carry it
        consistently. A per-pod molding failure is skipped rather than fatal.
        """
        frames = []
        for e in self.entity_performance:
            pdf = e.performance_data_frame
            if pdf is None or len(pdf) == 0 or "consumption_type" not in pdf.columns:
                continue
            try:
                by_channel = pdf.set_index("consumption_type")
                forecast_map = by_channel["forecast"].to_dict()
                channels = list(forecast_map.keys())
                wide = build_forecast_df(
                    forecast_map,
                    customer_id=e.customer_id,
                    pod_id=e.entity_id,
                    cons_types=channels,
                    user_forecast_method_id=e.user_forecast_method_id,
                )
                frames.append(wide)
            except Exception:
                continue
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
