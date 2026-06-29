from dataclasses import dataclass, field
import numpy as np
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import math

@dataclass
class ModelPodPerformance:
    ModelName: str
    CustomerID: str
    PodID: str
    DataBrickID: Optional[int]
    UserForecastMethodID: int
    # A mapping from consumption_type -> {'RMSE': float, 'R2': float}
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_row(self) -> Dict[str, Any]:
        """
        Flatten this dataclass into a single dict suitable for pd.DataFrame:
          - metadata fields as‐is
          - dynamic RMSE_<type> and R2_<type> columns
          - RMSE_Avg and R2_Avg over the values present (appear first)
        """
        # First calculate averages
        rmse_vals, r2_vals = [], []
        metrics_fields = {}
        for ctype, m in self.metrics.items():
            rmse = m.get("RMSE")
            r2 = m.get("R2")
            metrics_fields[f"RMSE_{ctype}"] = rmse
            metrics_fields[f"R2_{ctype}"] = r2
            if rmse is not None and not math.isnan(rmse):
                rmse_vals.append(rmse)
            if r2 is not None and not math.isnan(r2):
                r2_vals.append(r2)

        rmse_avg = sum(rmse_vals) / len(rmse_vals) if rmse_vals else None
        r2_avg = sum(r2_vals) / len(r2_vals) if r2_vals else None

        # Now build the dict with desired order
        row = {
            "CustomerID": self.CustomerID,
            "PodID": self.PodID,
            "DataBrickID": self.DataBrickID,
            "UserForecastMethodID": self.UserForecastMethodID,
            "ModelName": self.ModelName,
            "RMSE_Avg": rmse_avg,
            "R2_Avg": r2_avg,
        }

        # Append the dynamic metric fields at the end
        row.update(metrics_fields)

        return row


import pandas as pd
from typing import Optional, List

import pandas as pd
from typing import Dict, List, Any, Union

def build_forecast_df(
    forecast_map: Dict[str, pd.Series],
    customer_id: str,
    pod_id: str,
    cons_types: List[str],
    user_forecast_method_id: int
) -> pd.DataFrame:
    """
    Build the per-pod forecast DataFrame by inferring the date axis
    directly from the first Series in forecast_map.

    Parameters:
    - forecast_map: mapping from consumption_type to pd.Series (indexed by date)
    - customer_id, pod_id: identifiers to stamp on every row
    - cons_types: list of all consumption types (columns) you expect
    - user_forecast_method_id: UFMID to attach

    Returns a DataFrame with columns:
      ['ForecastDate','CustomerID','PodID', *cons_types, 'UserForecastMethodID']
    """
    if not forecast_map:
        raise ValueError("forecast_map is empty — nothing to build dates from.")

    # 1) Derive the date axis from any one of the series
    first_series = next(iter(forecast_map.values()))
    forecast_dates = list(first_series.index)
    n_periods = len(forecast_dates)

    # 2) Seed your dict with the metadata columns
    data: Dict[str, Union[List[Any], pd.Series]] = {
        "PodID": [pod_id] * n_periods,
        "UserForecastMethodID": [user_forecast_method_id] * n_periods,
        "CustomerID": [customer_id] * n_periods,
        "ReportingMonth": forecast_dates,
    }

    # 3) Fill in each consumption column
    for ct in cons_types:
        ser = forecast_map.get(ct)
        if isinstance(ser, pd.Series):
            data[ct] = list(ser.values)
        else:
            data[ct] = [None] * n_periods

    # 4) Build the DataFrame
    df = pd.DataFrame(data)

    # df['CustomerID'] = df['CustomerID'].astype('int64')

    # 5) Round the consumption columns and fill missing with zero
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
class CustomerPerformanceData:
    customer_id: str
    columns: List[str]
    pod_by_id_performance: List[PodIDPerformanceData] = field(default_factory=list)

    def get_pod_performance_data(self) -> pd.DataFrame:
        dataframes = [
            pod_data.performance_data_frame
            for pod_data in self.pod_by_id_performance
            if hasattr(pod_data, 'performance_data_frame')
        ]
        # Combine all DataFrames into one.
        combined_df = pd.concat(dataframes, ignore_index=True)
        return combined_df

    def convert_pod_id_performance_data(self, df: pd.DataFrame, consumption_filter: List[str] = None) -> pd.DataFrame:
        """
        Convert a wide-format performance DataFrame into a long-format DataFrame, filtering by specified consumption types.

        The input DataFrame is expected to contain at least the following columns:
            - 'pod_id'
            - 'customer_id'
            - 'consumption_type'
            - Metric columns: 'RMSE', 'MAE', 'R2', 'forecast'

        The function performs the following steps:
            1. Creates a pivot table using 'pod_id' and 'customer_id' as the index and the
               'consumption_type' values as columns. Metric values are aggregated using the first occurrence.
            2. Stacks the consumption type level to convert the DataFrame from wide to long format.
            3. Filters the long-format DataFrame to retain only rows where the consumption type is in the
               provided `consumption_filter`.

        Parameters:
            df (pd.DataFrame): The input DataFrame in long format with required columns.
            consumption_filter (List[str], optional): A list of allowed consumption types to retain.
                Defaults to ['PeakConsumption', 'StandardConsumption', 'OffPeakConsumption'].

        Returns:
            pd.DataFrame: A long-format DataFrame where each row corresponds to a unique combination
                          of (pod_id, customer_id, consumption_type) and contains the corresponding metric values.
        """
        if consumption_filter is None:
            consumption_filter = ['PeakConsumption', 'StandardConsumption', 'OffPeakConsumption']

        # Pivot the DataFrame: Each (pod_id, customer_id) pair becomes a unique index,
        # with consumption types as columns for each metric.
        pod_df = df.pivot_table(
            index=['pod_id', 'customer_id'],
            columns='consumption_type',
            values=['RMSE', 'MAE', 'R2', 'forecast'],
            aggfunc='first'  # In case of duplicates, take the first occurrence.
        )

        # Stack the consumption type level to convert the wide format into a long format.
        pod_pf = pod_df.stack(level=1).reset_index()

        # Filter rows to include only the allowed consumption types.
        pod_filtered = pod_pf[pod_pf['consumption_type'].isin(consumption_filter)]

        return pod_filtered

@dataclass
class PredictionUnit:
    entity_id: str
    entity_type: str        # "POD" | "Combo" | "CSA"
    tariff_type: str        # "LPU" | "SPU" | "PPU"
    customer_id: str
    tariff_id: int
    series: pd.DataFrame    # indexed by ReportingMonth, sorted ascending

@dataclass
class EntityPerformanceData:
    entity_id: str
    entity_type: str
    tariff_type: str
    customer_id: str
    forecast_method_name: str
    user_forecast_method_id: int
    performance_data_frame: pd.DataFrame
    tariff_id: int = 0

@dataclass
class UnbundledResults:
    forecast_method_name: str
    entity_performance: List[EntityPerformanceData] = field(default_factory=list)

    def get_performance_data(self) -> pd.DataFrame:
        # Stamp the entity identifiers onto every output row. The per-entity
        # performance frame is keyed by pod_id/consumption_type only; EntityID,
        # EntityType, TariffType and TariffID live on the wrapper, so inject them
        # here (the single flattening point) to satisfy DoD item 8.
        frames = []
        for e in self.entity_performance:
            pdf = e.performance_data_frame
            if pdf is None or len(pdf) == 0:
                continue
            pdf = pdf.assign(
                EntityID=e.entity_id,
                EntityType=e.entity_type,
                TariffType=e.tariff_type,
                TariffID=e.tariff_id,
            )
            frames.append(pdf)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def to_forecast_fact(self) -> pd.DataFrame:
        """Mold the entity forecasts into the ForecastFact write-shape WITHOUT writing.

        Reuses the SAME molder the bundled writer uses (:func:`build_forecast_df`): one
        row per ReportingMonth, a column per consumption type, keyed by
        PodID/CustomerID/UserForecastMethodID. EntityID maps to PodID (a real POD for
        LPU; a synthetic Combo/CSA composite for SPU/PPU), and the entity-keyed identity
        (EntityID/EntityType/TariffType/TariffID) is carried alongside so the mapping
        stays visible. This is a display/validation preview of what the bundled writer
        would persist — the unbundled path itself performs no DB writes. A per-entity
        molding failure is skipped rather than fatal.
        """
        frames = []
        for e in self.entity_performance:
            pdf = e.performance_data_frame
            if pdf is None or len(pdf) == 0 or "consumption_type" not in pdf.columns:
                continue
            try:
                forecast_map = pdf.set_index("consumption_type")["forecast"].to_dict()
                wide = build_forecast_df(
                    forecast_map,
                    customer_id=e.customer_id,
                    pod_id=e.entity_id,
                    cons_types=list(forecast_map.keys()),
                    user_forecast_method_id=e.user_forecast_method_id,
                ).assign(
                    EntityID=e.entity_id,
                    EntityType=e.entity_type,
                    TariffType=e.tariff_type,
                    TariffID=e.tariff_id,
                )
                frames.append(wide)
            except Exception:
                continue
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()