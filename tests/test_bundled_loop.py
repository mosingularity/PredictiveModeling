"""Covers the bundled customer->pod loop (models/algorithms/_bundled.run_bundled).

The real bundled path writes to SQL Server via jdbc_write, so it was untestable
before the loop was extracted. With the loop isolated we can mock the Spark write
+ the conversion helpers and assert the loop structure: forecast_pod is called
once per (customer, pod), and exactly two jdbc_write calls fire (ForecastFact +
StatisticalPerformanceMetrics).
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from models.algorithms import _bundled


def _make_model(n_customers=2, n_pods=2):
    rows = [
        {"CustomerID": f"C{c}", "PodID": f"P{c}_{p}",
         "ReportingMonth": pd.Timestamp("2024-01-01"), "PeakConsumption": 1.0}
        for c in range(n_customers)
        for p in range(n_pods)
    ]
    df = pd.DataFrame(rows)
    dataset = SimpleNamespace(
        processed_df=df,
        ufm_config=SimpleNamespace(forecast_method_id=1, forecast_method_name="ARIMA"),
        extract_unique_customers_and_pods=lambda: (
            sorted(df["CustomerID"].unique()), sorted(df["PodID"].unique())
        ),
        variable_ids=None,
    )
    config = SimpleNamespace(consumption_types=["PeakConsumption"])
    return SimpleNamespace(dataset=dataset, config=config)


def test_run_bundled_calls_pod_once_per_pod_and_writes_twice():
    model = _make_model(n_customers=2, n_pods=2)
    spy = MagicMock(return_value=SimpleNamespace(performance_data_frame=pd.DataFrame()))
    with patch.object(_bundled, "jdbc_write") as mock_write, \
         patch.object(_bundled, "run_forecast_sanity_checks"), \
         patch.object(_bundled, "ensure_numeric_consumption_types", side_effect=lambda d, m: d), \
         patch.object(_bundled, "_convert_to_model_performance_row",
                      return_value=SimpleNamespace(to_row=lambda: {"x": 1})), \
         patch.object(_bundled, "_convert_forecast_map_to_df",
                      return_value=pd.DataFrame({"y": [1]})):
        _bundled.run_bundled(model, spark=None, forecast_pod=spy)

    assert spy.call_count == 4          # 2 customers x 2 pods
    assert mock_write.call_count == 2   # ForecastFact + StatisticalPerformanceMetrics
    # The two writes target the two expected tables.
    tables = {call.args[2] for call in mock_write.call_args_list}
    assert tables == {_bundled.target_table_name, _bundled.performance_metrics_table}
