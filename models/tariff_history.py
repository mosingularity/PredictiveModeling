import pandas as pd


def assemble_entity_series(pod_id: str, data: pd.DataFrame) -> pd.DataFrame:
    """
    Return all rows for pod_id from data, sorted by ReportingMonth ascending.

    Treatment decision: uses complete POD-level history regardless of TariffType
    changes — the current tariff is determined by the most recent row, not
    per-period. This ensures entities that switched tariff type mid-window are
    fitted on their full consumption history rather than only the post-change
    period, preventing under-fitted models.
    """
    series = data[data["PodID"] == pod_id].sort_values("ReportingMonth")
    return series.reset_index(drop=True)
