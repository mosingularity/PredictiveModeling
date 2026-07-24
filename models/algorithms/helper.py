import numpy as np
import pandas as pd


def _apply_log(series: pd.Series, log: bool) -> pd.Series:
    """Log-transform the series (small constant added for zero values)."""
    if log:
        return np.log(series + 1e-6)
    return series


def _collect_metrics(pod_id, customer_id, consumption_type, forecast, metrics=None,
                     baseline_metrics=None, in_sample=None, validation_reason=None):
    """One result row: forecast, its in-sample fit (for the fitted-history overlay),
    and its metrics. A skipped or failed series carries no metrics, so those columns
    are NaN, not a fake 0.0 that would read as a perfect score. ``validation_reason``
    names why a series fell back to zero (gap, too short, all-zero, fit failed);
    it is None on the scored path.
    """
    row = {'pod_id': pod_id, 'customer_id': customer_id, 'consumption_type': consumption_type,
           'forecast': forecast, 'in_sample': in_sample, 'validation_reason': validation_reason}
    for metric in ['RMSE', 'MAE', 'R2']:
        row[metric] = (metrics or {}).get(metric, float("nan"))
        row[f'{metric}_baseline'] = (baseline_metrics or {}).get(metric, float("nan"))
    return row