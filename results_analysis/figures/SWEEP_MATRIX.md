# Sweep Matrix — reproducible OFAT `param_set_id`s

The hyperparameter panels ([sweep.py](sweep.py)) read the tidy table only. A
swept variant is identified by its `param_set_id`, and **by convention a
variant's `param_set_id` is the raw `model_parameters` string the runners
parse** — the same encoding consumed by
`data.dml.extract_{random_forest,xgboost,sarimax}_params` via
`hyperparameters.get_model_hyperparameters`. So a variant is fully reproducible
from the tidy table alone: parse the `param_set_id`, feed it to the matching
runner, and you regenerate that forecast.

The canonical source of this table is `SWEEP_MATRIX` in [sweep.py](sweep.py); this
document mirrors it for humans and records *which knob each model sweeps*.

## Encodings (knob order = position in the `param_set_id` string)

| model | `param_set_id` encoding | baseline (OFAT centre) | swept knob |
|-------|-------------------------|------------------------|------------|
| **RandomForest** | `(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap)` | `(100,10,2,1,5,True)` | `n_estimators` (alt: `max_depth`) |
| **XGBoost** | `(n_estimators, max_depth, learning_rate, subsample, colsample_bytree)` | `(100,5,0.1,0.8,0.8)` | `learning_rate` (alt: `max_depth`) |
| **ARIMA** | `(p,d,q)` | `(3,3,3)` | `p` (alt: `d`, `q`) |
| **SARIMA** | `(p,d,q)(P,D,Q,s)` | `(1,1,1)(1,1,1,4)` | seasonal `P` (alt: `D`, `Q`) |

Notes:
- The XGBoost baseline fills the `learning_rate` slot (the plan's `_`) at `0.1`,
  the value the variants vary around.
- `max_features` (RF) accepts `sqrt` / `log2` / `None` / int / float; `bootstrap`
  accepts `true`/`false`. The parallel-coordinates panel encodes such
  categorical/boolean knobs onto coded axes with readable tick labels.
- The fan **auto-detects** the swept knob as the single knob whose value varies
  across the present variants; the `swept` column above is the documented intent
  and the fallback when detection is ambiguous.

## Producing a sweep slice

Each OFAT variant is one full forecast run with one knob moved off the baseline,
the rest held fixed, stamped with `param_set_id = "<that param string>"`. The
fixture generator [tests/fixtures/make_sweep_fixtures.py](../../tests/fixtures/make_sweep_fixtures.py)
emits `tidy_sweep.parquet` covering ~5 variants per model this way.

```bash
python -m tests.fixtures.make_sweep_fixtures
python -m results_analysis.figures.sweep --model RF --slice tests/fixtures/tidy_sweep.parquet --out /tmp/sweep_rf.html
```
