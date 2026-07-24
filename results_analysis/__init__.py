"""results_analysis — a downstream, consume-only results-analysis layer.

The layer never fits a model: it reads the tidy forecast contract only. The
single coupling point to the generate side (the four ESF-* runners) is
:mod:`results_analysis.tidy`, which adapts an ``ForecastResults`` into the one
tidy long-format table the dashboard figures read
(:mod:`results_analysis.figures.dashboard`).
"""

from results_analysis.tidy import (
    TIDY_COLUMNS,
    REQUIRED_COLUMNS,
    OPTIONAL_COLUMNS,
    to_tidy,
    validate_tidy,
)
from results_analysis.grammar import (
    aggregate_bottom_up,
    model_colour,
    LEVEL_ENTITY,
    LEVEL_GROUP,
)

__all__ = [
    "TIDY_COLUMNS",
    "REQUIRED_COLUMNS",
    "OPTIONAL_COLUMNS",
    "to_tidy",
    "validate_tidy",
    "aggregate_bottom_up",
    "model_colour",
    "LEVEL_ENTITY",
    "LEVEL_GROUP",
]
