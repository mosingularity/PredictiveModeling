"""Run-level outcome summary — keeps log output bounded on large datasets.

Per-entity validation skips and forecast failures are recorded here and emitted as
a single ``INFO`` line at the end of a run (counts per reason / per error category),
instead of one warning line per entity·channel. Per-entity detail is still available
at ``DEBUG``. This bounds output and Databricks cell/cache size to O(distinct reasons)
rather than O(entities × channels).
"""
import logging
from collections import Counter

logger = logging.getLogger(__name__)


def classify_error(exc: Exception) -> str:
    """Coarse error category for the summary (the per-entity backstop taxonomy).

    Convergence/linear-algebra failures from the model fit, data-shape problems
    (missing/renamed columns), and everything else unexpected.
    """
    if isinstance(exc, (KeyError, ValueError)):
        return "data_shape"
    name = type(exc).__name__.lower()
    if "converg" in name or "linalg" in name:
        return "convergence"
    return "unexpected"


class RunSummary:
    """Accumulate per-entity outcomes; emit one summary line via :meth:`log`."""

    def __init__(self, label: str):
        self.label = label
        self.ok = 0
        self.warned = Counter()    # non-fatal validation reason -> count
        self.skipped = Counter()   # validation reason -> count
        self.failed = Counter()    # error category -> count

    def record_ok(self, reason: str = "ok"):
        self.ok += 1
        if reason and reason != "ok":
            self.warned[reason] += 1

    def record_skip(self, reason: str, entity_id=None):
        self.skipped[reason] += 1
        logger.debug("skipped %s: %s", entity_id, reason)

    def record_failure(self, exc: Exception, entity_id=None) -> str:
        category = classify_error(exc)
        self.failed[category] += 1
        logger.debug("failed %s [%s]: %s", entity_id, category, exc, exc_info=True)
        return category

    @property
    def total(self) -> int:
        return self.ok + sum(self.skipped.values()) + sum(self.failed.values())

    def log(self) -> None:
        """Emit the one-line summary (and a DEBUG hint if anything was dropped)."""
        parts = [f"{self.ok} ok"]
        if self.warned:
            parts.append("warned " + ", ".join(f"{r}={c}" for r, c in self.warned.most_common()))
        if self.skipped:
            parts.append("skipped " + ", ".join(f"{r}={c}" for r, c in self.skipped.most_common()))
        if self.failed:
            parts.append("failed " + ", ".join(f"{c}={n}" for c, n in self.failed.most_common()))
        logger.info("📊 %s summary (%d entities): %s", self.label, self.total, " · ".join(parts))
        if self.skipped or self.failed:
            logger.info("   (set logger '%s' to DEBUG for per-entity detail)", __name__)
