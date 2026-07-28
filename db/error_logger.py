"""Log validation failures immediately or as a grouped end-of-run summary."""

import logging
import threading

logger = logging.getLogger(__name__)

# Keep concurrent dashboard runs from collecting each other's warnings.
_local = threading.local()


def _current_collector():
    return getattr(_local, "buffer", None)


class _ValidationErrorCollector:
    """Collect validation errors by type and emit one summary."""

    def __init__(self, label):
        self.label = label
        self.groups = {}

    def add(self, error_type, message, entity_id, severity, component):
        group = self.groups.setdefault(
            error_type,
            {
                "count": 0,
                "entities": [],
                "example": message,
                "severity": severity,
                "component": component,
            },
        )
        group["count"] += 1
        if entity_id and entity_id not in group["entities"]:
            group["entities"].append(str(entity_id))

    def emit(self):
        if not self.groups:
            return
        total = sum(group["count"] for group in self.groups.values())
        logger.warning(
            "⚠️ %s — %d validation warning(s) across %d type(s):",
            self.label,
            total,
            len(self.groups),
        )
        for error_type, group in self.groups.items():
            entities = group["entities"]
            if entities:
                shown = ", ".join(entities[:12])
                if len(entities) > 12:
                    shown += f", … (+{len(entities) - 12} more)"
                subject = f"{len(entities)} entity(ies): {shown}"
            else:
                subject = "no entity recorded"
            logger.warning(
                "   • [%s] ×%d (severity=%s, component=%s) — %s",
                error_type,
                group["count"],
                group["severity"],
                group["component"],
                subject,
            )
            logger.warning("     e.g. %s", group["example"])


class collect_validation_errors:
    """Collect validation warnings within a block and summarize them on exit."""

    def __init__(self, label="run"):
        self.label = label
        self._owner = False

    def __enter__(self):
        if _current_collector() is None:
            _local.buffer = _ValidationErrorCollector(self.label)
            self._owner = True
        return _current_collector()

    def __exit__(self, _exc_type, _exc, _traceback):
        if self._owner:
            collector = _current_collector()
            _local.buffer = None
            if collector is not None:
                collector.emit()
        return False


def report_validation_error(
    *,
    error_type,
    error,
    severity="medium",
    component="validation",
    log_id=None,
    traceback="",
    entity_id=None,
):
    """Log a validation failure or add it to the active collector.

    ``log_id`` and ``traceback`` remain accepted for compatibility with old callers.
    """
    collector = _current_collector()
    if collector is not None:
        collector.add(error_type, error, entity_id, severity, component)
        logger.debug("[%s] %s (entity=%s)", error_type, error, entity_id)
        return
    logger.warning(
        "[%s] %s (severity=%s, component=%s)",
        error_type,
        error,
        severity,
        component,
    )
