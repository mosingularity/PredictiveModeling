import logging

logger = logging.getLogger(__name__)


def report_validation_error(*, error_type, error, severity="medium",
                            component="validation", log_id=None, traceback=""):
    """Log a validation failure.

    Replaces the former ``insert_profiling_error`` profiling-DB sink. The keyword
    signature is kept compatible with the old call sites (``log_id``/``traceback``
    are accepted and ignored) so validation reporting now goes to the application
    log instead of a SQL Server table — no profiling subsystem required.
    """
    logger.warning("[%s] %s (severity=%s, component=%s)",
                   error_type, error, severity, component)
