import logging

def insert_profiling_error(*, log_id, error, traceback, error_type, severity, component):
    from profiler.profiler_switch import profiling_switch
    if not profiling_switch.log_errors:
        logging.info(f"[Profiling OFF] Suppressed error log: {error_type} | {error}")
        return

    try:
        from db.queries import insert_profiling_error as real_insert
        real_insert(
            log_id=log_id,
            error=error,
            traceback=traceback,
            error_type=error_type,
            severity=severity,
            component=component
        )
    except Exception as e:
        logging.warning(f"[Profiling] Failed to insert error log: {e}")
