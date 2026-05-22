import re
import warnings
import sys
import traceback

from db.utilities import clean_text


def log_warning(message, category, filename, lineno, file=None, line=None):
    from db.error_logger import insert_profiling_error
    tb = f"{filename}:{lineno} - {category.__name__}: {message}"
    print(f"[Warning Intercepted] {tb}")  # Optional console output

    insert_profiling_error(
        log_id=None,  # Or use latest profiling log ID if available
        error=clean_error_message(str(message)),
        traceback=clean_error_message(tb),
        error_type=category.__name__,
        severity="low",
        component="global_warning_handler"
    )

def log_uncaught_exception(exc_type, exc_value, exc_tb):
    from db.error_logger import insert_profiling_error
    error_msg = str(exc_value)
    tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))

    print(f"[Uncaught Exception] {error_msg}")  # Optional console output

    insert_profiling_error(
        log_id=None,
        error=clean_error_message(error_msg),
        traceback=clean_error_message(tb_str),
        error_type=exc_type.__name__,
        severity="critical",
        component="global_exception_hook"
    )

def init_error_hooks():
    # Warnings
    warnings.showwarning = log_warning
    # Uncaught Exceptions
    sys.excepthook = log_uncaught_exception

def clean_error_message(message):
    # Remove non-printable or non-ASCII characters
    return re.sub(r'[^\x20-\x7E]+', '', message)