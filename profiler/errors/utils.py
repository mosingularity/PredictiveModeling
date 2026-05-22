from .catalog_loader import load_error_catalog
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_error_metadata(key: str, context: dict = None):
    context = context or {}
    catalog = load_error_catalog()
    entry = catalog.get(key, catalog["UnknownError"])
    try:
        message = entry["message"].format(**context)
    except Exception:
        message = entry["message"]

    return {
        "code": entry.get("code", "E9999"),
        "message": message,
        "severity": entry.get("severity", "critical"),
        "component": entry.get("component", "unknown"),
        "alert_channel": entry.get("alert_channel", None)
    }
