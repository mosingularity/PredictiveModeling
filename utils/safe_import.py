import logging
from py4j.protocol import Py4JNetworkError
from socket import error as SocketError, timeout as SocketTimeout

def safe_import(import_fn, logger: logging.Logger):
    try:
        return import_fn()
    except (ConnectionResetError, SocketError, SocketTimeout) as e:
        logger.error(f"❌ Connection error: {type(e).__name__} — {str(e)}")
    except Py4JNetworkError as e:
        logger.error("🔥 Py4JNetworkError during Spark operation", exc_info=True)
    except Exception as e:
        logger.error("🚨 Unexpected import error", exc_info=True)
    return None
