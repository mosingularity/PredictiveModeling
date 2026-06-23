import yaml
import os
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_catalog_cache = None

def load_error_catalog(path=None):
    global _catalog_cache
    if _catalog_cache is None:
        # Resolve relative to this module so it works regardless of cwd.
        full_path = path or os.path.join(os.path.dirname(__file__), "error_catalog.yml")
        with open(full_path, "r", encoding="utf-8") as f:
            _catalog_cache = yaml.safe_load(f)
    return _catalog_cache
