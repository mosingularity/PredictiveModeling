import yaml
import os
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_catalog_cache = None

def load_error_catalog(path="profiler/errors/error_catalog.yml"):
    global _catalog_cache
    if _catalog_cache is None:
        full_path = os.path.abspath(path)
        with open(full_path, "r", encoding="utf-8") as f:  # ✅ fix here
            _catalog_cache = yaml.safe_load(f)
    return _catalog_cache
