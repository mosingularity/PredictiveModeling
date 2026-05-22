from db.utilities import load_yaml_config, logger


class CategoryValidator:
    def __init__(self):
        self._allowed = set(
            load_yaml_config().get("profiling", {}).get("categories", [])
        )

    def validate(self, category: str):
        if category not in self._allowed:
            logger.warning(f"[Profiler] ⚠️ Unknown profiling category: '{category}'. Allowed: {sorted(self._allowed)}")
            # Optional: raise ValueError(...) instead
            return False
        return True

category_validator = CategoryValidator()
