# profiler_switch.py

class ProfilingSwitch:
    def __init__(self):
        from db.utilities import load_yaml_config  # ✅ moved inside to break cycle
        config = load_yaml_config()
        self._enabled = config.get("profiling", {}).get("enabled", True)
        self._log_errors = config.get("profiling", {}).get("log_errors", True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def log_errors(self):
        return self._log_errors

# Singleton instance
profiling_switch = ProfilingSwitch()
