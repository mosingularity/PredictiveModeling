import uuid
import os

class RunContext:
    def __init__(self):
        self._run_id = None
        self._app_name = None

    def init(self, app_name: str = None, run_id: str = None):
        self._app_name = app_name or os.getenv("APP_NAME", "default-app")
        self._run_id = run_id or os.getenv("RUN_ID") or str(uuid.uuid4())

    @property
    def run_id(self):
        return self._run_id

    @property
    def app_name(self):
        return self._app_name


# Singleton instance
run_context = RunContext()
