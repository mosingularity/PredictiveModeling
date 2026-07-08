"""
Root conftest.py — stubs Databricks/PySpark modules that are unavailable
outside the Databricks runtime so that unit tests can run locally.
"""
import sys
import types


def _make_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


def _stub_pyspark():
    # Sentinel object used wherever a Spark type is referenced but never called in tests.
    class _Stub:
        def __init__(self, *a, **kw): pass
        def __call__(self, *a, **kw): return self
        def __getattr__(self, _): return self

    stub = _Stub()

    # Each stub module needs __path__ so Python treats it as a package and allows
    # sub-imports like `from pyspark.sql.types import ...`.
    def _pkg(name, **attrs):
        mod = _make_module(name, **attrs)
        mod.__path__ = []
        mod.__package__ = name
        return mod

    pyspark              = _pkg("pyspark")
    pyspark_sql          = _pkg("pyspark.sql", DataFrame=_Stub, SparkSession=_Stub)
    pyspark_sql_functions = _pkg(
        "pyspark.sql.functions",
        col=stub, to_date=stub, expr=stub, explode=stub, array=stub,
        lit=stub, when=stub, to_timestamp=stub, month=stub, year=stub,
    )
    pyspark_sql_window   = _pkg("pyspark.sql.window", Window=_Stub)
    pyspark_sql_types    = _pkg(
        "pyspark.sql.types",
        StructType=_Stub, StructField=_Stub,
        IntegerType=_Stub, StringType=_Stub, TimestampType=_Stub,
        FloatType=_Stub, LongType=_Stub,
    )
    pyspark_dbutils      = _pkg("pyspark.dbutils", DBUtils=_Stub)

    for mod, name in [
        (pyspark,               "pyspark"),
        (pyspark_sql,           "pyspark.sql"),
        (pyspark_sql_functions, "pyspark.sql.functions"),
        (pyspark_sql_window,    "pyspark.sql.window"),
        (pyspark_sql_types,     "pyspark.sql.types"),
        (pyspark_dbutils,       "pyspark.dbutils"),
    ]:
        sys.modules.setdefault(name, mod)


# Run once at collection time, before any project module is imported.
_stub_pyspark()
