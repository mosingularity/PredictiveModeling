"""Interactive sandbox for exploring the FortrackDB SQL Server from a local shell.

Everything goes through the same ``db.utilities.read_sql_query`` path the models use
(Spark JDBC, executed on the Databricks cluster, authenticated by its managed identity),
so what works here works in a notebook, and ``ENV`` picks the host exactly as it does in
a real run. Nothing here writes: queries are gated to SELECT/WITH, which is what makes
it safe to point at a live environment while poking around.

Usage (run from the ``predictive_modeling`` folder, or anywhere — it chdirs itself):

    python scripts/sandbox.py                                   # REPL, helpers preloaded
    python scripts/sandbox.py --columns dbo.UserForecastMethod  # column list
    python scripts/sandbox.py --tables '%Forecast%'             # matching tables
    python scripts/sandbox.py --peek dbo.UserForecastMethod -n 3
    python scripts/sandbox.py --find-columns '%DatabrickID%'    # which tables have it
    python scripts/sandbox.py -q "SELECT TOP 5 * FROM dbo.DataBrickTasks"
    python scripts/sandbox.py -f my_query.sql --csv out.csv

One SQL Server quirk to know: Spark's JDBC reader wraps your query in a subquery, and
SQL Server rejects ORDER BY inside one unless TOP is also present. So `SELECT * FROM t
ORDER BY x` fails while `SELECT TOP 100 * FROM t ORDER BY x` works — or just sort the
returned pandas frame, which is what the helpers below do.

In the REPL:

    >>> columns("dbo.UserForecastMethod")     # printed, and returned as a DataFrame
    >>> df = q("SELECT TOP 5 * FROM dbo.UserForecastMethod")
    >>> get_user_forecast_data(spark, 39).toPandas()   # the named queries, live
"""
import argparse
import logging
import os
import re
import sys
from pathlib import Path

# Resolve and enter the package root before importing anything project-local: config.yaml
# is read by relative path (db.utilities.load_yaml_config), and `db`/`notebook_bootstrap`
# are only importable from there.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PACKAGE_ROOT)
sys.path.insert(0, str(PACKAGE_ROOT))

logger = logging.getLogger(__name__)

# A statement is allowed only if it starts with one of these. Anything else (INSERT,
# UPDATE, DELETE, DROP, EXEC, MERGE, ...) is refused — the sandbox is a read tool, and
# ENV may well be pointing at PROD.
_READ_ONLY_STARTS = ("select", "with")

_spark = None


def _load_dotenv():
    """Load ``.env`` the way VS Code does, so a plain terminal run behaves the same.

    Looks in the package root first, then the repo root (both carry one). Values already
    exported in the shell win, so `ENV=QA python scripts/sandbox.py` still does what it says.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.debug("python-dotenv not installed — relying on the ambient environment.")
        return
    for candidate in (PACKAGE_ROOT / ".env", PACKAGE_ROOT.parent / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)


def get_spark(guard_workspace: bool = True):
    """Attach to Spark once per process (databricks-connect locally) and cache it.

    ``guard_workspace`` keeps notebook_bootstrap's "am I pointed at the DEV workspace"
    assertion. Pass False only when you deliberately want another workspace.
    """
    global _spark
    if _spark is not None:
        return _spark
    from notebook_bootstrap import assert_local_workspace, init_spark, resolve_env

    env = resolve_env()
    if env.upper() == "PROD":
        print("⚠️  ENV=PROD — queries will hit the production database (reads only).")
    _spark, _ = init_spark()
    if guard_workspace:
        assert_local_workspace(_spark)
    return _spark


def _assert_read_only(query: str):
    """Reject anything that is not a single read statement.

    Comments are stripped first so a leading ``-- note`` doesn't hide the real verb, and
    trailing statements are refused outright rather than silently executed.
    """
    stripped = re.sub(r"/\*.*?\*/", " ", query, flags=re.S)
    stripped = re.sub(r"--[^\n]*", " ", stripped).strip().rstrip(";").strip()
    if not stripped:
        raise ValueError("Empty query.")
    if ";" in stripped:
        raise ValueError("Multiple statements are not allowed — run them one at a time.")
    verb = stripped.split(None, 1)[0].lower()
    if verb not in _READ_ONLY_STARTS:
        raise ValueError(
            f"Refusing to run a '{verb.upper()}' statement — the sandbox is read-only. "
            f"Allowed: {', '.join(s.upper() for s in _READ_ONLY_STARTS)}."
        )
    return stripped


def q(query: str, spark=None, to_pandas: bool = True):
    """Run a read query and hand back a pandas DataFrame (Spark DataFrame if to_pandas=False)."""
    from db.utilities import read_sql_query

    sql = _assert_read_only(query)
    df = read_sql_query(sql, spark or get_spark())
    return df.toPandas() if to_pandas else df


def columns(table: str, spark=None, show: bool = True):
    """List a table's columns, in ordinal order, with type and nullability.

    ``table`` may be ``dbo.UserForecastMethod`` or just ``UserForecastMethod``
    (schema then defaults to dbo).
    """
    schema, name = _split_table(table)
    df = q(
        f"""
        SELECT ORDINAL_POSITION AS Pos, COLUMN_NAME AS Column_, DATA_TYPE AS Type,
               CHARACTER_MAXIMUM_LENGTH AS MaxLen, IS_NULLABLE AS Nullable,
               COLUMN_DEFAULT AS Default_
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{name}'
        """,
        spark,
    ).sort_values("Pos").reset_index(drop=True)
    if show:
        if df.empty:
            print(f"❓ No columns found for {schema}.{name} — check the name with --tables.")
        else:
            print(f"\n📋 {schema}.{name} — {len(df)} column(s)\n")
            _print_df(df)
            print("\nAs a list:")
            for pos, col, typ in zip(df["Pos"], df["Column_"], df["Type"]):
                print(f"  {pos:>3}. {col} ({typ})")
    return df


def tables(pattern: str = "%", spark=None, show: bool = True):
    """List tables and views whose name matches a SQL LIKE pattern (e.g. '%Forecast%')."""
    df = q(
        f"""
        SELECT TABLE_SCHEMA AS Schema_, TABLE_NAME AS Name, TABLE_TYPE AS Type
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_NAME LIKE '{pattern}'
        """,
        spark,
    ).sort_values(["Schema_", "Name"]).reset_index(drop=True)
    if show:
        print(f"\n🗂️  {len(df)} object(s) matching '{pattern}'\n")
        _print_df(df)
    return df


def find_columns(pattern: str, spark=None, show: bool = True):
    """Find every table carrying a column matching a LIKE pattern.

    The fastest way to answer "where does DatabrickID actually live?" without guessing
    at table names.
    """
    df = q(
        f"""
        SELECT TABLE_SCHEMA AS Schema_, TABLE_NAME AS Table_,
               COLUMN_NAME AS Column_, DATA_TYPE AS Type
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE COLUMN_NAME LIKE '{pattern}'
        """,
        spark,
    ).sort_values(["Schema_", "Table_", "Column_"]).reset_index(drop=True)
    if show:
        print(f"\n🔎 {len(df)} column(s) matching '{pattern}'\n")
        _print_df(df)
    return df


def peek(table: str, n: int = 5, spark=None, show: bool = True):
    """Return the first ``n`` rows of a table — a quick look at real values."""
    schema, name = _split_table(table)
    df = q(f"SELECT TOP {int(n)} * FROM [{schema}].[{name}]", spark)
    if show:
        print(f"\n👀 {schema}.{name} — top {int(n)} row(s)\n")
        _print_df(df)
    return df


def _split_table(table: str):
    """Split 'dbo.Foo' / '[dbo].[Foo]' / 'Foo' into (schema, name), defaulting to dbo."""
    cleaned = table.replace("[", "").replace("]", "").strip()
    if "." in cleaned:
        schema, name = cleaned.split(".", 1)
        return schema.strip(), name.strip()
    return "dbo", cleaned


def _print_df(df):
    """Print a DataFrame in full — no column truncation, which defeats the purpose here."""
    import pandas as pd

    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.max_colwidth", 60, "display.max_rows", 200):
        print(df.to_string(index=False))


def _repl(guard_workspace: bool):
    """Drop into an interactive shell with the helpers and a live Spark session bound."""
    import pandas as pd

    import db.queries as queries

    spark = get_spark(guard_workspace)
    namespace = {
        "spark": spark, "pd": pd, "q": q, "columns": columns, "tables": tables,
        "peek": peek, "find_columns": find_columns, "queries": queries,
        "show": _print_df,
    }
    # The named loaders (get_user_forecast_data, get_actual_data, ...) are the whole
    # reason to sandbox against this schema — bind them at top level so they're one
    # call away, e.g. get_user_forecast_data(spark, 39).toPandas().
    namespace.update({k: v for k, v in vars(queries).items()
                      if callable(v) and k.startswith("get_")})

    banner = (
        "\n🧪 FortrackDB sandbox — ENV=" + os.environ.get("ENV", "?") + "\n"
        "   q(sql) · columns(table) · tables(pattern) · peek(table, n) · find_columns(pattern)\n"
        "   spark, pd, queries, and the db.queries get_* loaders are bound.\n"
        "   Reads only. Ctrl-D to exit.\n"
    )
    try:
        from IPython import start_ipython
        start_ipython(argv=[], user_ns=namespace, display_banner=False)
        print(banner)
    except ImportError:
        import code
        code.interact(banner=banner, local=namespace, exitmsg="")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Read-only sandbox over the FortrackDB SQL Server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage", 1)[1] if "Usage" in __doc__ else None,
    )
    parser.add_argument("-q", "--query", help="SQL to run (SELECT/WITH only).")
    parser.add_argument("-f", "--file", help="Path to a .sql file to run.")
    parser.add_argument("--columns", metavar="TABLE", help="List a table's columns.")
    parser.add_argument("--tables", nargs="?", const="%", metavar="LIKE",
                        help="List tables matching a LIKE pattern (default all).")
    parser.add_argument("--find-columns", metavar="LIKE",
                        help="Find tables carrying a column matching a LIKE pattern.")
    parser.add_argument("--peek", metavar="TABLE", help="Show the first rows of a table.")
    parser.add_argument("-n", "--rows", type=int, default=5, help="Row count for --peek.")
    parser.add_argument("--csv", metavar="PATH", help="Write the result to a CSV instead of printing.")
    parser.add_argument("--no-guard", action="store_true",
                        help="Skip the DEV-workspace assertion (use deliberately).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _load_dotenv()
    guard = not args.no_guard

    if args.query or args.file or args.columns or args.tables or args.find_columns or args.peek:
        spark = get_spark(guard)
    else:
        _repl(guard)
        return 0

    show = args.csv is None
    if args.columns:
        df = columns(args.columns, spark, show=show)
    elif args.tables:
        df = tables(args.tables, spark, show=show)
    elif args.find_columns:
        df = find_columns(args.find_columns, spark, show=show)
    elif args.peek:
        df = peek(args.peek, args.rows, spark, show=show)
    else:
        sql = args.query or Path(args.file).read_text()
        df = q(sql, spark)
        if show:
            print(f"\n📊 {len(df)} row(s)\n")
            _print_df(df)

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"💾 Wrote {len(df)} row(s) to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
