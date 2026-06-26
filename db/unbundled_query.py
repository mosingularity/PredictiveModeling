"""Decomposed unbundled (entity-keyed) predictive query.

The DEV/Databricks scripts for SPU and LPU are byte-identical apart from a single
WHERE clause, so the shared body lives here once and each tariff scope supplies its
own filter (see :data:`UNBUNDLED_FILTERS`). The caller runs the query per filter and
appends the results — that union is the unbundled prediction input.

Two constraints come from how the query is executed (``read_sql_query`` runs it via
Spark's JDBC ``query`` option, which wraps it as ``SELECT * FROM (<query>) alias``):

* No multi-statement batch — the SSMS ``#category/#csa/#tariff`` temp tables are
  rendered here as inline ``(VALUES …)`` derived tables, so the whole thing is one
  SELECT.
* No ``ORDER BY`` — invalid inside the wrapper; order downstream in pandas if needed.

``to_contract`` maps the raw result to the entity-keyed contract the unbundled loop
groups on (the same shape ``scripts/load_ermelo_csvs.py`` produces).
"""
import pandas as pd

from models.entity_router import ENTITY_TYPE_BY_TARIFF

# Earliest ReportingMonth pulled from ActualData (matches the source scripts).
START_MONTH = "2025-05-01"

# ── lookup dimensions (were #category / #csa / #tariff temp tables) ───────────────

CATEGORY_VALUES = [
    ("Agriculture", "AGR"), ("Commercial", "COM"), ("Industrial", "IND"),
    ("Mining", "MIN"), ("Public Lighting", "PUB"), ("Residential", "RES"),
    ("Prepayment", "PRE"), ("Internal", "INT"),
]

CSA_VALUES = [
    ("Bellville", "BELLV"), ("Benoni", "BENON"), ("Bethlehem", "BETHL"),
    ("Bloemfontein", "BLOEM"), ("East London", "ELOND"), ("Empangeni", "EMPAN"),
    ("Ermelo", "ERMEL"), ("George", "GEORG"), ("Groblersdal", "GROBL"),
    ("Kanyamazane", "KANYA"), ("Kimberley", "KIMBE"), ("Klerksdorp", "KLERK"),
    ("Lephalale", "LEPHA"), ("Margate", "MARGA"), ("Mmabatho", "MMABA"),
    ("Mthatha", "MTHAT"), ("Nelspruit", "NELSP"), ("Newcastle", "NEWCA"),
    ("Nigel", "NIGEL"), ("Phalaborwa", "PHALA"), ("Pietermaritzburg", "PMARIT"),
    ("Polokwane", "POLOK"), ("Port Elizabeth", "PELIZ"), ("Pretoria", "PRETO"),
    ("Queenstown", "QUEEN"), ("Randfontein", "RANDF"), ("Rustenburg", "RUSTE"),
    ("Soweto", "SOWET"), ("Table View", "TVIEW"), ("Thohoyandou", "THOHO"),
    ("Upington", "UPING"), ("Vereeniging", "VEREE"), ("West Coast", "WCOAS"),
    ("Witbank", "WITBA"), ("Worcester", "WORCE"),
]

TARIFF_VALUES = [
    ("Businessrate 1", "BUSS1"), ("Businessrate 2", "BUSS2"),
    ("Businessrate 3", "BUSS3"), ("Businessrate 4", "BUSS4"),
    ("Converted Basic", "CONBASIC"), ("Dummy Rate for conversion", "CONV"),
    ("Homeflex 1", "HOMEFLE1"), ("Homeflex 2", "HOMEFLE2"),
    ("Homeflex 3", "HOMEFLE3"), ("Homeflex 4", "HOMEFLE4"),
    ("Homelight 1 20A", "HOMELIG1-20A"), ("Homelight 1 60A", "HOMELIG1-60A"),
    ("Homelight 2 20A", "HOMELIG2-20A"), ("Homelight 2 60A", "HOMELIG2-60A"),
    ("Homepower 1", "HOMESTD1"), ("Homepower 2", "HOMESTD2"),
    ("Homepower 3", "HOMESTD3"), ("Homepower 4", "HOMESTD4"),
    ("Homepower Standard", "HOMESTD"),
    ("Hometake 1 (60A I5TI 00 Free)", "HOMETK60-00"),
    ("Hometake 1 (60A I5TI 50 FBE)", "HOMETK60-50"),
    ("Hometake 2 (20A I6TI 00 Free)", "HOMETK20-00"),
    ("Hometake 2 (20A I6TI 50 FBE)", "HOMETK20-50"),
    ("Landlight 20A", "LANDL20"), ("Landlight 60A", "LANDL60"),
    ("Landrate 1", "LANDR1"), ("Landrate 2", "LANDR2"),
    ("Landrate 3", "LANDR3"), ("Landrate 4", "LANDR4"),
    ("Landrate DX", "LANDRDX"), ("Municrate 1", "MRATE1"),
    ("Municrate 2", "MRATE2"), ("Municrate 3", "MRATE3"),
    ("Municrate 4", "MRATE4"), ("Pre-Paid T/x Rate Schedule", "PREP-TxRC"),
    ("Public Lighting All Night", "PUBLIGHTAN"), ("Public Lighting 24H", "PUBLIGHT24"),
    ("Public Lighting Munic", "MUPUBLIGHTAN"),
    ("Public Lighting - Telkom Urban Munic", "MUPUBLIGHT24"),
    ("Ruraflex", "RURA"), ("Homepower Bulk", "HOMEB"), ("Miniflex", "MINI"),
    ("Public Lighting", "PUBLIGHT"), ("Public Lighting Urban", "PUBLIGHT"),
]

# ── tariff-scope filters: the ONLY thing that differs between the SPU and LPU runs ──
# (label, where_clause). Currently the Ermelo DEV-test scope; broaden / parameterise
# from the UFM (region, category) when the query goes beyond Ermelo.
UNBUNDLED_FILTERS = [
    ("SPU", "Category = 'Residential' and CustomerServiceArea = 'Ermelo' "
            "and TariffDescription like 'landrate 1'"),
    ("LPU", "Category = 'Industrial' and CustomerServiceArea = 'Ermelo' "
            "and CustomerGroup = 'Key Customer'"),
]


def _render_values(rows, alias, columns) -> str:
    """Render rows as an inline ``(VALUES …) AS alias(col, …)`` derived table."""
    tuples = ", ".join(f"('{a}', '{b}')" for a, b in rows)
    cols = ", ".join(f"[{c}]" for c in columns)
    return f"(values {tuples}) as {alias} ({cols})"


def build_query(where_clause: str) -> str:
    """The shared unbundled query body with ``where_clause`` slotted in.

    A single SELECT (no temp tables, no ORDER BY) so Spark's JDBC ``query`` option
    can wrap it. ``where_clause`` filters the per-pod rows before the entity rollup.
    """
    category = _render_values(CATEGORY_VALUES, "cat", ["Category", "Cat"])
    csa = _render_values(CSA_VALUES, "csa", ["CustomerServiceArea", "CSA"])
    tariff = _render_values(TARIFF_VALUES, "t", ["Tarif", "Id"])
    entity_id = ("case when CustomerType = 'LPU' then PodID "
                 "else concat(Cat, CSA, Tariff, 'V', VoltId, 'Z', ZoneId) end")
    group_cols = (
        "CustomerType, CustomerGroup, Cat, CSA, Tariff, VoltId, ZoneId, "
        "CustomerServiceArea, TariffDescription, TariffGrouping, TariffType, "
        "Category, ReportingMonth")
    return f"""
select
    {entity_id} as ID
    , {group_cols}
    , isnull(sum(OffpeakConsumption), 0) as OffpeakConsumption
    , isnull(sum(StandardConsumption), 0) as StandardConsumption
    , isnull(sum(PeakConsumption), 0) as PeakConsumption
    , isnull(sum(TotalConsumption), 0) as TotalConsumption
    , isnull(sum(NotifiedMaximumDemand), 0) as NMD
    , isnull(sum(NumberOfActivePods), 0) as NoActivePods
    , isnull(sum(NumberOfActiveAccounts), 0) as NoActiveAccounts
from (
    select
        case when h.VoltageGroup is null then '500'
             else replace(translate(h.VoltageGroup, '<>=&VK ', '#######'), '#', '') end as VoltId
        , case when h.TxZone is null or h.TxZone = '' then 'X' else h.TxZone end as ZoneId
        , csa.CSA, cat.Cat, t.Id as Tariff, act.PodID
        , isnull(h.CustomerType, 'SPU') as CustomerType
        , h.CustomerGroup, h.CSA as CustomerServiceArea
        , h.TariffDescription, h.TariffGrouping, h.TariffType
        , act.Category, ReportingMonth
        , OffpeakConsumption, StandardConsumption, PeakConsumption, TotalConsumption
        , NotifiedMaximumDemand, h.NumberofActivePODs, h.NumberOfActiveAccounts
    from ActualData act
    left join {category} on cat.Category = act.Category
    left join {csa} on csa.CustomerServiceArea = act.CustomerServiceArea
    inner join Hierarchy h on h.ID = act.PodID
    left join {tariff} on t.Tarif = h.TariffDescription
    where ReportingMonth >= '{START_MONTH}'
) act
where {where_clause}
group by
    {entity_id}
    , {group_cols}
""".strip()


def to_contract(pdf: pd.DataFrame) -> pd.DataFrame:
    """Map a raw query result to the entity-keyed contract the unbundled loop expects.

    EntityID ← ID, TariffType ← CustomerType (LPU/SPU/PPU), EntityType derived from
    it, TariffID ← Tariff (code). Block/NonTOU columns are absent from the TOU source,
    so they default to 0 — adjust here if a flat-tariff scope needs NonTOU populated.
    """
    out = pd.DataFrame()
    out["EntityID"] = pdf["ID"].astype(str)
    out["TariffType"] = pdf["CustomerType"].astype(str)
    out["EntityType"] = out["TariffType"].map(ENTITY_TYPE_BY_TARIFF)
    out["TariffID"] = pdf["Tariff"].astype(str)
    out["ReportingMonth"] = pd.to_datetime(pdf["ReportingMonth"])
    out["PeakConsumption"] = pd.to_numeric(pdf["PeakConsumption"], errors="coerce").fillna(0.0)
    out["StandardConsumption"] = pd.to_numeric(pdf["StandardConsumption"], errors="coerce").fillna(0.0)
    out["OffPeakConsumption"] = pd.to_numeric(pdf["OffpeakConsumption"], errors="coerce").fillna(0.0)
    for col in ("Block1Consumption", "Block2Consumption", "Block3Consumption",
                "Block4Consumption", "NonTOUConsumption"):
        out[col] = 0.0
    return out
