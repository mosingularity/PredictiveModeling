# Tariff type → entity type. The single source of truth, shared by the router
# and the unbundled DB query's contract mapping.
ENTITY_TYPE_BY_TARIFF = {"LPU": "POD", "SPU": "Combo", "PPU": "CSA"}


def route_entity(tariff_type: str, row: dict) -> tuple[str, str]:
    """Return (entity_id, entity_type) for a given tariff type and data row."""
    entity_id = row["EntityID"]
    try:
        return entity_id, ENTITY_TYPE_BY_TARIFF[tariff_type]
    except KeyError:
        raise ValueError(f"Unknown TariffType: {tariff_type!r}")
