_ENTITY_TYPE = {"LPU": "POD", "SPU": "Combo", "PPU": "CSA"}


def route_entity(tariff_type: str, row: dict) -> tuple[str, str]:
    """Return (entity_id, entity_type) for a given tariff type and data row."""
    entity_id = row["EntityID"]
    try:
        return entity_id, _ENTITY_TYPE[tariff_type]
    except KeyError:
        raise ValueError(f"Unknown TariffType: {tariff_type!r}")
