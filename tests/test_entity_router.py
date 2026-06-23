import pytest

from models.entity_router import route_entity


@pytest.mark.parametrize("tariff_type,entity_id,expected_type", [
    ("LPU", "POD_001", "POD"),
    ("SPU", "COMBO_001", "Combo"),
    ("PPU", "CSA_001", "CSA"),
])
def test_route_entity(tariff_type, entity_id, expected_type):
    row = {"EntityID": entity_id}
    result_id, result_type = route_entity(tariff_type, row)
    assert result_id == entity_id
    assert result_type == expected_type


def test_route_unknown_raises():
    with pytest.raises(ValueError, match="Unknown TariffType"):
        route_entity("UNKNOWN", {"EntityID": "X"})


def test_route_entity_id_reflects_row():
    row_a = {"EntityID": "POD_A"}
    row_b = {"EntityID": "POD_B"}
    id_a, _ = route_entity("LPU", row_a)
    id_b, _ = route_entity("LPU", row_b)
    assert id_a != id_b
