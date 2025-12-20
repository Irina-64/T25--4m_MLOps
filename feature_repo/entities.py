from feast import Entity
from feast.value_type import ValueType

fight = Entity(
    name="fight_id",
    join_keys=["fight_id"],
    value_type=ValueType.INT64,
)
