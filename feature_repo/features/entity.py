from feast import Entity
from feast import ValueType

user_id = Entity(
    name="user_id",
    value_type=ValueType.INT64,
    description="User identifier",
)

amount_count = Entity(
    name="amount_count",
    value_type=ValueType.INT64,
    description="Count of amount identifier",
)

amount_sum = Entity(
    name="amount_sum",
    value_type=ValueType.FLOAT,
    description="Sum of amount identifier",
)
