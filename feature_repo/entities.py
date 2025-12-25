from feast import Entity
from feast.value_type import ValueType

user = Entity(
    name="user_id",
    value_type=ValueType.INT64,
    description="User identifier",
)

