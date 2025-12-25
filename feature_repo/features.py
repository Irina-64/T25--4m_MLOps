from feast import Entity, FeatureView, FileSource, Field
from feast.types import Float32, Int64
from datetime import timedelta

# -------------------------
# Entity: шаг игры
# -------------------------
step = Entity(
    name="step_id",
    join_keys=["step_id"]
)

# -------------------------
# Offline source (CSV)
# -------------------------
rl_state_source = FileSource(
    path="data/rl_state_features.csv",
    timestamp_field="event_timestamp",
)

# -------------------------
# Feature View
# -------------------------
rl_state_features = FeatureView(
    name="rl_state_features",
    entities=[step],
    ttl=timedelta(days=1),
    schema=[
        Field(name="deck_frac", dtype=Float32),
        Field(name="discard_frac", dtype=Float32),
        Field(name="is_attacker", dtype=Int64),
        Field(name="is_defender", dtype=Int64),
        Field(name="hand_size_frac", dtype=Float32),
        Field(name="opponents_avg_hand", dtype=Float32),
        Field(name="trump_suit_id", dtype=Int64),
    ],
    source=rl_state_source,
)
