from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int32
from feast.value_type import ValueType

user = Entity(
    name="user",
    description="Пользователь для предсказания экстраверсии/интроверсии",
    join_keys=["user_id"],
    value_type=ValueType.INT64
)

personality_source = FileSource(
    name="personality_source",
    path="data/processed/processed_for_feast.csv",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp"
)

personality_features = FeatureView(
    name="personality_features",
    entities=[user],
    ttl=timedelta(days=365*10),
    schema=[
        Field(name="time_broken_spent_alone", dtype=Float32,
              description="Время, проведенное в одиночестве (часы)"),

        Field(name="stage_fear", dtype=Int32,
              description="Боязнь сцены (0=нет, 1=да)"),

        Field(name="social_event_attendance", dtype=Int32,
              description="Посещение социальных мероприятий"),
        Field(name="going_outside", dtype=Int32,
              description="Частота выхода из дома"),

        Field(name="drained_after_socializing", dtype=Int32,
              description="Чувство истощения после общения (0=нет, 1=да)"),
        
        Field(name="friends_circle_size", dtype=Int32,
              description="Размер круга друзей"),
        Field(name="post_frequency", dtype=Int32,
              description="Частота постов в соцсетях"),

        Field(name="personality_encoded", dtype=Int32,
              description="Закодированная личность (0=интроверт, 1=экстраверт)")
    ],
    source=personality_source,
    online=False
)

derived_personality_features = FeatureView(
    name="derived_personality_features",
    entities=[user],
    ttl=timedelta(days=365*10),
    schema=[
        Field(name="social_activity_score", dtype=Float32,
              description="Общий балл социальной активности"),
        Field(name="alone_time_ratio", dtype=Float32,
              description="Доля времени в одиночестве"),
        Field(name="social_burnout_indicator", dtype=Int32,
              description="Индикатор социального выгорания")
    ],
    source=personality_source
)