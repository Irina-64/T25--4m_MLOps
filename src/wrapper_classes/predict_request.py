from pydantic import BaseModel


class PredictRequest(BaseModel):
    carrier: str
    connection: str
    name: str
    id_main: str
    arrival: str
    year: int
    month: int
    day: int
    dayofweek: int
    season: int
    is_weekend: bool
    is_holiday: bool
    hour: int
