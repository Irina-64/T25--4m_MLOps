import pytest
from fastapi.testclient import TestClient
from src.api import app

@pytest.fixture(scope="module")
def test_client():
    with TestClient(app) as client:
        yield client

@pytest.fixture
def sample_extrovert_data():
    return {"Time_broken_spent_Alone": 4.0,"Stage_fear": 0,"Social_event_attendance": 4.0,"Going_outside": 6.0,"Drained_after_socializing": 0,"Friends_circle_size": 13.0,"Post_frequency": 5.0}

@pytest.fixture
def sample_introvert_data():
    return {"Time_broken_spent_Alone": 9.0,"Stage_fear": 1,"Social_event_attendance": 1.0,"Going_outside": 2.0,"Drained_after_socializing": 1,"Friends_circle_size": 5.0,"Post_frequency": 2.0}