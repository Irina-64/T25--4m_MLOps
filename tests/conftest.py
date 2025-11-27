from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture
def client():
    with patch("src.api.joblib.load") as mock_model_load, patch("builtins.open"), patch(
        "json.load"
    ) as mock_json_load:
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = [[0.7]]
        mock_model_load.return_value = mock_model

        mock_json_load.return_value = ["a", "b", "c"]

        yield TestClient(app)
