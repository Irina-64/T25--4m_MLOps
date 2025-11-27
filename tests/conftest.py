import sys
import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import src.api as api


@pytest.fixture(scope="session")
def client():
    """
    Создаём тестовый FastAPI client.
    Но заменяем настоящую модель и токенайзер на фейки.
    """

    # мок токенайзера
    api.tokenizer = MagicMock()
    api.tokenizer.return_value = {"input_ids": [0], "attention_mask": [1]}

    api.tokenizer.decode.return_value = "detoxified text"

    # мок модели
    fake_model = MagicMock()
    fake_model.generate.return_value = [[1, 2, 3]]

    api.model = fake_model

    return TestClient(api.app)
