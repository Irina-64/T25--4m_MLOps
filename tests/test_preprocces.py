import pandas as pd
import pytest

from src import preprocess


@pytest.fixture(autouse=True)
def mock_read_csv(monkeypatch):
    def fake_read_csv(path, *args, **kwargs):
        return pd.DataFrame({"a": [1, 2, 3]})

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)


def test_normalize_result_basic():
    s = pd.Series(["W", "l", "No Contest", "won"])
    result = preprocess.normalize_result(s)
    expected = pd.Series(["win", "loss", "nc", "win"])
    assert result.tolist() == expected.tolist()


def test_pick_cols_output_shape():
    df = pd.DataFrame(
        {
            "Full Name": ["a", "b"],
            "num1": [1, 2],
            "num2": [3, 4],
            "cat1": ["x", "y"],
            "cat2": ["p", "q"],
        }
    )
    sub = preprocess.pick_cols("R", df, "Full Name")
    assert "R_num1" in sub.columns
    assert "R_cat1" in sub.columns
    assert "_R_key" not in sub.columns
