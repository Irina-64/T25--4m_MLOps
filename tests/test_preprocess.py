import pandas as pd

from src import preprocess


def test_normalize_result():
    s = pd.Series(["W", "l", "No Contest", "won"])
    out = preprocess.normalize_result(s)
    assert out.tolist() == ["win", "loss", "nc", "win"]


def test_pick_cols():
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
