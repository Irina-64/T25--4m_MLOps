import os
import pandas as pd
import pytest
import src.preprocess as preprocess


def test_preprocess(tmp_path, monkeypatch):
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    df = pd.DataFrame({
        "input_text": ["a", "b", "c", "d", "e"],
        "target_text": ["A", "B", "C", "D", "E"]
    })
    fake_csv = raw_dir / "data.csv"
    df.to_csv(fake_csv, index=False)

    monkeypatch.chdir(tmp_path)

    import importlib
    importlib.reload(preprocess)

    assert os.path.exists("data/processed/train.csv")
    assert os.path.exists("data/processed/test.csv")

    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    assert len(train_df) + len(test_df) == 5
    assert len(test_df) == 1
