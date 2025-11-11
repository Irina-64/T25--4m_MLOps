import importlib

import pandas as pd


def test_preprocess_creates_output(tmp_path, monkeypatch):
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    raw_dir.mkdir(parents=True)

    fighters = pd.DataFrame(
        {
            "Full Name": ["John", "Mike"],
            "Ht.": [180, 175],
            "Wt.": [77, 70],
            "Reach": [190, 180],
            "Stance": ["Orthodox", "Southpaw"],
        }
    )
    fights = pd.DataFrame(
        {
            "Fighter_1": ["John"],
            "Fighter_2": ["Mike"],
            "Result_1": ["win"],
            "Result_2": ["loss"],
            "KD_1": [1],
            "KD_2": [0],
            "Weight_Class": ["Lightweight"],
            "Method": ["KO"],
            "Round": [1],
            "Fight_Time": [60],
            "Event_Id": [1],
        }
    )
    events = pd.DataFrame({"Event_Id": [1], "Name": ["UFC Test"]})
    fstats = pd.DataFrame(columns=["dummy"])
    fstats.to_csv(raw_dir / "Fstats.csv", index=False)

    fighters.to_csv(raw_dir / "Fighters.csv", index=False)
    fights.to_csv(raw_dir / "Fights.csv", index=False)
    events.to_csv(raw_dir / "Events.csv", index=False)
    fstats.to_csv(raw_dir / "Fstats.csv", index=False)

    monkeypatch.chdir(tmp_path)
    importlib.invalidate_caches()

    out_path = processed_dir / "processed.csv"
    assert out_path.exists(), "Файл processed.csv не создан"
    df = pd.read_csv(out_path)
    assert not df.empty, "Выходной CSV пуст"
    assert "target" in df.columns, "Нет столбца target"
