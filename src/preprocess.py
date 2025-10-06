import os

import pandas as pd

os.makedirs("data/processed", exist_ok=True)

fighters = pd.read_csv("data/raw/Fighters.csv")
fights = pd.read_csv("data/raw/Fights.csv")
events = pd.read_csv("data/raw/Events.csv")
fstats = pd.read_csv("data/raw/Fstats.csv")


def preprocess(df):
    df = df.dropna()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["day_of_week"] = df["Date"].dt.weekday
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


fighters_processed = preprocess(fighters)
fights_processed = preprocess(fights)
events_processed = preprocess(events)
fstats_processed = preprocess(fstats)

fighters_processed.to_csv("data/processed/fighters_processed.csv", index=False)
fights_processed.to_csv("data/processed/fights_processed.csv", index=False)
events_processed.to_csv("data/processed/events_processed.csv", index=False)
fstats_processed.to_csv("data/processed/fstats_processed.csv", index=False)
