import numpy as np
import pandas as pd

df = pd.read_csv("data/raw/dataset.csv")

df.replace(["nan", "None", ""], np.nan, inplace=True)

df["owners"] = df["owners"].astype("Int64")
df["year"] = df["year"].astype("Int64")
df["price"] = df["price"].astype("Int64")
df["mileage"] = df["mileage"].astype("Int64")
df["power"] = df["power"].astype("Int64")
df["displacement"] = df["displacement"].astype(float)
df["super_gen_year_from"] = df["super_gen_year_from"].astype("Int64")
df["super_gen_year_to"] = df["super_gen_year_to"].astype(float)

str_cols = [
    "region",
    "mark",
    "model",
    "complectation",
    "steering_wheel",
    "gear_type",
    "engine",
    "transmission",
    "characteristics",
    "color",
    "body_type_type",
    "body_type_name",
    "super_gen_name",
]
for col in str_cols:
    df[col] = df[col].astype(str)

df.dropna(how="all", inplace=True)


df.to_csv("data/processed/dataset.csv", index=False, encoding="utf-8-sig")
