import pandas as pd
import numpy as np
import os

def load_data(path: str) -> pd.DataFrame:
    """Загрузка данных из CSV."""
    df = pd.read_csv(path, encoding_errors='strict')
    df.dropna(how='all', inplace=True)
    df.dropna(subset=["user_id", "date", "amount", "churn"], inplace=True)
    return df

def basic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Базовые признаки по дате и операциям."""
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.weekday
    df["is_income"] = (df["amount"] > 0).astype(int)
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    return df

def get_window_features(df, start_date, end_date, prefix=""):
    """Агрегация по временным окнам."""
    window = df[(df["date"] >= start_date) & (df["date"] < end_date)]
    agg = window.groupby("user_id")["amount"].agg(
        ["sum", "mean", "std", "count", "max", "min"]
    ).add_prefix(f"{prefix}_")
    return agg.reset_index()

def compute_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """Извлечение пользовательских признаков."""
    weekend_ratio = df.groupby("user_id")["is_weekend"].mean().reset_index(name="weekend_ratio")

    features_all = get_window_features(df, "2024-01-01", "2024-12-01", "all")
    features_3m = get_window_features(df, "2024-09-01", "2024-12-01", "last3m")
    features_1m = get_window_features(df, "2024-11-01", "2024-12-01", "last1m")

    activity_days = df.groupby("user_id")["date"].apply(lambda x: x.dt.date.nunique()).reset_index(name="active_days")

    df_sorted = df.sort_values(["user_id", "date"])
    df_sorted["prev_date"] = df_sorted.groupby("user_id")["date"].shift()
    df_sorted["delta_days"] = (df_sorted["date"] - df_sorted["prev_date"]).dt.total_seconds() / (3600 * 24)
    avg_gap = df_sorted.groupby("user_id")["delta_days"].mean().reset_index(name="avg_gap_days")

    last_activity = df.groupby("user_id")["date"].max().reset_index()
    last_activity["days_since_last"] = (pd.to_datetime("2024-12-01") - last_activity["date"]).dt.days
    last_activity = last_activity[["user_id", "days_since_last"]]

    income_share = df.groupby("user_id")["is_income"].mean().reset_index(name="income_share")

    df["date_only"] = df["date"].dt.date
    income_days = df[df["amount"] > 0].groupby("user_id")["date_only"].nunique().reset_index(name="income_days")
    outcome_days = df[df["amount"] < 0].groupby("user_id")["date_only"].nunique().reset_index(name="outcome_days")

    df_sorted["gap_days"] = (df_sorted["date"] - df_sorted["prev_date"]).dt.days
    has_large_gap = df_sorted.groupby("user_id")["gap_days"].max().reset_index()
    has_large_gap["has_gap_30d"] = (has_large_gap["gap_days"] > 30).astype(int)

    # Объединяем все фичи
    features = (
        weekend_ratio
        .merge(features_all, on="user_id", how="left")
        .merge(features_3m, on="user_id", how="left")
        .merge(features_1m, on="user_id", how="left")
        .merge(activity_days, on="user_id", how="left")
        .merge(avg_gap, on="user_id", how="left")
        .merge(last_activity, on="user_id", how="left")
        .merge(income_share, on="user_id", how="left")
        .merge(income_days, on="user_id", how="left")
        .merge(outcome_days, on="user_id", how="left")
        .merge(has_large_gap[["user_id", "has_gap_30d"]], on="user_id", how="left")
    )

    return features


if __name__ == "__main__":
    raw_path = "data/raw/train_data.csv"
    processed_dir = "data/processed"
    processed_path = os.path.join(processed_dir, "processed.csv")

    # Проверяем, что папка существует
    os.makedirs(processed_dir, exist_ok=True)

    # Загружаем и обрабатываем
    df = load_data(raw_path)
    df = basic_feature_engineering(df)
    features = compute_user_features(df)

    # Сохраняем
    features.to_csv(processed_path, index=False)
    print(f"✅ Обработанные данные сохранены: {processed_path}")