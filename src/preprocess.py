import pandas as pd
import numpy as np
import os

def load_data(path):
    """Загрузка исходных данных"""
    df = pd.read_csv(path, sep=";", encoding_errors="ignore")
    print(f"✅ Данные успешно загружены: {df.shape[0]} строк, {df.shape[1]} столбцов")
    return df

def preprocess_data(df):
    """Предобработка и генерация признаков"""
    # Удаляем пустые строки
    df.dropna(how='all', inplace=True)
    df.dropna(subset=["user_id", "date", "amount", "churn"], inplace=True)

    # Преобразуем дату и создаём базовые признаки
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.weekday
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["is_income"] = (df["amount"] > 0).astype(int)
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

    # Отношение транзакций по выходным
    weekend_ratio = df.groupby("user_id")["is_weekend"].mean().reset_index(name="weekend_ratio")

    # Функция для создания агрегированных признаков за разные окна
    def get_window_features(df, start_date, end_date, prefix=""):
        window = df[(df["date"] >= start_date) & (df["date"] < end_date)]
        agg = window.groupby("user_id")["amount"].agg(
            ["sum", "mean", "std", "count", "max", "min"]
        ).add_prefix(f"{prefix}_")
        return agg.reset_index()

    features_all = get_window_features(df, "2024-01-01", "2024-12-01", "all")
    features_3m = get_window_features(df, "2024-09-01", "2024-12-01", "last3m")
    features_1m = get_window_features(df, "2024-11-01", "2024-12-01", "last1m")

    # Активность пользователя
    activity_days = df.groupby("user_id")["date"].apply(lambda x: x.dt.date.nunique()).reset_index(name="active_days")

    # Средние интервалы между транзакциями
    df_sorted = df.sort_values(["user_id", "date"])
    df_sorted["prev_date"] = df_sorted.groupby("user_id")["date"].shift()
    df_sorted["delta_days"] = (df_sorted["date"] - df_sorted["prev_date"]).dt.total_seconds() / (3600 * 24)
    avg_gap = df_sorted.groupby("user_id")["delta_days"].mean().reset_index(name="avg_gap_days")
    max_gap = df_sorted.groupby("user_id")["delta_days"].max().reset_index(name="max_gap_days")

    # Последняя активность
    last_activity = df.groupby("user_id")["date"].max().reset_index()
    last_activity["days_since_last"] = (pd.to_datetime("2024-12-01") - last_activity["date"]).dt.days
    last_activity = last_activity[["user_id", "days_since_last"]]

    # Доля положительных транзакций
    income_share = df.groupby("user_id")["is_income"].mean().reset_index(name="income_share")

    # Кол-во дней с приходами и расходами
    df["date_only"] = df["date"].dt.date
    income_days = df[df["amount"] > 0].groupby("user_id")["date_only"].nunique().reset_index(name="income_days")
    outcome_days = df[df["amount"] < 0].groupby("user_id")["date_only"].nunique().reset_index(name="outcome_days")

    # Признак наличия больших пропусков между транзакциями
    df_sorted["gap_days"] = (df_sorted["date"] - df_sorted["prev_date"]).dt.days
    has_large_gap = df_sorted.groupby("user_id")["gap_days"].max().reset_index()
    has_large_gap["has_gap_30d"] = (has_large_gap["gap_days"] > 30).astype(int)
    has_large_gap = has_large_gap[["user_id", "has_gap_30d"]]

    # Отношение активности между сентябрем и ноябрем
    user_month_counts = df.groupby(["user_id", df["date"].dt.month])["amount"].count().unstack(fill_value=0)
    user_month_counts["activity_trend_sep_nov"] = user_month_counts.get(11, 0) / (user_month_counts.get(9, 1))
    user_month_trend = user_month_counts[["activity_trend_sep_nov"]].reset_index()

    # Объединяем все признаки
    features = (
        features_all.merge(features_3m, on="user_id", how="left")
        .merge(features_1m, on="user_id", how="left")
        .merge(activity_days, on="user_id", how="left")
        .merge(avg_gap, on="user_id", how="left")
        .merge(max_gap, on="user_id", how="left")
        .merge(last_activity, on="user_id", how="left")
        .merge(income_share, on="user_id", how="left")
        .merge(income_days, on="user_id", how="left")
        .merge(outcome_days, on="user_id", how="left")
        .merge(has_large_gap, on="user_id", how="left")
        .merge(weekend_ratio, on="user_id", how="left")
        .merge(user_month_trend, on="user_id", how="left")
    )

    # Добавляем целевую переменную
    target = df[["user_id", "churn"]].drop_duplicates(subset=["user_id"])
    features = features.merge(target, on="user_id", how="left")

    # Заполняем пропуски
    features.fillna(0, inplace=True)

    print(f"Признаки успешно сгенерированы: {features.shape[1]} столбцов")
    return features


if __name__ == "__main__":
    raw_path = "/workspace/data/raw/train_data.csv"
    processed_path = "/workspace/data/processed/processed.csv"

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)

    df = load_data(raw_path)
    features = preprocess_data(df)

    features.to_csv(processed_path, index=False)
    print(f"✅ Обработанный датасет сохранён в: {processed_path}")
