from pathlib import Path

import holidays
import numpy as np
import pandas as pd

base_dir = Path(__file__).resolve().parent.parent
raw_data = Path(base_dir, "data/raw/delays.csv")
processed_dir = Path(base_dir, "data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)


def preprocess(data: Path) -> pd.DataFrame:
    # === Загрузка ===
    df = pd.read_csv(data)

    print("=== ИСХОДНЫЕ ДАННЫЕ ===")
    print(f"Столбцы: {list(df.columns)}")
    print(f"Всего строк: {len(df)}")

    # Переименовываем date в departure_date
    df = df.rename(columns={"date": "departure_date"})

    id_column_name = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    s = df[id_column_name].astype(str)

    df["id_main"] = s.str.extract(r"(\d+(?:/\d+)?)", expand=False)
    df["id_bracket"] = s.str.extract(r"\((\d+)\)", expand=False)

    df["id_text"] = (
        s.str.replace(r"\(\d+\)", "", regex=True)
        .str.replace(r"\d+(?:/\d+)?", "", regex=True)
        .str.strip()
        .replace("", pd.NA)
    )

    # Удаляем исходный id и вспомогательные колонки
    df = df.drop(columns=[id_column_name, "id_bracket", "id_text"])

    # === Обработка delay ===
    delay_column = next(
        (c for c in df.columns if "delay" in c.lower() or "задержка" in c.lower()), None
    )

    if delay_column is None:
        delay_column = df.columns[5] if len(df.columns) > 5 else df.columns[-1]

    print(f"\nОбработка столбца задержек: {delay_column}")

    if df[delay_column].dtype == "object":
        df[delay_column] = (
            df[delay_column]
            .astype(str)
            .str.replace("min", "", regex=False)
            .str.strip()
            .pipe(pd.to_numeric, errors="coerce")
        )

    # === Обработка datetime ===
    datetime_column = df.columns[0]
    print(f"Обработка datetime столбца: {datetime_column}")

    if df[datetime_column].dtype == "object":
        df[datetime_column] = pd.to_datetime(df[datetime_column], errors="coerce")
        dt = df[datetime_column]

        df["year"] = dt.dt.year.astype("int16")
        df["month"] = dt.dt.month.astype("int8")
        df["day"] = dt.dt.day.astype("int8")
        df["dayofweek"] = dt.dt.weekday.astype("int8")
        df["hour"] = dt.dt.hour.astype("int8")

        df["season"] = ((dt.dt.month % 12) // 3).astype("int8")
        df["is_weekend"] = (df["dayofweek"] >= 5).astype("boolean")

        pl_holidays = holidays.country_holidays("PL", years=df["year"].unique())
        df["is_holiday"] = dt.dt.date.isin(pl_holidays).astype("boolean")

        # Циклическое кодирование часа
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        df = df.drop(columns=[datetime_column, "hour"])

    # === Очистка данных ===
    print("\n=== ОЧИСТКА ДАННЫХ ===")

    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    critical_columns = ["arrival", "day", "id_main", delay_column]
    df = df.dropna(subset=critical_columns)

    # Порядок колонок
    new_order = [
        "carrier",
        "connection",
        "name",
        "id_main",
        "arrival",
        "delay",
        "year",
        "month",
        "day",
        "dayofweek",
        "season",
        "is_weekend",
        "is_holiday",
        "hour_sin",
        "hour_cos",
    ]
    df = df[new_order]

    print("\n=== СТАТИСТИКА ===")
    print(f"Уникальных основных ID: {df['id_main'].nunique()}")

    # Перемешивание (один раз)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # === Сохранение ===
    output_path = Path(processed_dir, "delays.csv")
    df.to_csv(output_path, index=False)

    print("\n=== СОХРАНЕН ФАЙЛ ===")
    print(f"  {output_path}")
    print(f"  Размер: {output_path.stat().st_size / 1024:.1f} KB")

    return df


if __name__ == "__main__":
    df_processed = preprocess(raw_data)
    print("\n✅ Предобработка завершена успешно!")
