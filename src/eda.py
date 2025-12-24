import logging
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

IN = Path("data/raw/delays.csv")
OUT = Path("reports/eda")
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    hdr = pd.read_csv(path, nrows=0).columns.tolist()
    parse_dates = [c for c in ("datetime", "date") if c in hdr] or None
    dtype: dict[str, str] | None = {
        c: "category" for c in ("carrier", "connection", "name") if c in hdr
    }
    LOG.info(
        "Чтение %s parse_dates=%s dtype_поля=%s",
        path,
        parse_dates,
        list(dtype.keys()) if dtype else None,
    )
    return pd.read_csv(path, parse_dates=parse_dates, dtype=dtype, low_memory=False)  # type: ignore


def to_numeric_from_text(s: pd.Series):
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    nums = s.astype(str).str.extract(r"(-?\d+\.?\d*)", expand=False)
    return pd.to_numeric(nums, errors="coerce")


def save(fig, name):
    p = OUT / name
    try:
        fig.savefig(p, dpi=150, bbox_inches="tight")
        LOG.info("Сохранено: %s", p)
    except Exception as e:
        LOG.warning("Ошибка при сохранении %s: %s", p, e)
    finally:
        plt.close(fig)
    return p.name


def main():
    df = read_csv(IN).copy()
    LOG.info("Размер=%s колонки=%s", df.shape, df.columns.tolist())

    if "delay" not in df.columns:
        raise ValueError("Отсутствует колонка 'delay'")

    df["delay"] = to_numeric_from_text(df["delay"])
    before = len(df)
    df = df.dropna(subset=["delay"]).reset_index(drop=True)
    if before - len(df):
        LOG.warning("Удалено строк с некорректными задержками: %d", before - len(df))

    imgs = []

    # Распределение + знаковый логарифм
    if not df["delay"].empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        df["delay"].hist(bins=50, edgecolor="black", ax=axes[0])
        axes[0].set(title="Распределение задержек", xlabel="Минуты", ylabel="Частота")

        df["delay_log_signed"] = np.sign(df["delay"]) * np.log1p(df["delay"].abs())
        pd.Series(df["delay_log_signed"]).hist(bins=50, edgecolor="black", ax=axes[1])
        axes[1].set(
            title="Знаковый log(задержка + 1)", xlabel="Signed log", ylabel="Частота"
        )

        imgs.append(save(fig, "delay_distribution.png"))

    # Категориальные признаки
    for c in ("carrier", "connection", "name"):
        if c in df.columns and df[c].notna().any():
            LOG.info(
                "%s: уникальных=%s топ=%s",
                c,
                df[c].nunique(),
                df[c].value_counts().head(5).to_dict(),
            )

    # Признаки даты/времени (приоритет datetime, затем date)
    if "datetime" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"]).reset_index(drop=True)
        if not df.empty:
            df["hour"] = df["datetime"].dt.hour.astype("Int64")  # type: ignore
    elif "date" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).reset_index(drop=True)
        if not df.empty:
            df["day"] = df["date"].dt.day.astype("Int64")  # type: ignore

    # По часам
    if "hour" in df.columns and df["hour"].notna().any():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        df["hour"].value_counts().sort_index().plot(kind="bar", ax=ax1)
        ax1.set_title("Количество поездов по часам")
        ax1.set_ylabel("Количество")

        df.groupby("hour")["delay"].mean().plot(marker="o", ax=ax2)
        ax2.set_title("Средняя задержка по часам")
        ax2.set_xlabel("Час")
        ax2.set_ylabel("Средняя задержка (мин)")

        imgs.append(save(fig, "delay_by_hour.png"))

    # Средняя задержка по перевозчикам
    if "carrier" in df.columns and df["carrier"].notna().any():
        avg_by_carrier = (
            df.groupby("carrier")["delay"].mean().sort_values(ascending=False).head(20)
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        avg_by_carrier.plot(kind="barh", ax=ax)
        ax.set_title("Средняя задержка по перевозчикам (Топ-20)")
        ax.set_xlabel("Средняя задержка (мин)")
        ax.invert_yaxis()

        imgs.append(save(fig, "avg_delay_by_carrier.png"))

    # Пропущенные значения
    missing = df.isnull().sum()
    miss_df = pd.DataFrame(
        {"count": missing, "percent": 100 * missing / max(1, len(df))}
    )
    LOG.info(
        "Колонки с пропусками:\n%s",
        miss_df[miss_df["count"] > 0].to_string(),
    )

    # Корреляции
    numeric = df[
        [c for c in ("delay", "delay_log_signed", "day", "hour") if c in df.columns]
    ].select_dtypes(include=[np.number])
    if numeric.shape[1] > 1:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Корреляционная матрица")
        imgs.append(save(fig, "correlation_matrix.png"))
    else:
        LOG.info(
            "Недостаточно числовых колонок для корреляции: %s",
            numeric.columns.tolist(),
        )

    # Агрегации
    for col in ("hour", "day"):
        if col in df.columns and df[col].notna().any():
            agg = df.groupby(col)["delay"].mean()
            LOG.info("Средняя задержка по %s:\n%s", col, agg.to_string())
            fig, ax = plt.subplots(figsize=(8, 4))
            agg.plot(kind="bar", ax=ax)
            ax.set(title=f"Средняя задержка по {col}", xlabel=col, ylabel="Минуты")
            imgs.append(save(fig, f"avg_delay_by_{col}.png"))

    # --- HTML-отчёт ---
    html_path = OUT / "eda.html"
    imgs_html = "\n".join(
        f"""
        <figure>
            <img src="{img}" alt="{img}">
            <figcaption>{img.replace("_", " ").replace(".png", "").title()}</figcaption>
        </figure>
        """
        for img in imgs
    )

    html_report = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="utf-8">
        <title>EDA-отчёт по задержкам поездов</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 30px; line-height: 1.5; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; border-bottom: 2px solid #eee; padding-bottom: 6px; }}
            figure {{ margin: 20px 0; }}
            figcaption {{ text-align: center; font-size: 0.9em; color: #555; }}
            table {{ border-collapse: collapse; margin: 10px 0; }}
            th, td {{ border: 1px solid #ccc; padding: 5px 10px; }}
            th {{ background-color: #f0f0f0; }}
            pre {{ background: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
        </style>
    </head>
    <body>
        <h1>EDA-отчёт по задержкам поездов</h1>

        <h2>Обзор датасета</h2>
        <p><strong>Размер:</strong> {df.shape}</p>
        <p><strong>Колонки:</strong> {", ".join(df.columns.tolist())}</p>

        <h2>Графики</h2>
        {imgs_html}

        <h2>Статистика задержек</h2>
        <pre>{df["delay"].describe(percentiles=[0.75, 0.90, 0.95]).to_string()}</pre>

        <p><em>Сгенерировано {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</em></p>
    </body>
    </html>
    """

    html_path.write_text(html_report, encoding="utf-8")
    LOG.info("HTML-отчёт сохранён: %s", html_path)


if __name__ == "__main__":
    main()
