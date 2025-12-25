# create_parquet.py (сохраните в корне проекта)
import pandas as pd
import os

print("=== Создание Parquet файла для Feast ===")

# 1. Найдите CSV файл
csv_files = []
for f in os.listdir("data/processed"):
    if f.endswith(".csv") and "feast" in f.lower():
        csv_files.append(f)

if not csv_files:
    print("❌ Не найден CSV файл для Feast")
    print("   Доступные файлы:", os.listdir("data/processed"))
    exit(1)

csv_file = csv_files[0]
csv_path = f"data/processed/{csv_file}"
print(f"📁 Найден CSV: {csv_path}")

# 2. Читаем CSV
df = pd.read_csv(csv_path)
print(f"   Размер: {df.shape[0]} строк, {df.shape[1]} колонок")

# 3. Убедимся, что есть currency_date_id (ключ для Feast)
if 'currency_date_id' not in df.columns:
    print("➕ Добавляем currency_date_id...")
    df['currency_date_id'] = range(1, len(df) + 1)

# 4. Конвертируем дату
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])

# 5. Сохраняем как Parquet
parquet_path = "data/processed/processed_for_feast.parquet"
df.to_parquet(parquet_path, index=False)

print(f"✅ Создан Parquet файл: {parquet_path}")
print(f"   Колонки: {len(df.columns)} шт.")
print("   Первые 5 колонок:", df.columns[:5].tolist())
print(f"   Пример ключей: {df['currency_date_id'].iloc[:3].tolist()}")