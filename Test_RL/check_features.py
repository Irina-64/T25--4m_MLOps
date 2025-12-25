# Test_RL/check_features.py
import pandas as pd
import os

CSV_PATH = "rl_state_features.csv"

if not os.path.exists(CSV_PATH):
    print(f"Файл {CSV_PATH} не найден. Запусти матч RL-агент vs бот сначала.")
    exit()

# читаем CSV
df = pd.read_csv(CSV_PATH)

print(f"=== INFO ===")
print(f"Количество записей: {len(df)}")
print(f"Колонки: {df.columns.tolist()}")
print(df.head(5))  # первые 5 строк

# --- Визуализация структуры фич ---
import ast
import matplotlib.pyplot as plt

# преобразуем строку списка обратно в список
if "features" in df.columns:
    df["features_list"] = df["features"].apply(lambda x: ast.literal_eval(x))
    # длина фич
    feature_lengths = df["features_list"].apply(len)
    print(f"Длина векторов фич: min={feature_lengths.min()}, max={feature_lengths.max()}")
    
    # пример визуализации первой фичи
    plt.figure(figsize=(10,4))
    plt.plot(df["features_list"].iloc[0])
    plt.title("Пример фичей состояния (первый шаг)")
    plt.xlabel("Индекс фичи")
    plt.ylabel("Значение")
    plt.show()
