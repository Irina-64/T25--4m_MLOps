# Test_RL/visualize_features.py
import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import os

CSV_PATH = "rl_state_features.csv"

if not os.path.exists(CSV_PATH):
    print(f"Файл {CSV_PATH} не найден. Сначала запусти матч RL-агент vs бот.")
    exit()

# читаем CSV
df = pd.read_csv(CSV_PATH)

# преобразуем строку списка обратно в список
df["features_list"] = df["features"].apply(lambda x: ast.literal_eval(x))

# --- базовая статистика ---
feature_lengths = df["features_list"].apply(len)
print(f"Длина векторов фич: min={feature_lengths.min()}, max={feature_lengths.max()}")
print("Пример первой строки фич:", df["features_list"].iloc[0][:20], "...")

# --- распределение значений фич ---
all_features = [f for sublist in df["features_list"] for f in sublist]

plt.figure(figsize=(10,5))
sns.histplot(all_features, bins=50, kde=True)
plt.title("Распределение всех значений фич по всем шагам")
plt.xlabel("Значение фичи")
plt.ylabel("Частота")
plt.show()

# --- пример фич одного шага ---
plt.figure(figsize=(12,4))
plt.plot(df["features_list"].iloc[0])
plt.title("Фичи первого состояния (эпизод 1, шаг 0)")
plt.xlabel("Индекс фичи")
plt.ylabel("Значение")
plt.show()
