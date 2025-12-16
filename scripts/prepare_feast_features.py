import pandas as pd

# Загружаем обработанные данные
df = pd.read_csv("data/processed/processed.csv")

# Переименовываем ID под Feast
df = df.rename(columns={"customerID": "customer_id"})

# Добавляем обязательный timestamp
df["event_timestamp"] = pd.to_datetime("2020-01-01")

# Убираем target
df = df.drop(columns=["Churn"])

# Сохраняем в Feast data
df.to_csv("feature_repo/data/telco_features.csv", index=False)

print("✅ Feast features prepared")
print(df.head())
