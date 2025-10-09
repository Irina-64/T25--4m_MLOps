import os
import pandas as pd
import numpy as np

# Read raw data
raw_path = 'data/raw/toxic_comments_sample.csv'
df = pd.read_csv(raw_path)

# Handle NaN: drop rows with NaN in 'Toxic Text' or 'Non-Toxic Analog'
df = df.dropna(subset=['Toxic Text', 'Non-Toxic Analog'])

# Feature engineering (для детоксификации: фичи на основе текста и уровня)
df['toxic_length'] = df['Toxic Text'].str.len()  # Длина токсичного текста
df['non_toxic_length'] = df['Non-Toxic Analog'].str.len()  # Длина аналога
df['toxicity_level'] = df['Toxicity Level'].astype(int)  # Лейбл как int
df['is_high_toxicity'] = np.where(df['Toxicity Level'] > 3, 1, 0)  # Бинарный флаг: высокая токсичность

# Save processed (добавляем все колонки)
os.makedirs('data/processed', exist_ok=True)
processed_path = 'data/processed/processed.csv'
df.to_csv(processed_path, index=False)

print(f'Processed shape: {df.shape}')
print(df.head())