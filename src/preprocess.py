import pandas as pd
import numpy as np
from pathlib import Path


def load_data(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isna().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
    
    if 'Stage_fear' in df_clean.columns:
        df_clean['Stage_fear'] = df_clean['Stage_fear'].map({'Yes': 1, 'No': 0})
        df_clean['Stage_fear'] = df_clean['Stage_fear'].fillna(df_clean['Stage_fear'].mode()[0])
    
    if 'Drained_after_socializing' in df_clean.columns:
        df_clean['Drained_after_socializing'] = df_clean['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
        df_clean['Drained_after_socializing'] = df_clean['Drained_after_socializing'].fillna(
            df_clean['Drained_after_socializing'].mode()[0]
        )
    
    if 'Personality' in df_clean.columns:
        df_clean['Personality'] = df_clean['Personality'].str.strip()
        df_clean['Personality_encoded'] = df_clean['Personality'].map({'Extrovert': 1, 'Introvert': 0})
    
    return df_clean


def save_data(df: pd.DataFrame, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Данные сохранены в {output_path}")
    print(f"Размер: {len(df)} строк, {len(df.columns)} колонок")


def main():
    input_path = "data/raw/pattern_sample.csv"
    output_path = "data/processed/processed.csv"

    try:
        df = load_data(input_path)
        df_clean = clean_data(df)
        save_data(df_clean, output_path)
        
    except Exception as e:
        print(f"Ошибка: {e}")
        raise


if __name__ == "__main__":
    main()