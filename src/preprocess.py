import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


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


def prepare_for_feast(df: pd.DataFrame) -> pd.DataFrame:
    df_feast = df.copy()
    df_feast["user_id"] = range(1, len(df_feast) + 1)
    current_time = datetime.now()
    
    df_feast["event_timestamp"] = current_time
    df_feast["created_timestamp"] = current_time
    column_mapping = {
        "Time_broken_spent_Alone": "time_broken_spent_alone",
        "Stage_fear": "stage_fear", 
        "Social_event_attendance": "social_event_attendance",
        "Going_outside": "going_outside",
        "Drained_after_socializing": "drained_after_socializing",
        "Friends_circle_size": "friends_circle_size",
        "Post_frequency": "post_frequency",
        "Personality": "personality",
        "Personality_encoded": "personality_encoded"
    }
    
    df_feast = df_feast.rename(columns=column_mapping)
    numeric_features = [
        "time_broken_spent_alone", "social_event_attendance", 
        "going_outside", "friends_circle_size", "post_frequency"
    ]
    
    for col in numeric_features:
        if col in df_feast.columns:
            df_feast[col] = pd.to_numeric(df_feast[col], errors='coerce')
    
    return df_feast


def save_data(df: pd.DataFrame, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Данные сохранены в {output_path}")
    print(f"Размер: {len(df)} строк, {len(df.columns)} колонок")


def main():
    input_path = "/opt/airflow/data/raw/pattern_sample.csv"
    output_path = "/opt/airflow/data/processed/processed.csv"
    feast_output_path = "/opt/airflow/data/processed/processed_for_feast.csv"

    try:
        df = load_data(input_path)
        df_clean = clean_data(df)
        save_data(df_clean, output_path)
        df_feast = prepare_for_feast(df_clean)
        save_data(df_feast, feast_output_path)
        
        import pandas as pd
        parquet_path = "/opt/airflow/data/processed/personality_features.parquet"
        df_feast.to_parquet(parquet_path, index=False)
        print(f"✅ Parquet файл создан для Feast: {parquet_path}")
        print(f"   Колонки: {df_feast.columns.tolist()}")
        print(f"   Размер: {len(df_feast)} строк")
        
        if 'event_timestamp' in df_feast.columns:
            print(f"   event_timestamp пример: {df_feast['event_timestamp'].iloc[0]}")
        if 'created_timestamp' in df_feast.columns:
            print(f"   created_timestamp пример: {df_feast['created_timestamp'].iloc[0]}")
        
    except Exception as e:
        print(f"❌ Ошибка: {str(e)}")
        raise


if __name__ == "__main__":
    main()