# feature_repo/check_feast.py
from feast import FeatureStore
import pandas as pd

print("=== ПРОВЕРКА FEAST API ===")

try:
    # 1. Инициализация
    print("\n1. 🚀 ИНИЦИАЛИЗАЦИЯ FEAST...")
    store = FeatureStore(repo_path='.')
    print("   ✅ Feature Store загружен")
    
    # 2. Feature Views
    print("\n2. 📊 FEATURE VIEWS:")
    feature_views = store.list_feature_views()
    print(f"   Найдено: {len(feature_views)} шт.")
    for fv in feature_views:
        print(f"   - {fv.name}")
    
    # 3. Проверка данных - ИСПРАВЛЕННАЯ ВЕРСИЯ!
    print("\n3. 🔍 ПРОВЕРКА ДАННЫХ (ИСПРАВЛЕННАЯ)...")
    
    # Загружаем ваши данные
    df = pd.read_parquet('data/currency_data.parquet')
    print(f"   Загружен currency_data.parquet: {df.shape}")
    
    # Берем первые 3 record_id И их даты!
    df_sample = df.head(3).copy()
    
    # ПРАВИЛЬНЫЙ entity_df с event_timestamp!
    entity_df = pd.DataFrame({
        'record_id': df_sample['record_id'].tolist(),
        'event_timestamp': df_sample['date'].tolist()  # ← ВАЖНО: event_timestamp!
    })
    
    print(f"   Пример record_id: {entity_df['record_id'].tolist()}")
    print(f"   Пример дат: {entity_df['event_timestamp'].head().tolist()}")
    
    # Получаем данные из Feast
    features = store.get_historical_features(
        entity_df=entity_df,
        features=[
            'currency_features:USD_RUB',
            'currency_features:EUR_RUB',
            'currency_features:GBP_RUB',
            'currency_features:day_of_week',
            'currency_features:is_weekend'
        ]
    )
    
    result = features.to_df()
    print(f"\n   ✅ ДАННЫЕ ПОЛУЧЕНЫ ИЗ FEAST!")
    print(f"      Размер: {result.shape}")
    print(f"      Колонки: {result.columns.tolist()}")
    
    if len(result) > 0:
        print(f"\n      ПРИМЕР ДАННЫХ:")
        # Показываем только ключевые колонки
        display_cols = ['record_id', 'event_timestamp', 'USD_RUB', 'EUR_RUB']
        available_cols = [c for c in display_cols if c in result.columns]
        print(result[available_cols].head())
    else:
        print("\n      ⚠️  Получен пустой DataFrame!")
        
except Exception as e:
    print(f"\n❌ ОШИБКА: {e}")
    import traceback
    traceback.print_exc()