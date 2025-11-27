В этой лабораторной мы вынесли признаки в Feature Store (Feast) и заменили локальное чтение CSV в обучении на выборку из оффлайн стора. Подготовили `feature_repo/` с конфигом `feature_store.yaml` (offline=file, online=sqlite), определили entity `sample_id` и `FeatureView` `detox_features` c признаками `split`, `input_text`, `target_text`. Из `data/processed/train.csv` и `test.csv` сформировали оффлайн-таблицу `data/features/detox_features.csv` с `event_timestamp` и выполнили materialize, после чего `src/train.py` собирает обучающую и тестовую выборки через `FeatureStore.get_historical_features`, а не напрямую из файлов.

Инструкция по запуску:
- Подготовка окружения: `pip install -r requirements.txt`.
- Сборка оффлайн-таблицы признаков (если нужно пересоздать):  
  ```bash
  python - <<'PY'
  import pandas as pd
  from pathlib import Path
  base = Path('data/features'); base.mkdir(parents=True, exist_ok=True)
  train = pd.read_csv('data/processed/train.csv'); test = pd.read_csv('data/processed/test.csv')
  train['split'] = 'train'; test['split'] = 'test'
  full = pd.concat([train, test], ignore_index=True)
  full.insert(0, 'sample_id', range(1, len(full)+1))
  full['event_timestamp'] = pd.Timestamp('2020-06-01')
  cols = ['sample_id','split','input_text','target_text','event_timestamp']
  base.joinpath('detox_features.csv').write_text(full[cols].to_csv(index=False))
  PY
  ```
- Настройка и материализация Feast (запускать из корня):  
  ```
  cd feature_repo
  feast apply
  feast materialize 2020-01-01 2020-12-31
  cd ..
  ```
- Быстрая проверка оффлайн стора (опционально, но полезно):  
  ```
  python - <<'PY'
  import pandas as pd
  from feast import FeatureStore
  store = FeatureStore(repo_path='feature_repo')
  ent = pd.read_csv('data/features/detox_features.csv')[['sample_id','event_timestamp']]
  df = store.get_historical_features(
      entity_df=ent,
      features=['detox_features:split','detox_features:input_text','detox_features:target_text']
  ).to_df()
  print(df.head())
  PY
  ```
- Обучение: `python src/train.py`. Скрипт забирает фичи через `FeatureStore.get_historical_features`, обучает T5, логирует в MLflow и сохраняет модель/токенизатор в `models/detox_model`.
- Артефакты для проверки: `data/features/detox_features.csv`, `feature_repo/data/registry.db` (и sqlite online store), `models/detox_model/`, логи тренировки.

Распределение задач:
Власюк Данил: Настроил конфигурацию Feast (`feature_store.yaml`), описал offline/online store и параметры проекта.

Скрыпник Михаил: Подготовил оффлайн-таблицу признаков `data/features/detox_features.csv` из обработанных train/test, добавил идентификаторы и временные метки.

Беспалый Максим: Интегрировал Feast в `src/train.py`, заменил чтение CSV на `FeatureStore.get_historical_features`, проверил сборку датасетов и работу метрик.

Провков Иван: Выполнил `feast apply`/`materialize`, проверил возврат фичей из оффлайн стора и наличие артефактов в `feature_repo/data`.

Яковенко Максим: Подготовил отчёт по лабораторной, систематизировал шаги запуска и артефакты.
