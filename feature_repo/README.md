# Репозиторий Фич Feast — Пример

Эта папка демонстрирует минимальный репозиторий фич Feast для лабораторной работы.

## Структура

feature_repo/
├── feature_store.yaml
├── data/
│   └── features.csv
├── features/
│   ├── __init__.py
│   ├── entity.py
│   └── feature_views.py

## Быстрый старт

1. Установите Feast:

```bash
pip install feast
```

2. Инициализируйте (если ещё не сделано):

```bash
feast init feature_repo
```

3. Подготовьте фичи и выполните команды (убедитесь, что `feature_repo` является текущей директорией):

```bash
cd feature_repo
python ../src/feast_prepare_features.py --raw-path ../data/raw/churn_predict.csv --out-path data/features.csv
feast apply
# затем материализуйте диапазон (убедитесь, что ваши оффлайн-данные имеют временные метки в этом диапазоне)
feast materialize 2024-01-01 2024-01-31
```

`feast apply` регистрирует сущности и feature views в локальном реестре; `feast materialize` читает фичи из оффлайн-источников и записывает их в онлайн-хранилище (по умолчанию SQLite).
Оффлайн-источник — CSV-файл `data/features.csv` включённый в репозиторий.

# Примечания
- ример использует локальное онлайн-хранилище SQLite и локальный оффлайн-источник файлов. Реестр и файлы онлайн-хранилища будут храниться по умолчанию в `feature_repo/data`.
- Чтобы использовать другой оффлайн-источник (например, BigQuery), измените `feature_store.yaml` и определения источников данных.
