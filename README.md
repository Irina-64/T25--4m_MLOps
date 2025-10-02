# Used cars price predict

Проект для предсказания цен на подержанные автомобили.

## Описание

Использование нейронных сетей для предсказания цен на автомобили с пробегом. Данные получены путем парсинга сайта auto.ru

## Запуск

1. Установите окружение:
```bash
pip install -r requirements.txt
```

## Структура проекта

```bash

.
├── data
│   ├── processed             # Обработанные данные
│   └── raw                   # Исходные данные
│       └── dataset.csv.dvc
├── models                    # Модели
├── src                       # Файлы приложения
│   └── preprocess.py         # Предобработка датасета
├── tests                     # Тесты
├── dvc.lock
├── dvc.yaml
├── LICENSE
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock
```

### Предобработка данных:

```bash
python src/preprocess.py
```
