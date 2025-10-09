# MLOps для детоксификации текста

Проект по созданию пайплайна для детекции и нейтрализации токсичного текста с использованием моделей вроде Detoxify и Transformers.

## Установка
- Создайте venv: python3.10 -m venv text_detox
- Активируйте: source text_detox/bin/activate (Linux/macOS)
- Установите: pip install -r requirements.txt

## Структура
- src/: Исходный код (препроцессинг, обучение).
- data/: Датасеты (используйте dvc add для версионирования).
- models/: Сохраненные модели.
- tests/: Тесты.