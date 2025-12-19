# MLOps

## Описание
Проект для предсказания классификации экстраверт/интроверт на основе таких признаков как социальная активность, время в одиночестве и участие в мероприятиях.

## Старт
1. Клонировать репозиторий: 
   ```
   git clone git@github.com:LeeDef18/Team9.git
   cd team-9_mlops
   ```
2. Создать виртуальное окружение: `conda create -n mlops python=3.10 -y`
3. Активировать: `conda activate mlops`
4. Установить зависимости: `pip install -r requirements.txt`

# Docker
```
docker build -t personality-api:lab6 .
docker run -p 8080:8080 personality-api:lab6
```
## Локальный API-test
```
python -m uvicorn src.api:app --host 0.0.0.0 --port 8080 --reload
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Time_broken_spent_Alone": 4.0,
    "Stage_fear": 0,
    "Social_event_attendance": 4.0,
    "Going_outside": 6.0,
    "Drained_after_socializing": 0,
    "Friends_circle_size": 13.0,
    "Post_frequency": 5.0
  }'
```

# Airflow pipeline/Feature_store(Feast)
Мы добавили оркестрацию пайплайна preprocess `>> feast_materialize_task >> train_with_feast >> evaluate >> save_report`
вместе с хранилищем `feast` для признаков. `feature_repo` содержит конфигурацию `feature_store.yaml` вместе с описанием entity, featureView - `definitions.py`. Использовали формат parquet для датасета.

## Старт
Переходим в папку с докером
```
cd .\Airflow
```
Запускаем docker-compose.yaml - инициализация airflow, баз данных, пользователя.
```
docker compose up -d
```
Переходим в UI по ссылке: `http://localhost:8081` и входим по логину:`airflow` и паролю: `airflow`

Переходим в DAGs и ищем `feast_personality_pipeline` - написанный dag, который через BashOperator запускает `preprocess.py`, `train.py`, `evaluate.py`, и сохраняет отчет в `/opt/airflow/reports/feast_integration_report.txt`
Запускаем через `Trigger`, и, если все проходит без ошибок и метрики показывают значения выше порога, то `status_dag = Success`


## Лицензия
MIT