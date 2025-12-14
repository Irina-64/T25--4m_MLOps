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

## Docker
```
docker build -t personality-api:lab6 .
docker run -p 8080:8080 personality-api:lab6
```
## API-test
```
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

## Лицензия
MIT