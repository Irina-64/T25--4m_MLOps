# Лабораторная работа №10  
## Развёртывание модели Detox T5 в Kubernetes

### Цель работы
Развернуть REST API с моделью `t5-small` в Kubernetes-кластере (Docker Desktop Kubernetes), подготовить Deployment, Service, настроить контейнеризацию, собрать Docker-образ, выполнить деплой и проверить работоспособность эндпоинта `/predict`.

---

## Выполненные шаги

### 1. Подготовка Docker-образа
- Создан Dockerfile для запуска FastAPI-приложения с моделью `t5-small`.
- Корректно установлен PyTorch через официальный wheel (`index-url=https://download.pytorch.org/whl/cpu`), чтобы избежать ошибок хешей.
- Установлены зависимости: `transformers`, `datasets`, `accelerate`, `fastapi`, `uvicorn`, `pydantic` и др.
- Образ собран командой:

```bash
docker build -t flight-delay-api:lab10 .
```
2. Подготовка манифестов Kubernetes
Deployment (k8s/deployment.yaml)
    Запуск 1 реплики приложения.
    Образ: flight-delay-api:lab10
    Контейнер слушает порт 8080.
    Блок resources был убран, т.к. модель T5 требует значительно больше памяти.

Service (k8s/service.yaml)

Тип: NodePort
    Проброс порта 8080 внутри кластера.
    NodePort открыт на 30080.

HorizontalPodAutoscaler (k8s/hpa.yaml)
    Настроен HPA от 1 до 5 подов по CPU 50% (как по методичке).

Манифесты применены командой:

```bash
kubectl apply -f k8s/
```

3. Запуск Kubernetes-кластера
    В Docker Desktop включён Kubernetes.
    После устранения конфликтов (VPN → WSL2 → Docker Desktop) кластер успешно запустился.
    Проверка:
```bash
kubectl get nodes
```

Результат:
```bash
docker-desktop   Ready   control-plane
```

4. Проверка работы приложения в Kubernetes
После деплоя:
```bash
kubectl get pods
```

Результат:
```bash
flight-delay-xxxxxx   1/1   Running
```

В логах:
```csharp
[API] Модель загружена из: t5-small
Uvicorn running on http://0.0.0.0:8080
```

5. Публикация API через port-forward
```bash
kubectl port-forward service/flight-delay-svc 8080:8080
```
6. Тестирование эндпоинта /predict
Запрос:
```bash
Invoke-RestMethod `
  -Uri "http://localhost:8080/predict" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"text":"I hate this movie so much"}'
```

Ответ модели:
```json
{
  "detox_text": "Ich hate diesen Film so sehr."
}
```

API успешно работает через Kubernetes.

Распределение задач в команде

Яковенко Максим:
Настроил структуру k8s/, подготовил манифесты Deployment/Service/HPA, проверил корректность портов и запуска подов.

Скрыпник Михаил:
Исправил Dockerfile, добавил установку Torch через официальный wheel. Проверил сборку образа и совместимость зависимостей.

Беспалый Максим:
Отладил запуск FastAPI-приложения в контейнере, устранил OOMKilled, проверил работу Uvicorn и загрузку модели T5.

Провков Иван:
Поднял Kubernetes-кластер в Docker Desktop, устранил ошибки WSL2/VPN, выполнил деплой приложения, проверил работу /predict.

Власюк Данил:
Подготовил отчёт по лабораторной работе, описал этапы деплоя, структуру манифестов и демонстрацию результата.