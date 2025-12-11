# Настройка React Frontend

## Быстрый старт

### 1. Установите зависимости

```bash
cd frontend-react
npm install
```

### 2. Запуск в режиме разработки

В одном терминале запустите React dev server:
```bash
cd frontend-react
npm run dev
```

В другом терминале запустите API:
```bash
python3 -m uvicorn src.api:app --host 0.0.0.0 --port 8080
```

React приложение будет доступно на `http://localhost:3000` с hot reload.

### 3. Сборка для продакшена

```bash
cd frontend-react
npm run build
```

После сборки файлы будут в `frontend-build/`. API автоматически раздаст их.

### 4. Запуск продакшен версии

```bash
# Сначала соберите React приложение
cd frontend-react
npm run build

# Затем запустите API
cd ..
python3 -m uvicorn src.api:app --host 0.0.0.0 --port 8080
```

Откройте `http://localhost:8080` в браузере.

## Docker

Для запуска через Docker, обновите `docker-compose.api.yaml` чтобы включить сборку React:

```yaml
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8080:8080"
    volumes:
      - ./models:/app/models
      - ./frontend-build:/app/frontend-build
    # ... остальное
```

И добавьте шаг сборки React в Dockerfile или соберите перед запуском Docker.

## Особенности React приложения

- **Современный UI** в стиле ChatGPT
- **Темная/светлая тема** с сохранением в localStorage
- **Адаптивный дизайн** для мобильных устройств
- **История сообщений** с возможностью очистки
- **Настройки модели** через панель настроек
- **Плавные анимации** и переходы

