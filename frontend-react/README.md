# AI Detox - React Frontend

Современный React интерфейс для AI Detox с поддержкой темной/светлой темы.

## Установка

```bash
cd frontend-react
npm install
```

## Разработка

Запуск dev сервера (с hot reload):

```bash
npm run dev
```

Приложение будет доступно на `http://localhost:3000`

**Важно:** Убедитесь, что API сервер запущен на `http://localhost:8080`

## Сборка для продакшена

```bash
npm run build
```

Собранные файлы будут в папке `../frontend-build`

После сборки API автоматически будет раздавать React приложение.

## Запуск с API

1. Соберите React приложение:
   ```bash
   npm run build
   ```

2. Запустите API сервер:
   ```bash
   python3 -m uvicorn src.api:app --host 0.0.0.0 --port 8080
   ```

3. Откройте в браузере: `http://localhost:8080`

## Особенности

- ✅ Современный интерфейс в стиле ChatGPT
- ✅ Темная/светлая тема с переключателем
- ✅ Адаптивный дизайн
- ✅ Плавные анимации
- ✅ Настройки модели (max_length, num_beams)
- ✅ История сообщений
- ✅ Обработка ошибок

## Структура проекта

```
frontend-react/
├── src/
│   ├── components/      # React компоненты
│   ├── App.jsx          # Главный компонент
│   ├── main.jsx         # Точка входа
│   └── index.css        # Глобальные стили
├── index.html
├── package.json
└── vite.config.js
```


