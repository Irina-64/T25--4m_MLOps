Проект по разработке и обучению игрового ии (оппонента) на основе ml в карточной игре дурак.

├── Core/ # Ядро игры и RL
│ ├── core.py # Логика игры в Дурака
│ ├── agents.py # RLAgent, эвристический и случайный агенты
│ ├── demo.py # Демонстрационная игра с человеком
│ ├── replay_logger.py # Логирование реплеев
│ └── reward_system.py # Система наград для RL
│
├── env/
│ └── durak_env.py # Gym-среда для RL с человеческим игроком
│
├── src/
│ ├── Api.py # FastAPI для игры через HTTP-запросы
│ ├── preprocess.py # Кодирование состояния и карт в тензоры
│ ├── train.py # Скрипт обучения RL-агента с MLflow
│ ├── evaluate.py # Скрипт оценки агента vs эвристика
│ └── register_model.py # Регистрация модели в MLflow Model Registry
│
├── Test_RL/ # Юнит-тесты и тестовые сессии
│ ├── test_one_game.py
│ ├── test_rl_agent.py
│ ├── test_rl_vs_heuristic.py
│ └── test_rl_vs_random.py
│
├── data/
│ ├── raw/
│ └── processed/
├── .dvc/
├── mlruns/ # MLflow эксперименты и модели
├── rl_weights.pth # Сохранённые веса RL-агента
├── environment.yml # Conda environment
├── requirements.txt # pip зависимости
├── Dockerfile # Для контейнеризации проекта
└── README.md

Демонстрационная игра

Запуск игры с человеком через консоль: python Core/demo.py


RL-агент
Core/agents.py содержит RLAgent:
  -Выбор действий с ε-greedy
  -Функции обучения learn
  -Сохранение и загрузка весов
Эвристический бот (heuristic_agent) и случайный бот (random_agent) для тестов.


API для игры через HTTP

FastAPI сервер (src/Api.py) предоставляет эндпоинты:

Метод	URL	        Описание
GET	/reset	        Сброс игры и получение начального состояния
POST	/step	        Сделать ход, принимает JSON { "action": "6♣" }
GET	/legal_actions	Получить список легальных действий

Запуск сервера: python src/Api.py




Тестирование
Папка Test_RL содержит скрипты:
  -test_one_game.py — одиночная игра RL vs heuristic
  -test_rl_agent.py — базовые проверки RLAgent
  -test_rl_vs_heuristic.py — RL vs heuristic в цикле

Предобработка карт и состояния
src/preprocess.py:
  -Кодирование карт в one-hot векторы
  -Кодирование состояния игры в тензор для RL
  -state_to_tensor возвращает тензор для подачи в нейросеть агента

Система наград
Core/reward_system.py определяет бонусы и штрафы:
  -За уменьшение руки, успешную защиту
  -За комбинации карт (пары, тройки, четверки)
  -За незаконные ходы (ILLEGAL_MOVE_PENALTY)
  -Бонус за “погоны” (пара шестерок)

Реплеи
Логирование реплеев в JSON через Core/replay_logger.py
Сохраняются:
  -Ходы всех игроков
  -Состояние до и после хода
  -Победители и метаданные игры
  -Используются для анализа и обучения


Весовые файлы агента (rl_weights.pth) хранятся в корне

MLflow хранит артефакты и модель в mlruns/

DVC используется для версионирования данных и моделей (.dvc/)

Контейнеризация
Для запуска проекта в Docker:
  -docker build -t durak_rl .
  -docker run -p 8000:8000 durak_rl

        -Кодирование карт в one-hot векторы
        -Кодирование состояния игры в тензор для RL
        -state_to_tensor возвращает тензор для подачи в нейросеть агента
