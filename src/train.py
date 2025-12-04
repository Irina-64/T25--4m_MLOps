# src\train.py
"""
src/train.py — обучение RL-агента для игры в дурака с полным логированием через MLflow.
Использует play_demo_game() из Core/demo.py без изменений.
Логирует:
- reward RL-агента по шагам
- метрики win_rate и avg_turns
- модель RL
- реплеи
"""
import sys, os

# Добавляет корневую директорию проекта в PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
from pathlib import Path
import mlflow
import os
import json
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Core.demo import play_demo_game
from Core.agents import RLAgent, heuristic_agent, random_agent

RL_NAME_IN_GAME = "RL"  # имя RL-агента в play_demo_game

# ---------------- Обёртка RLAgent для логирования шагов ----------------
class RLAgentMLflow(RLAgent):
    def __init__(self, pid, state_size, action_size, epsilon=0.1, run_id=None):
        super().__init__(pid, state_size, action_size, epsilon)
        self.run_id = run_id
        self.step_count = 0

    def learn_and_log(self, state, action_idx, reward, next_state, done, gamma=0.99):
        # стандартное обучение
        super().learn(state, action_idx, reward, next_state, done, gamma)
        # логирование reward через MLflow
        if self.run_id is not None:
            mlflow.log_metric("reward", reward, step=self.step_count)
            self.step_count += 1

# ---------------- Подсчёт метрик по реплеям ----------------
def compute_rl_metrics(replay_path):
    """Считает победы RL-агента и длину партии."""
    with open(replay_path, "r", encoding="utf-8") as f:
        replay = json.load(f)
    winners = replay.get("winners", [])
    rl_win = 1 if RL_NAME_IN_GAME in winners else 0
    total_turns = len(replay.get("steps", []))
    return rl_win, total_turns

# ---------------- Основной скрипт ----------------
def main():
    mlflow.set_experiment("Durak_RL_Training")
    n_games = 5
    epsilon = 0.1
    agents_setup = [random_agent, heuristic_agent, RLAgentMLflow]
    agent_names = ["Random", "Heuristic", "RLAgent"]

    total_rl_wins = 0
    total_turns = 0

    with mlflow.start_run(run_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

        mlflow.log_param("episodes", n_games)
        mlflow.log_param("epsilon", epsilon)
        mlflow.log_param("agents", agent_names)

        for i in range(n_games):
            print(f"\n🎮 --- Запуск игры {i+1}/{n_games} ---")

            # play_demo_game создаёт агентов RLAgentMLflow с run_id и логирует reward по шагам
            play_demo_game(agents_setup, names=["Bot1", "Bot2", "RL"], epsilon=epsilon)

            # логируем модель RL
            if os.path.exists("rl_agent_model.pth"):
                mlflow.log_artifact("rl_agent_model.pth", artifact_path=f"models/game_{i+1}")

            # логируем реплей
            replays_dir = Path("replays")
            latest_replays = sorted(replays_dir.glob("replay_*.json"))
            if latest_replays:
                replay_path = latest_replays[-1]
                mlflow.log_artifact(str(replay_path), artifact_path=f"replays/game_{i+1}")

                # считаем метрики
                rl_win, turns = compute_rl_metrics(replay_path)
                total_rl_wins += rl_win
                total_turns += turns

        # логируем итоговые метрики
        rl_win_rate = total_rl_wins / n_games
        avg_turns = total_turns / n_games if n_games else 0
        mlflow.log_metric("rl_win_rate", rl_win_rate)
        mlflow.log_metric("avg_turns_per_game", avg_turns)
        mlflow.log_metric("num_games", n_games)

        print("\n✅ Обучение завершено. Данные сохранены в MLflow.")
        print(f"RL win rate: {rl_win_rate:.2f}, avg turns per game: {avg_turns:.1f}")


if __name__ == "__main__":
    main()
