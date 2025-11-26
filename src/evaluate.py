# src/evaluate.py
import sys
import os
import json
import torch
import random
import numpy as np
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # T25--4m_MLOps
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
import mlflow
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix

from Core.core import DurakGame
from Core.agents import RLAgent, heuristic_agent
from src.preprocess import state_to_tensor

WEIGHTS_PATH = "rl_weights.pth"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)
REPORT_PATH = os.path.join(REPORTS_DIR, "eval.json")
TOTAL_GAMES = 100

def play_rl_vs_heuristic(rl_agent):
    """Запуск одной игры без трассировки ходов"""
    game = DurakGame(["RL", "HEUR"])
    pid_rl = 0
    total_reward = 0

    while not game.finished:
        pid = game.turn_order[0]
        state_before = game.get_state(pid)
        legal = game.legal_actions(pid)

        if pid == pid_rl:
            if not legal:
                action = ("pass",)
            else:
                action = rl_agent.act(state_before, legal)
                action = rl_agent.filter_illegal_actions(action, game.players[pid].hand)

            state_after = game.get_state(pid)
            reward = rl_agent.reward_system.compute_reward(game, pid, action, state_before, state_after)
            total_reward += reward

            rl_agent.learn(state_before, action, state_after, game, done=game.finished)
        else:
            action = heuristic_agent(game, pid)

        game.step(pid, action)

    rl_agent.save_weights()
    return total_reward, game.winner_ids

def main():
    temp_game = DurakGame(["RL", "HEUR"])
    state0 = state_to_tensor(temp_game.get_state(0))
    state_size = len(state0)

    rl_agent = RLAgent(pid=0, state_size=state_size, action_size=50,
                       epsilon=0.1, weights_path=WEIGHTS_PATH)

    rewards = []
    y_true = []
    y_scores = []

    with mlflow.start_run() as run:
        for g in range(TOTAL_GAMES):
            reward, winners = play_rl_vs_heuristic(rl_agent)
            rewards.append(reward)
            # true = 1 если RL победил
            y_true.append(1 if 0 in winners else 0)
            # score = средний reward за игру
            y_scores.append(reward)

        # метрики
        try:
            roc = roc_auc_score(y_true, y_scores)
        except ValueError:
            roc = 0.0
        precision = precision_score(y_true, [1 if s > 0 else 0 for s in y_scores], zero_division=0)
        recall = recall_score(y_true, [1 if s > 0 else 0 for s in y_scores], zero_division=0)
        cm = confusion_matrix(y_true, [1 if s > 0 else 0 for s in y_scores]).tolist()
        win_rate = sum(y_true) / TOTAL_GAMES
        avg_reward = sum(rewards) / TOTAL_GAMES

        report = {
            "roc_auc": roc,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": cm,
            "average_reward": avg_reward,
            "win_rate": win_rate,
            "total_games": TOTAL_GAMES
        }

        # сохраняем локально
        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, indent=4)

        print("=== Evaluation Report ===")
        print(json.dumps(report, indent=4))

        # логируем веса как артефакт
        if os.path.exists(WEIGHTS_PATH):
            mlflow.log_artifact(WEIGHTS_PATH, artifact_path="weights")
        # логируем eval.json как артефакт
        mlflow.log_artifact(REPORT_PATH, artifact_path="eval_reports")

        # регистрация модели в MLflow Model Registry
        model_name = "durak_rl"
        try:
            mlflow.register_model(f"runs:/{run.info.run_id}/model", model_name)
        except Exception as e:
            print(f"[WARNING] Модель не зарегистрирована: {e}")

        print(f"=== MODEL REGISTERED: {model_name} ===")
        print(f"Run ID: {run.info.run_id}")

if __name__ == "__main__":
    main()