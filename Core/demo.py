# Core/demo.py
"""
Улучшённый demo: поддерживает запуск как модуль (py -m Core.demo) или как скрипт.
Добавлены:
 - CLI (episodes, opponent, epsilon, log-mlflow)
 - MLflow logging (one run per whole evaluation/training session)
 - Сохранение модели и реплеев и логирование артефактов
 - Надёжный импорт Core при запуске из корня или напрямую
"""

import os
import sys
import random
import argparse
import hashlib
from datetime import datetime

import torch

# --- allow running as "python Core/demo.py" or "python -m Core.demo"
if __package__ is None:
    # add repo root to path so absolute imports work
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from Core.core import DurakGame
from Core.replay_logger import ReplayLogger
from Core.agents import RLAgent, heuristic_agent, random_agent

# default model filename
MODEL_PATH = "rl_agent_model.pth"


def format_hand(hand, trump_suit):
    """Возвращает список карт игрока, помечая козыри."""
    return [f"{card}{' (trump)' if card.suit == trump_suit else ''}" for card in hand]


def file_hash(path: str):
    """Возвращает SHA1-хеш файла (или None, если файла нет)."""
    if not os.path.exists(path):
        return None
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def play_one_game(agents_list, names=None, epsilon=0.1, save_replay=True, model_path=MODEL_PATH):
    """
    Запускает одну игру и возвращает dict с результатами:
    { "winners": [...], "length": steps, "replay_path": path, "win_rl": 0/1, "reward_sum": float }
    """
    g = DurakGame(names or [f"Bot{i}" for i in range(len(agents_list))])
    logger = ReplayLogger(g)

    agents = {}
    for i, agent in enumerate(agents_list):
        if isinstance(agent, type):
            state_dict = g.get_state(i)
            state_size = len(RLAgent.state_to_tensor(None, state_dict))
            action_size = 50
            agents[i] = agent(pid=i, state_size=state_size, action_size=action_size, epsilon=epsilon)
            if os.path.exists(model_path):
                try:
                    agents[i].model.load_state_dict(torch.load(model_path))
                    print(f"[INFO] RLAgent {i} модель загружена из {model_path}")
                except Exception as e:
                    print(f"[WARN] Не удалось загрузить модель {model_path}: {e}")
        else:
            agents[i] = agent

    trump_suit = g.trump_suit
    # логируем начальные руки в реплей
    logger.replay["initial_hands"] = {player.name: format_hand(player.hand, trump_suit) for player in g.players}

    steps = 0
    total_reward = 0.0

    while not g.finished:
        for pid in list(g.turn_order):
            legal = g.legal_actions(pid)
            if not legal:
                continue

            state_before = g.get_state(pid)
            agent = agents[pid]

            if isinstance(agent, RLAgent):
                action = agent.act(g, pid)
                action = agent.filter_illegal_actions(action, g.players[pid].hand)
            else:
                action = agent(g, pid)

            g.step(pid, action)
            state_after = g.get_state(pid)

            if isinstance(agent, RLAgent):
                reward = agent.reward_system.compute_reward(g, pid, action, state_before, state_after)
                # выбираем индекс действия для learn: здесь используется ближайший индекс в legal
                try:
                    legal = g.legal_actions(pid)
                    action_idx = next(i for i, a in enumerate(legal) if a == action)
                except StopIteration:
                    action_idx = 0
                agent.learn(
                    agent.state_to_tensor(state_before),
                    action_idx,
                    reward,
                    agent.state_to_tensor(state_after),
                    g.finished,
                )
                total_reward += reward

            # логируем шаг
            hands_after = {player.name: format_hand(player.hand, trump_suit) for player in g.players}
            logger.log_step(pid, action, state_before, state_after, hands_after)

            steps += 1

            # (опционально) печать
            # print(f"{g.players[pid].name} делает ход: {action}")

            if g.finished:
                break

    # финал: сохраняем модель (если есть RL agent)
    for agent in agents.values():
        if isinstance(agent, RLAgent):
            torch.save(agent.model.state_dict(), model_path)

    # сохраняем replay
    if save_replay:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("replays", exist_ok=True)
        replay_path = f"replays/replay_{timestamp}.json"
        logger.finalize(replay_path)
    else:
        replay_path = None

    winners = [g.players[i].name for i in g.winner_ids]
    win_rl = 1 if any("RL" in w for w in winners) else 0

    return {
        "winners": winners,
        "length": steps,
        "replay_path": replay_path,
        "win_rl": win_rl,
        "reward_sum": total_reward,
    }


def run_sessions(episodes: int, opponent: str, epsilon: float, log_mlflow: bool, model_path=MODEL_PATH):
    """
    Запуск множества игр, логирование (MLflow опционален).
    opponent: "random" or "heuristic"
    """
    import mlflow

    # формируем конфиг агентов: RL (pid=0) и оппонент (pid=1)
    if opponent == "random":
        opponents = [random_agent]
    else:
        opponents = [heuristic_agent]

    agents_setup = [type] * 2  # placeholder: we'll map types to actual callables below

    # Prepare MLflow
    if log_mlflow:
        mlflow.set_experiment("durak-training")
        run = mlflow.start_run()
        run_id = run.info.run_id
        print(f"[MLFLOW] run_id: {run_id}")
    else:
        run = None
        run_id = None

    total_wins = 0
    total_len = 0
    total_reward = 0.0
    replay_paths = []

    # initial model hash (before training)
    before_hash = file_hash(model_path)
    if before_hash:
        print(f"[INFO] model hash before session: {before_hash}")

    for ep in range(episodes):
        # build agent list: RLAgent class + opponent callable
        agents_list = [RLAgent, opponents[0]]
        names = ["RL", "OPP"]

        res = play_one_game(agents_list, names=names, epsilon=epsilon, save_replay=True, model_path=model_path)
        total_wins += res["win_rl"]
        total_len += res["length"]
        total_reward += res["reward_sum"]
        if res["replay_path"]:
            replay_paths.append(res["replay_path"])

        # log per-episode metrics
        if log_mlflow:
            mlflow.log_metric("ep_win", res["win_rl"], step=ep)
            mlflow.log_metric("ep_length", res["length"], step=ep)
            mlflow.log_metric("ep_reward", res["reward_sum"], step=ep)
            mlflow.log_param(f"episode_{ep}_replay", res["replay_path"])

        print(f"[Episode {ep+1}/{episodes}] win={res['win_rl']} len={res['length']} reward={res['reward_sum']:.2f}")

    # final model hash (after training)
    after_hash = file_hash(model_path)
    if after_hash:
        print(f"[INFO] model hash after session: {after_hash}")

    # aggregate metrics
    win_rate = total_wins / episodes if episodes else 0.0
    avg_len = total_len / episodes if episodes else 0.0
    avg_reward = total_reward / episodes if episodes else 0.0

    if log_mlflow:
        mlflow.log_metric("win_rate", win_rate)
        mlflow.log_metric("avg_length", avg_len)
        mlflow.log_metric("avg_reward", avg_reward)

        # log model artifact
        if os.path.exists(model_path):
            mlflow.log_artifact(model_path, artifact_path="models")
        # log last N replays (or all)
        for p in replay_paths[-5:]:
            if p and os.path.exists(p):
                mlflow.log_artifact(p, artifact_path="replays")

        mlflow.end_run()

    # print summary
    summary = {
        "episodes": episodes,
        "opponent": opponent,
        "epsilon": epsilon,
        "win_rate": win_rate,
        "avg_length": avg_len,
        "avg_reward": avg_reward,
        "model_hash_before": before_hash,
        "model_hash_after": after_hash,
        "replays_logged": len(replay_paths),
        "run_id": run_id,
    }
    print("\n=== Session summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Play Durak demo games (RL training) with MLflow logging.")
    p.add_argument("--episodes", type=int, default=1, help="Number of episodes/games to run")
    p.add_argument("--opponent", choices=["random", "heuristic"], default="random", help="Type of opponent")
    p.add_argument("--epsilon", type=float, default=0.1, help="RL agent epsilon")
    p.add_argument("--log-mlflow", action="store_true", help="Enable MLflow logging")
    p.add_argument("--model-path", type=str, default=MODEL_PATH, help="Path to save/load RL model")
    return p.parse_args()


def main():
    args = parse_args()
    run_info = run_sessions(episodes=args.episodes, opponent=args.opponent, epsilon=args.epsilon, log_mlflow=args.log_mlflow, model_path=args.model_path)
    # optionally, save a summary report locally
    os.makedirs("reports", exist_ok=True)
    rep_path = os.path.join("reports", f"train_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    import json
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Summary saved to {rep_path}")


if __name__ == "__main__":
    main()
