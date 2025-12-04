# Test_RL/test_rl_vs_heuristic.py
import sys
import os
import torch
import random

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from Core.core import DurakGame
from Core.agents import RLAgent, heuristic_agent
from src.preprocess import state_to_tensor

WEIGHTS_PATH = "rl_weights.pth"

def play_rl_vs_heuristic(rl_agent):
    """Запуск одной игры без трассировки ходов"""
    player_names = ["RL", "HEUR"]
    game = DurakGame(player_names)
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

if __name__ == "__main__":
    # RLAgent создаём один раз, state_size берём из игры
    temp_game = DurakGame(["RL", "HEUR"])
    state0 = state_to_tensor(temp_game.get_state(0))
    state_size = len(state0)

    # проверка, загружены ли старые веса
    loaded_existing_weights = False
    if os.path.exists(WEIGHTS_PATH):
        loaded_state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
        first_layer_weight = loaded_state_dict[list(loaded_state_dict.keys())[0]]
        if first_layer_weight.shape[1] == state_size:
            loaded_existing_weights = True

    rl_agent = RLAgent(pid=0, state_size=state_size, action_size=50, epsilon=0.1,
                       weights_path=WEIGHTS_PATH)

    print(f"Использовались старые веса: {loaded_existing_weights}")

    # запускаем несколько эпизодов
    for episode in range(100):
        total_reward, winners = play_rl_vs_heuristic(rl_agent)
        print(f"=== EPISODE {episode + 1} ===")
        print("Победители:", winners)
        print("Общий reward RL:", total_reward)
