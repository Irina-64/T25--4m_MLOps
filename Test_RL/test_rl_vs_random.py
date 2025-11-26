# Test_RL/test_rl_vs_random.py
import sys
import os
import torch
import random
import time
import shutil

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from Core.core import DurakGame
from Core.agents import RLAgent, random_agent
from Core.reward_system import WIN_REWARD, LOSE_PENALTY
from src.preprocess import state_to_tensor
from Core.reward_system import TAKE_PENALTY

WEIGHTS_PATH = "rl_weights.pth"

def play_rl_vs_random_progress(rl_agent, game_index=1):
    """Игра RL vs RANDOM с прогресс-баром вместо детальной трассировки."""
    player_names = ["RL", "RANDOM"]
    game = DurakGame(player_names)
    pid_rl = 0

    total_reward = 0.0
    steps = 0

    # ширина консоли для прогресс-бара
    term_width = shutil.get_terminal_size((80, 20)).columns
    bar_width = max(20, term_width - 20)

    while not game.finished:
        steps += 1
        pid = game.turn_order[0]
        state_before = game.get_state(pid)
        legal = game.legal_actions(pid)

        if pid == pid_rl:
            if not legal:
                action = ("pass",)
            else:
                action = rl_agent.act(state_before, legal)
                action = rl_agent.filter_illegal_actions(action, game.players[pid].hand)

            game.step(pid, action)
            state_after = game.get_state(pid)

            reward = rl_agent.reward_system.compute_reward(
                game, pid, action, state_before, state_after
            )
            total_reward += reward

            # простенькое движение прогресс-бара
            filled = min(bar_width, steps % bar_width)
            bar = "[" + "#" * filled + "-" * (bar_width - filled) + "]"
            print(f"Game {game_index} {bar} step {steps}", end="\r")

            rl_agent.learn(state_before, action, state_after, game, done=game.finished)

        else:
            action = random_agent(game, pid)
            game.step(pid, action)

            # обновляем ту же строку (без вывода действий)
            filled = min(bar_width, steps % bar_width)
            bar = "[" + "#" * filled + "-" * (bar_width - filled) + "]"
            print(f"Game {game_index} {bar} step {steps}", end="\r")

    # игра закончена — печатаем итоги
    print()  # перенос строки

    winners = getattr(game, "winner_ids", [])
    if pid_rl in winners:
        result = "WIN"
        total_reward += WIN_REWARD
    else:
        result = "LOSE"
        total_reward += LOSE_PENALTY

    print(f"Game {game_index} → {result} | reward={total_reward:.1f} | steps={steps}")

    rl_agent.save_weights() 

    return total_reward, winners, steps

def run_training_session(rl_agent, total_games=200):
    win_count = 0
    rewards = []
    steps_list = []

    print(f"=== START TRAINING: {total_games} games ===")

    for i in range(1, total_games + 1):
        total_reward, winners, steps = play_rl_vs_random_progress(rl_agent, i)

        rewards.append(total_reward)
        steps_list.append(steps)

        if 0 in winners:
            win_count += 1

    # --- итоговая статистика ---
    winrate = win_count / total_games * 100
    avg_reward = sum(rewards) / total_games
    avg_steps = sum(steps_list) / total_games

    print("\n=== FINAL STATISTICS ===")
    print(f"Games played:       {total_games}")
    print(f"Wins:               {win_count}")
    print(f"Winrate:            {winrate:.2f}%")
    print(f"Average reward:     {avg_reward:.2f}")
    print(f"Reward min/max:     {min(rewards):.2f} / {max(rewards):.2f}")
    print(f"Average game steps: {avg_steps:.1f}")
    print(f"Longest game:       {max(steps_list)} steps")
    print(f"Shortest game:      {min(steps_list)} steps")

    print("========================\n")

    return {
        "winrate": winrate,
        "wins": win_count,
        "rewards": rewards,
        "steps": steps_list
    }



if __name__ == "__main__":
    temp_game = DurakGame(["RL", "RANDOM"])
    state0 = state_to_tensor(temp_game.get_state(0))
    state_size = len(state0)

    rl_agent = RLAgent(pid=0, state_size=state_size, action_size=50, epsilon=0.1,
                       weights_path=WEIGHTS_PATH)

    print(f"Использовались старые веса: {getattr(rl_agent, 'loaded_weights', False)}")

    total_games = 100
    rl_wins = 0

    for episode in range(total_games):
        total_reward, winners, steps = play_rl_vs_random_progress(rl_agent)
        if 0 in winners:  # pid_rl == 0
            rl_wins += 1

        print(f"=== EPISODE {episode + 1} ===")
        print("Победители:", winners)
        print("Общий reward RL:", float(total_reward))
    
    winrate = rl_wins / total_games * 100
    print(f"\n=== РЕЗУЛЬТАТЫ ПОСЛЕ {total_games} ИГР ===")
    print(f"RL победил в {rl_wins}/{total_games} игр ({winrate:.2f}% winrate)")
