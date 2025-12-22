# Test_RL/test_rl_vs_heuristic.py
import sys
import os
import torch
import pandas as pd

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from Core.core import DurakGame
from Core.agents import RLAgent, heuristic_agent
from src.preprocess import state_to_tensor

WEIGHTS_PATH = "rl_weights.pth"

# ----------------------------
# Simple Feature Store
# ----------------------------
class SimpleFeatureStore:
    def __init__(self, csv_path="rl_state_features.csv"):
        self.csv_path = csv_path
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
            if "state_id" not in self.df.columns:
                self.df["state_id"] = range(len(self.df))
            self.df.set_index("state_id", inplace=True)
        else:
            self.df = pd.DataFrame()

    def log_features(self, state_id, features_list):
        """
        Логируем фичи как список в одну колонку 'features'
        """
        row = pd.DataFrame([{"features": features_list}], index=[state_id])
        self.df = pd.concat([self.df, row])
        self.df.to_csv(self.csv_path)

    def get_features(self, state_id):
        if state_id in self.df.index:
            return self.df.loc[state_id]["features"]
        else:
            return None

# создаём feature store
fs = SimpleFeatureStore("rl_state_features.csv")

# ----------------------------
# Игровой цикл
# ----------------------------
def play_rl_vs_heuristic(rl_agent, episode_id):
    player_names = ["RL", "HEUR"]
    game = DurakGame(player_names)
    pid_rl = 0
    total_reward = 0

    step_id = 0

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

            # --- LOG FEATURES TO FEATURE STORE ---
            features_list = state_to_tensor(state_before).numpy().tolist()
            fs.log_features(state_id=f"{episode_id}_{step_id}", features_list=features_list)
            step_id += 1
            # -------------------------------------

            state_after = game.get_state(pid)
            reward = rl_agent.reward_system.compute_reward(
                game, pid, action, state_before, state_after
            )
            total_reward += reward

            rl_agent.learn(state_before, action, state_after, game, done=game.finished)
        else:
            action = heuristic_agent(game, pid)

        game.step(pid, action)

    rl_agent.save_weights()
    return total_reward, game.winner_ids

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    temp_game = DurakGame(["RL", "HEUR"])
    state0 = state_to_tensor(temp_game.get_state(0))
    state_size = len(state0)

    loaded_existing_weights = False
    if os.path.exists(WEIGHTS_PATH):
        loaded_state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
        first_layer_weight = loaded_state_dict[list(loaded_state_dict.keys())[0]]
        if first_layer_weight.shape[1] == state_size:
            loaded_existing_weights = True

    rl_agent = RLAgent(
        pid=0,
        state_size=state_size,
        action_size=50,
        epsilon=0.1,
        weights_path=WEIGHTS_PATH
    )

    print(f"Использовались старые веса: {loaded_existing_weights}")

    # запускаем несколько эпизодов
    for episode in range(100):
        total_reward, winners = play_rl_vs_heuristic(rl_agent, episode_id=episode)
        print(f"=== EPISODE {episode + 1} ===")
        print("Победители:", winners)
        print("Общий reward RL:", total_reward)
