# Core\demo.py
import sys
import os
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from env.durak_env import DurakEnv
from Core.core import DurakGame
from Core.agents import RLAgent
from src.preprocess import state_to_tensor

# === создаём временную игру для вычисления state_size ===
temp_game = DurakGame(["PLAYER", "DUMMY"])
state0 = state_to_tensor(temp_game.get_state(0))
state_size = len(state0)

# === создаем агента ===
agent = RLAgent(pid=1, state_size=state_size)

# === создаем env с нашим агентом ===
env = DurakEnv(agent)

obs, info = env.reset()
print("Козырь:", info["trump_card"])

while True:
    print("\n===== Ваш ход =====")
    print("Карты на руке:", ", ".join(info["your_hand"]))
    print("Карты на столе:")
    for idx, (atk, dfd) in enumerate(info["table"]):
        print(f" {idx}: {atk} -> {dfd if dfd else '---'}")

    print("Легальные действия:", info["legal_actions"])
    print("Можно выбрать несколько карт через запятую, например: 'Q♣, 7♦'")
    if any(d is None for _, d in info["table"]):
        print("Для защиты просто введите карты по порядку атакующих карт на столе.")

    human_action = input("Введите действие (например, '6♣', 'take', 'pass'): ").strip()

    obs, reward, done, truncated, info = env.step(human_action)

    if "message" in info:
        print("⚠", info["message"])

    if done:
        print("\nИгра закончена!")
        print("Победители:", info["winners"])
        break
