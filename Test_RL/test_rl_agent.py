# Test_RL\test_rl_agent.py
import sys
import os

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # T25--4m_MLOps
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
    
import torch
from Core.core import DurakGame
from Core.agents import RLAgent
from src.preprocess import state_to_tensor

# Простая проверка: 2 игрока, RLAgent vs random moves
player_names = ["RL", "BOT"]
game = DurakGame(player_names)

# Создаём RLAgent
pid = 0
state_dict = game.get_state(pid)
state_vector = state_to_tensor(state_dict)
state_size = len(state_vector)
action_size = 50  # как в RLAgent по умолчанию

agent = RLAgent(pid=pid, state_size=state_size, action_size=action_size, epsilon=0.0)

# Игровой цикл (несколько шагов)
print("=== Starting test ===")
for step in range(10):
    if game.finished:
        print("Game finished early")
        break

    current_pid = game.turn_order[0]  # текущий игрок
    state = game.get_state(current_pid)

    if current_pid == pid:
        # RLAgent
        action = agent.act(game, current_pid)
        action = agent.filter_illegal_actions(action, game.players[current_pid].hand)
        print(f"[RLAgent] step {step}, pid {current_pid}, action: {action}")
    else:
        # random opponent
        legal = game.legal_actions(current_pid)
        action = legal[0] if legal else None
        print(f"[BOT] step {step}, pid {current_pid}, action: {action}")

    # Применяем действие
    game.step(current_pid, action)

    # Проверка: преобразуем текущее состояние в тензор
    tensor = state_to_tensor(game.get_state(pid))
    print(f"Tensor for RLAgent: {torch.tensor(tensor).float()}")

print("=== Test finished ===")
