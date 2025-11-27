# Test_RL/test_one_game.py
import sys
import os

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # T25--4m_MLOps
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from Core.core import DurakGame
from Core.agents import RLAgent
from src.preprocess import preprocess_state
from Core.agents import heuristic_agent   # <-- твой умный бот

print("\n=== STARTING MATCH: MODEL vs HEURISTIC BOT ===\n")

# ---------------------------
# СОЗДАНИЕ ИГРЫ И АГЕНТОВ
# ---------------------------
game = DurakGame(num_players=2)
agent0 = RLAgent(pid=0, state_dim=200, action_dim=50)   # твоя модель

game.reset()

# ---------------------------
# ОСНОВНОЙ ЦИКЛ ИГРЫ
# ---------------------------
while not game.finished:

    pid = game.current_player_id
    state_before = preprocess_state(game, pid)

    # ----------------------
    # ВЫБОР ДЕЙСТВИЯ
    # ----------------------
    if pid == 0:
        # модель делает ход
        action = agent0.select_action(state_before, game)
    else:
        # умный бот делает ход
        action = heuristic_agent(game, pid)

    print(f"Player {pid} action: {action}")

    # ----------------------
    # ПРИМЕНЯЕМ ХОД
    # ----------------------
    legal = game.apply_action(pid, action)
    if not legal:
        print(f"  -> Illegal action {action}")

    # ----------------------
    # СОСТОЯНИЕ ПОСЛЕ ХОДА
    # ----------------------
    state_after = preprocess_state(game, pid)

    # ----------------------
    # 🧠 ОБУЧАЕМ МОДЕЛЬ ТОЛЬКО ЕСЛИ ХОДИЛА ОНА
    # ----------------------
    if pid == 0:
        agent0.learn(
            state_before,
            action,
            state_after,
            game,
            done=game.finished
        )

    # ----------------------
    # ПЕЧАТЫ СТОЛ
    # ----------------------
    game.print_state()
    print("\n---\n")

print("\n=== GAME OVER ===")
print("Winner:", game.winner)
