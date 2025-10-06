# demo.py
import os
import random
from datetime import datetime

import torch
from durka import DurakGame, ReplayLogger
from durka.agents import RLAgent, heuristic_agent, random_agent

MODEL_PATH = "rl_agent_model.pth"


def format_hand(hand, trump_suit):
    """Возвращает список карт игрока, помечая козыри."""
    return [f"{card}{' (trump)' if card.suit == trump_suit else ''}" for card in hand]


def play_demo_game(agents_list, names=None, epsilon=0.1):
    if names is None:
        names = [f"Bot{i}" for i in range(len(agents_list))]

    g = DurakGame(names)
    logger = ReplayLogger(g)

    agents = {}
    for i, agent in enumerate(agents_list):
        if isinstance(agent, type):
            state_dict = g.get_state(i)
            state_size = len(RLAgent.state_to_tensor(None, state_dict))
            action_size = 50
            agents[i] = agent(
                pid=i, state_size=state_size, action_size=action_size, epsilon=epsilon
            )
            if os.path.exists(MODEL_PATH):
                agents[i].model.load_state_dict(torch.load(MODEL_PATH))
                print(f"RLAgent {i} модель загружена из {MODEL_PATH}")
        else:
            agents[i] = agent

    trump_suit = g.trump_suit
    print(f"Trump suit: {trump_suit}")

    # Логируем начальные руки
    initial_hands = {
        player.name: format_hand(player.hand, trump_suit) for player in g.players
    }
    logger.replay["initial_hands"] = initial_hands

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
                reward = agent.compute_reward(g, pid, action, state_before, state_after)
                agent.learn(
                    agent.state_to_tensor(state_before),
                    0,
                    reward,
                    agent.state_to_tensor(state_after),
                    g.finished,
                )

            # Логи после хода с визуализацией рук
            hands_after = {
                player.name: format_hand(player.hand, trump_suit)
                for player in g.players
            }
            logger.log_step(pid, action, state_before, state_after, hands_after)

            # Консольная визуализация
            print(f"{g.players[pid].name} делает ход: {action}")
            for player in g.players:
                print(f"{player.name} hand: {format_hand(player.hand, trump_suit)}")
            print("-" * 40)

            if g.finished:
                break

    print("Игра завершена!")
    winners = [g.players[i].name for i in g.winner_ids]
    losers = [
        player.name for i, player in enumerate(g.players) if i not in g.winner_ids
    ]
    print(f"Победители: {winners}")
    print(f"Проигравшие: {losers}")

    # Сохраняем модель
    for agent in agents.values():
        if isinstance(agent, RLAgent):
            torch.save(agent.model.state_dict(), MODEL_PATH)
            print(f"RLAgent модель сохранена в {MODEL_PATH}")

    # Сохраняем реплей с таймкодом
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("replays", exist_ok=True)
    replay_path = f"replays/replay_{timestamp}.json"
    logger.finalize(replay_path)
    print(f"Replay сохранён в {replay_path}")


if __name__ == "__main__":
    play_demo_game(
        [random_agent, heuristic_agent, RLAgent], names=["Bot1", "Bot2", "RL"]
    )
