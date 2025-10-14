# durka/agents.py
import random
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# helper: lower value = слабее карта (ее удобнее сыграть)
def _card_value(card, trump_suit):
    """
    Возвращает кортеж (is_trump, rank_index).Ф
    is_trump = False (0) для нетузов — они предпочтительнее (меньшая 'ценность'),
    затем по рангу: 0..8 (6..A).
    min() по этому ключу выберет "наименее ценную" карту, не жертвуя козырями.
    """
    is_trump = card.suit == trump_suit
    return (is_trump, card.rank_index())


def random_agent(game, pid):
    """Простейший агент: случайный выбор из доступных действий."""
    legal = game.legal_actions(pid)
    if not legal:
        return None
    return random.choice(legal)


def rule_based_agent(game, pid):
    """Чуть умнее: старается отбиваться и ходить маленькими картами."""
    legal = game.legal_actions(pid)
    if not legal:
        return None

    # Если есть возможность защищаться — отбиваемся минимальной картой
    defends = [a for a in legal if a[0] == "defend"]
    if defends:
        return min(defends, key=lambda x: game.deck.card_value(x[2], game.trump_suit))

    # Если атакуем — ходим минимальной некозырной картой
    attacks = [a for a in legal if a[0] == "attack"]
    if attacks:
        non_trumps = [a for a in attacks if a[1].suit != game.trump_suit]
        if non_trumps:
            return min(
                non_trumps, key=lambda x: game.deck.card_value(x[1], game.trump_suit)
            )
        return min(attacks, key=lambda x: game.deck.card_value(x[1], game.trump_suit))

    # Если подбрасываем — тоже минимальной картой
    adds = [a for a in legal if a[0] == "add"]
    if adds:
        return min(adds, key=lambda x: game.deck.card_value(x[1], game.trump_suit))

    # Если нечего делать — берем/пасуем
    return random.choice(legal)


def heuristic_agent(game, pid: int):
    """
    Чуть умнее:
    - при атаке/подбрасывании избегает козырей (если есть альтернатива)
    - при защите пытается бить некозырной минимальной, если возможно
    """
    legal = game.legal_actions(pid)
    if not legal:
        return None

    # атака
    attacks = [a for a in legal if isinstance(a, tuple) and a[0] == "attack"]
    if attacks:
        non_trumps = [a for a in attacks if a[1].suit != game.trump_suit]
        pool = non_trumps if non_trumps else attacks
        return min(pool, key=lambda x: _card_value(x[1], game.trump_suit))

    # подбрасывание
    adds = [a for a in legal if isinstance(a, tuple) and a[0] == "add"]
    if adds:
        non_trumps = [a for a in adds if a[1].suit != game.trump_suit]
        pool = non_trumps if non_trumps else adds
        return min(pool, key=lambda x: _card_value(x[1], game.trump_suit))

    # защита
    defends = [a for a in legal if isinstance(a, tuple) and a[0] == "defend"]
    if defends:
        non_trumps = [a for a in defends if a[2].suit != game.trump_suit]
        pool = non_trumps if non_trumps else defends
        return min(pool, key=lambda x: _card_value(x[2], game.trump_suit))

    # fallback
    return random.choice(legal)


class RLAgent:
    def __init__(self, pid: int, state_size: int, action_size: int, epsilon=0.1):
        self.pid = pid
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.model = nn.Sequential(
            nn.Linear(state_size, 128), nn.ReLU(), nn.Linear(128, action_size)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def state_to_tensor(self, state):
        vec = []
        for pid in sorted(state["hand_sizes"].keys()):
            vec.append(state["hand_sizes"][pid])
        vec.append(state["deck_count"])
        vec.append(state["discard_count"])
        vec.append(len(state["table"]))
        vec.append(state["attacker"])
        vec.append(state["defender"])
        return vec

    def compute_reward(self, game, pid, action, state_before, state_after):
        reward = 0
        if state_after["hand_sizes"][pid] > state_before["hand_sizes"][pid]:
            reward -= state_after["hand_sizes"][pid] - state_before["hand_sizes"][pid]
        if (
            action[0] == "defend"
            and state_after["hand_sizes"][pid] == state_before["hand_sizes"][pid]
        ):
            reward += 1
        if action[0] == "attack":
            defender = state_before["defender"]
            if (
                state_after["hand_sizes"][defender]
                > state_before["hand_sizes"][defender]
            ):
                reward += (
                    state_after["hand_sizes"][defender]
                    - state_before["hand_sizes"][defender]
                )
        if game.finished:
            if pid in game.winner_ids:
                reward += 5
            else:
                reward -= 5
        return reward

    def act(self, game, pid):
        legal_actions = game.legal_actions(pid)
        if not legal_actions:
            return None

        state_dict = game.get_state(pid)
        state_vector = self.state_to_tensor(state_dict)
        state_tensor = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0)

        if random.random() < self.epsilon:
            return random.choice(legal_actions)

        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze(0).numpy()

        # выбираем индекс наилучшего легального действия
        q_legal = [q_values[i % self.action_size] for i in range(len(legal_actions))]
        max_idx = int(np.argmax(q_legal))
        return legal_actions[max_idx]

    def learn(self, state, action_idx, reward, next_state, done, gamma=0.99):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            target = reward
            if not done:
                target += gamma * torch.max(self.model(next_tensor)).item()

        pred = self.model(state_tensor)[0, action_idx % self.action_size]
        target_tensor = torch.tensor(target, dtype=torch.float32)
        loss = self.loss_fn(pred, target_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def filter_illegal_actions(self, action, hand):
        # action = ('attack', card) или ('defend', n, card) или ('add', card)
        if action[0] in ["attack", "add"]:
            card = action[1]
            if card not in hand:
                return ("pass",)
        elif action[0] == "defend":
            card = action[2]
            if card not in hand:
                return ("pass",)
        return action
