# Core/agents.py
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.preprocess import state_to_tensor, decode_card, RANKS, SUITS
from Core.reward_system import RewardSystem
from collections import deque
import os

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
    def __init__(self, pid, state_size=None, action_size=50, epsilon=0.1,
                 buffer_size=5000, batch_size=32, gamma=0.99, lr=0.001,
                 weights_path="rl_weights.pth"):
        self.pid = pid
        self.action_size = action_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.weights_path = weights_path

        # устройство
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # replay buffer
        self.memory = deque(maxlen=buffer_size)

        # reward system
        self.reward_system = RewardSystem()

        if state_size is None:
            raise ValueError("state_size must be provided or calculated before creating RLAgent")

        # модель
        self.model = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # загружаем веса, если они есть и совпадает размер
        if os.path.exists(self.weights_path):
            loaded_state_dict = torch.load(self.weights_path, map_location=self.device)
            first_layer_weight = loaded_state_dict[list(loaded_state_dict.keys())[0]]
            if first_layer_weight.shape[1] == state_size:
                self.model.load_state_dict(loaded_state_dict)
                print(f"[INFO] Loaded RL weights from {self.weights_path}")
            else:
                print("[INFO] Weight size mismatch, starting fresh.")
        else:
            print("[INFO] No pre-existing weights found, starting fresh.")

    def state_to_tensor(self, state):
        return state_to_tensor(state).float()

    def act(self, state, legal_actions):
        # explore
        if random.random() < self.epsilon:
            return random.choice(legal_actions)

        # exploit
        with torch.no_grad():
            s = self.state_to_tensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(s)[0].cpu().numpy()

        legal_q = []
        for action in legal_actions:
            idx = self.action_to_index(action)
            legal_q.append(q_values[idx] if idx < len(q_values) else -1e9)

        best_idx = int(np.argmax(legal_q))
        return legal_actions[best_idx]

    def learn(self, state, action, next_state, game, done):
        self.memory.append((state, action, next_state, game, done))

        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, next_states, games, dones = zip(*batch)

        state_tensor = torch.stack([self.state_to_tensor(s) for s in states]).to(self.device)
        next_tensor = torch.stack([self.state_to_tensor(ns) for ns in next_states]).to(self.device)

        q_pred = self.model(state_tensor)
        q_target = q_pred.clone().detach()

        for i, (s, a, ns, g, d) in enumerate(batch):
            idx = self.action_to_index(a)
            reward = self.reward_system.compute_reward(g, self.pid, a, s, ns)
            target = reward
            if not d:
                ns_tensor = self.state_to_tensor(ns).unsqueeze(0).to(self.device)
                target += self.gamma * torch.max(self.model(ns_tensor)).item()
            q_target[i, idx % self.action_size] = target

        loss = self.loss_fn(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_weights(self):
        torch.save(self.model.state_dict(), self.weights_path)
        print(f"[INFO] RL weights saved to {self.weights_path}")

    def action_to_index(self, action):
        if action[0] == "pass":
            return 0
        if action[0] == "attack":
            card = action[1]
            card_id = self.card_to_id(card)
            return 1 + card_id
        if action[0] == "defend":
            attack_card, defend_card = action[1], action[2]
            a = self.card_to_id(attack_card)
            d = self.card_to_id(defend_card)
            return 1 + 52 + a * 52 + d
        return 0

    def filter_illegal_actions(self, action, hand):
        if action[0] in ["attack", "add"]:
            cards = action[1] if isinstance(action[1], list) else [action[1]]
            if not all(c in hand for c in cards):
                return ("pass",)
        elif action[0] == "defend":
            if action[2] not in hand:
                return ("pass",)
        return action

    def card_to_id(self, card):
        if hasattr(card, "rank") and hasattr(card, "suit"):
            rank = str(card.rank)
            suit = str(card.suit)
        elif isinstance(card, tuple):
            rank, suit = card
        elif isinstance(card, str):
            rank, suit = decode_card(card)
        else:
            return 0
        r = RANKS.index(rank)
        s = SUITS.index(suit)
        return s * len(RANKS) + r