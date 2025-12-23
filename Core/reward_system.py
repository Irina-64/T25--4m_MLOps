# Core/reward_system.py
from collections import defaultdict

# ранговые коэффициенты для карт: чем слабее карта, тем выше бонус
CARD_RANK_REWARD = {
    "6": 6,
    "7": 5,
    "8": 4,
    "9": 3,
    "10": 2,
    "J": 1,
    "Q": 0.5,
    "K": 0.2,
    "A": 0.1,
}

# коэффициенты комбо
COMBO_MULTIPLIER = 1.2
PAIR_BONUS = 1.5
TRIPLE_BONUS = 2.0
QUAD_BONUS = 2.5
POGON_BONUS = 5.0

ILLEGAL_MOVE_PENALTY = -3

class RewardSystem:
    def __init__(self):
        self.last_attacks_success = defaultdict(int)

    def compute_reward(self, game, pid, action, state_before, state_after):
        return self._reward(game, pid, action, state_before, state_after)

    def _reward(self, game, pid, action, state_before, state_after):
        reward = 0.0

        # --- Проверка легальности действия ---
        if not self.is_action_legal(action, state_before):
            return ILLEGAL_MOVE_PENALTY

        hand_before = state_before["hand_sizes"][pid]
        hand_after = state_after["hand_sizes"][pid]

        # --- базовый reward за любое легальное действие ---
        reward += 2.0

        # --- уменьшение руки ---
        cards_played = hand_before - hand_after
        if cards_played > 0:
            if action[0] in ["attack", "add"]:
                card_values = action[1] if isinstance(action[1], list) else [action[1]]
                for c in card_values:
                    c_str = str(c) if hasattr(c, '__repr__') else c
                    rank = ''.join(filter(str.isalnum, c_str))  # извлекаем ранг
                    reward += CARD_RANK_REWARD.get(rank, 0)
            elif action[0] == "defend":
                reward += 1.0

        # --- успешная атака (противник взял) ---
        if action[0] == "attack":
            defender = state_before["defender"]
            delta = state_after["hand_sizes"][defender] - state_before["hand_sizes"][defender]
            if delta > 0:
                self.last_attacks_success[pid] += 1
                reward += 2.0
                reward *= COMBO_MULTIPLIER ** (self.last_attacks_success[pid] - 1)
            else:
                self.last_attacks_success[pid] = 0

        # --- бонус за пары/тройки/четверки ---
        if action[0] == "attack" and isinstance(action[1], list):
            count = len(action[1])
            if count == 2:
                reward += PAIR_BONUS
            elif count == 3:
                reward += TRIPLE_BONUS
            elif count == 4:
                reward += QUAD_BONUS

        # --- победа / поражение ---
        if game.finished:
            if pid in game.winner_ids:
                reward += 100
                hand_hist = state_after.get("hand_history", {}).get(pid, [])
                if self.is_pogon(hand_hist):
                    reward += POGON_BONUS
            else:
                reward -= 100

        return reward

    def is_action_legal(self, action, state):
        hand = state.get("your_hand", [])
        # Преобразуем все элементы hand к строке
        hand = [str(c) for c in hand]

        if action[0] in ["attack", "add"]:
            cards = action[1] if isinstance(action[1], list) else [action[1]]
            cards = [str(c) for c in cards]
            return all(c in hand for c in cards)
        elif action[0] == "defend":
            card = str(action[2])
            return card in hand
        return True

    def is_pogon(self, hand_history):
        if len(hand_history) < 2:
            return False
        return all(str(c).startswith("6") for c in hand_history[-2:])
