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
COMBO_MULTIPLIER = 1.2  # за повторные успешные атаки
PAIR_BONUS = 1.5        # пара одинаковых карт
TRIPLE_BONUS = 2.0       # тройка одинаковых карт
QUAD_BONUS = 2.5         # четверка одинаковых карт
POGON_BONUS = 5.0        # “на погоны” – пара шестерок в конце

# штраф за попытку сделать недопустимый ход
ILLEGAL_MOVE_PENALTY = -3

class RewardSystem:
    def __init__(self):
        # хранение статистики для комбо внутри игры
        self.last_attacks_success = defaultdict(int)
    def compute_reward(self, game, pid, action, state_before, state_after):
        """Публичный метод для использования агентом и тестами"""
        return self._reward(game, pid, action, state_before, state_after)

    def _reward(self, game, pid, action, state_before, state_after):
        """
        Внутренний метод вычисления награды.
        Публичный метод compute_reward просто его вызывает.
        """
        reward = 0

        # проверяем на попытку недопустимого хода
        if not self.is_action_legal(action, state_before):
            return ILLEGAL_MOVE_PENALTY

        hand_size_before = state_before["hand_sizes"][pid]
        hand_size_after = state_after["hand_sizes"][pid]

        # выиграл или проиграл
        if game.finished:
            if pid in game.winner_ids:
                reward += 100
                if self.is_pogon(state_after["hand_history"].get(pid, [])):
                    reward += POGON_BONUS
            else:
                reward -= 100

        # уменьшение руки (выкидывание карт)
        cards_played = hand_size_before - hand_size_after
        if cards_played > 0:
            if action[0] in ["attack", "add"]:
                card_values = action[1] if isinstance(action[1], list) else [action[1]]
                for c in card_values:
                    reward += CARD_RANK_REWARD.get(str(c.rank), 0)
            elif action[0] == "defend":
                reward += 1  # базовый бонус за успешную защиту

        # бонус за атаки подряд
        if action[0] == "attack":
            defender = state_before["defender"]
            if state_after["hand_sizes"][defender] > state_before["hand_sizes"][defender]:
                self.last_attacks_success[pid] += 1
                reward *= COMBO_MULTIPLIER ** (self.last_attacks_success[pid] - 1)
            else:
                self.last_attacks_success[pid] = 0

        # бонус за пары/тройки/четверки
        if action[0] == "attack" and isinstance(action[1], list) and len(action[1]) > 1:
            count = len(action[1])
            if count == 2:
                reward *= PAIR_BONUS
            elif count == 3:
                reward *= TRIPLE_BONUS
            elif count == 4:
                reward *= QUAD_BONUS

        # бонус за создание трудностей оппоненту
        if action[0] == "attack":
            defender = state_before["defender"]
            delta = state_after["hand_sizes"][defender] - state_before["hand_sizes"][defender]
            if delta > 0:
                for c in action[1] if isinstance(action[1], list) else [action[1]]:
                    rank_val = getattr(c, "rank_val", 0)
                    reward += rank_val * 0.1

        return reward

    def is_action_legal(self, action, state):
        """Простейшая проверка легальности хода. Можно расширять."""
        hand = state.get("your_hand", [])

        if action[0] in ["attack", "add"]:
            card = action[1] if isinstance(action[1], list) else action[1]
            for c in card if isinstance(card, list) else [card]:
                if c not in hand:
                    return False
        elif action[0] == "defend":
            card = action[2]
            if card not in hand:
                return False
        return True

    def is_pogon(self, hand_history):
        """Проверяем, были ли последние две выкинутые карты – пара шестерок"""
        if len(hand_history) < 2:
            return False
        return all(c.rank == "6" for c in hand_history[-2:])
