# env\durak_env.py
import gymnasium as gym
import numpy as np
from typing import Optional, List
import sys
import os

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # T25--4m_MLOps
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from Core.core import DurakGame, Card
from src.preprocess import state_to_tensor
from Core.agents import RLAgent


class DurakEnv(gym.Env):
    """
    Двухигровочная RL-среда:
    - человек = player 0
    - RL агент = player 1
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, agent: RLAgent, render_mode: Optional[str] = None):
        super().__init__()
        self.agent = agent
        self.render_mode = render_mode
        self.game: Optional[DurakGame] = None

        temp_game = DurakGame(["Human", "Agent"])
        temp_state = state_to_tensor(temp_game.get_state(1))
        state_len = len(temp_state)

        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(state_len,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(200)

    def reset(self, seed=None, options=None):
        self.game = DurakGame(["Human", "Agent"])
        state = state_to_tensor(self.game.get_state(0))
        info = self._get_info()
        info["trump_card"] = str(self.game.trump_card)
        return state, info

    def step(self, human_action: str):
        if self.game.finished:
            state = state_to_tensor(self.game.get_state(0))
            info = self._get_info()
            return state, 0.0, True, False, info

        try:
            actions = self._encode_action(human_action)
        except ValueError as e:
            state = state_to_tensor(self.game.get_state(0))
            info = self._get_info()
            info["message"] = str(e)
            return state, 0.0, False, False, info

        # --- Ход человека ---
        if isinstance(actions, list):
            for a in actions:
                self.game.step(0, a)
        else:
            self.game.step(0, actions)

        # --- RL агент ходит ---
        if not self.game.finished:
            pid = 1
            legal = self.game.legal_actions(pid)
            if legal:
                rl_act = self.agent.act(self.game.get_state(pid), legal)
                rl_act = self.agent.filter_illegal_actions(rl_act, self.game.players[pid].hand)
                self.game.step(pid, rl_act)

        state = state_to_tensor(self.game.get_state(0))
        reward = 0.0
        done = self.game.finished
        info = self._get_info()
        return state, reward, done, False, info

    def _get_info(self):
        return {
            "your_hand": [str(c) for c in self.game.players[0].hand],
            "table": [[str(a), str(d) if d else None] for a, d in self.game.table],
            "legal_actions": [self._format_action(a) for a in self.game.legal_actions(0)],
            "attacker": self.game.turn_order[0],
            "defender": self.game.turn_order[1] if len(self.game.turn_order) > 1 else None,
            "finished": self.game.finished,
            "winners": getattr(self.game, "winner_ids", [])
        }

    def _format_action(self, action):
        if action[0] in ("attack", "add"):
            return f"{action[0]} {action[1]}"
        if action[0] == "defend":
            return f"{action[0]} {action[1]} -> {action[2]}"
        return str(action)

    def _encode_action(self, human_action: str):
        human_action = human_action.strip().replace("'", "").replace('"', '')
        legal = self.game.legal_actions(0)

        # --- take / pass ---
        if human_action.lower() == "take":
            return ("take",)
        if human_action.lower() == "pass":
            return ("pass",)

        # Разбор карт
        card_strs = [s.strip() for s in human_action.split(",")]

        # Преобразуем строки в реальные карты
        input_cards = []
        for card_str in card_strs:
            found = None
            for c in self.game.players[0].hand:
                if str(c) == card_str:
                    found = c
                    break
            if not found:
                raise ValueError(f"Вы не имеете карты {card_str} на руке.")
            input_cards.append(found)

        # --- Проверяем, есть ли незакрытые атаки ---
        open_attacks = [(i, a) for i, (a, d) in enumerate(self.game.table) if d is None]

        # === Защита ===
        if open_attacks:
            if len(input_cards) != len(open_attacks):
                raise ValueError(
                    f"Нужно ввести ровно {len(open_attacks)} карт(ы) "
                    f"для защиты: {', '.join(str(a) for _, a in open_attacks)}"
                )

            actions = []
            for (idx, atk_card), defend_card in zip(open_attacks, input_cards):
                act = ('defend', atk_card, defend_card)
                if act not in legal:
                    raise ValueError(f"Карта {defend_card} не может побить {atk_card}.")
                actions.append(act)

            return actions if len(actions) > 1 else actions[0]

        # === Атака ===
        actions = []
        for c in input_cards:
            matched = False
            for act in legal:
                if act[0] in ("attack", "add") and act[1] == c:
                    actions.append(act)
                    matched = True
                    break
            if not matched:
                raise ValueError(f"Карта {c} не может быть сыграна сейчас.")

        return actions if len(actions) > 1 else actions[0]
