import json
import os
from datetime import datetime


class ReplayLogger:
    def __init__(self, game, save_dir="replays"):
        self.game = game
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.replay = {
            "players": [p.name for p in game.players],
            "trump": game.trump_suit,
            "initial_hands": {},  # заполнится в demo
            "steps": [],
            "winner": None,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    def log_step(self, player_id, action, state_before, state_after, hands_after):
        """
        hands_after: dict {player_name: [card1, card2, ...]} с пометкой козырей
        """
        self.replay["steps"].append(
            {
                "player": player_id,
                "action": self._serialize_action(action),
                "state_before": state_before,
                "state_after": state_after,
                "hands_after": hands_after,
            }
        )

    def finalize(self, path=None):
        self.replay["winner"] = self.game.winner_ids
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.save_dir, f"replay_{timestamp}.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.replay, f, ensure_ascii=False, indent=2)

        return path

    def _serialize_action(self, action):
        if isinstance(action, tuple):
            return tuple(str(x) for x in action)
        return str(action)

    # Чтобы совместимо с demo.py: просто вызываем finalize
    def save(self, path=None):
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.replay, f, ensure_ascii=False, indent=2)
            return path
        return self.finalize()
