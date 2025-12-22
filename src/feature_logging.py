import csv
import os
from datetime import datetime

FEATURES_PATH = "feature_repo/data/rl_state_features.csv"

HEADER = [
    "episode_id",
    "step_id",
    "event_timestamp",
    "deck_frac",
    "discard_frac",
    "is_attacker",
    "is_defender",
    "hand_size_frac",
    "opponents_avg_hand",
    "trump_suit_id",
]

def log_state_features(state, pid, episode_id, step_id):
    os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
    file_exists = os.path.exists(FEATURES_PATH)

    hand_sizes = state["hand_sizes"]
    opponents = [v for k, v in hand_sizes.items() if k != pid]
    opponents_avg = sum(opponents) / len(opponents) if opponents else 0

    row = {
        "episode_id": episode_id,
        "step_id": step_id,
        "event_timestamp": datetime.utcnow().isoformat(),
        "deck_frac": state["deck_count"] / 36.0,
        "discard_frac": state["discard_count"] / 36.0,
        "is_attacker": 1 if state["attacker"] == pid else 0,
        "is_defender": 1 if state["defender"] == pid else 0,
        "hand_size_frac": hand_sizes.get(pid, 0) / 36.0,
        "opponents_avg_hand": opponents_avg / 36.0,
        "trump_suit_id": ["♣", "♦", "♥", "♠"].index(state["trump_suit"]),
    }

    with open(FEATURES_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
