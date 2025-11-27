# src\preprocess.py
import numpy as np
import torch

# ---- Карточные значения ----
CARD_VECTOR_SIZE = 13  # 9 рангов + 4 масти
RANKS = ["6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUITS = ["♣", "♦", "♥", "♠"]

# -----------------------------
#  ДЕКОДИРОВАНИЕ СТРОКОВОЙ КАРТЫ
# -----------------------------
def decode_card(card_str):
    if card_str is None:
        return None
    if card_str.startswith("10"):
        rank = "10"
        suit = card_str[2:]
    else:
        rank = card_str[0]
        suit = card_str[1:]
    return (rank, suit)

# -----------------------------
#  ЭНКОДИНГ ОДНОЙ КАРТЫ
# -----------------------------
def encode_card(card):
    """
    Поддерживает:
    - None
    - строку вида "10♣"
    - tuple ("10","♣")
    - объект Card(rank="10", suit="♣")
    """
    vec = np.zeros(CARD_VECTOR_SIZE, dtype=np.float32)
    if card is None:
        return torch.tensor(vec, dtype=torch.float32)

    # ---- ЕСЛИ card — объект Card ----
    if hasattr(card, "rank") and hasattr(card, "suit"):
        rank = str(card.rank)
        suit = str(card.suit)
    # ---- tuple ----
    elif isinstance(card, tuple) and len(card) == 2:
        rank, suit = card
    # ---- строка ----
    elif isinstance(card, str):
        rank, suit = decode_card(card)
    else:
        return torch.tensor(vec, dtype=torch.float32)

    if rank in RANKS:
        vec[RANKS.index(rank)] = 1.0
    if suit in SUITS:
        vec[9 + SUITS.index(suit)] = 1.0

    return torch.tensor(vec, dtype=torch.float32)


# -----------------------------
#  СТОЛ
# -----------------------------
def encode_table(table):
    encoded = []
    for pair in table:
        attack, defend = pair if isinstance(pair, tuple) else (pair, None)
        encoded.append(encode_card(attack))
        encoded.append(encode_card(defend) if defend else torch.zeros(CARD_VECTOR_SIZE))
    # padding до 12 карт
    while len(encoded) < 12:
        encoded.append(torch.zeros(CARD_VECTOR_SIZE))
    return torch.cat(encoded).float()

# -----------------------------
#  РУКА ИГРОКА
# -----------------------------
def encode_hand(hand_list):
    encoded = [encode_card(decode_card(c)) for c in hand_list]
    while len(encoded) < 36:
        encoded.append(torch.zeros(CARD_VECTOR_SIZE))
    return torch.cat(encoded).float()

# -----------------------------
#  STATE → ТЕНЗОР
# -----------------------------
def state_to_tensor(state):
    # scalar features
    vec = [
        SUITS.index(state["trump_suit"]) / 3.0,
        state["deck_count"] / 36.0,
        state["discard_count"] / 36.0,
        state["attacker"] / 5.0,
        state["defender"] / 5.0,
        1.0 if state.get("finished", False) else 0.0,
    ]
    for pid in range(6):
        vec.append(state["hand_sizes"].get(pid, 0) / 36.0)

    vec_tensor = torch.tensor(vec, dtype=torch.float32)
    table_tensor = encode_table(state.get("table", []))
    hand_tensor = encode_hand(state.get("your_hand", []))

    return torch.cat([vec_tensor, table_tensor, hand_tensor]).float()
