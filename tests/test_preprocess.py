import torch
from src.preprocess import encode_card, state_to_tensor


def test_encode_card_returns_tensor():
    card = "10♣"
    tensor = encode_card(card)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape[0] == 13
    assert tensor.dtype == torch.float32


def test_state_to_tensor_output():
    dummy_state = {
        "trump_suit": "♣",
        "deck_count": 36,
        "discard_count": 0,
        "attacker": 0,
        "defender": 1,
        "finished": False,
        "hand_sizes": {0: 6, 1: 6},
        "table": [],
        "your_hand": []
    }

    tensor = state_to_tensor(dummy_state)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.ndim == 1
    assert tensor.dtype == torch.float32
    assert tensor.numel() > 0
