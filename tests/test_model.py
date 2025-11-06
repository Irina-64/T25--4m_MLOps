from src.api import model, MODEL_FEATURES


def test_model_features_match_api_features():
    assert len(MODEL_FEATURES) == model.n_features_in_


