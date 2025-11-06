import pandas as pd
from src.preprocess import preprocess_data


def test_preprocess_returns_features_with_target_and_user_id():
    data = {
        "user_id": [1, 1, 2, 2],
        "date": [
            "2024-11-28",
            "2024-11-30",
            "2024-09-10",
            "2024-11-15",
        ],
        "amount": [100, -50, 200, -10],
        "churn": [0, 0, 1, 1],
    }
    df = pd.DataFrame(data)

    features = preprocess_data(df)

    # basic expectations
    assert "user_id" in features.columns
    assert "churn" in features.columns
    # should produce many engineered features
    assert features.shape[1] > 10
    # no NaNs after fillna
    assert features.isna().sum().sum() == 0


