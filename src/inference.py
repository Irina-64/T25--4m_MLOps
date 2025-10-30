import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import argparse
from typing import Tuple, Optional

SEQ_LEN = 335
BATCH_SIZE = 64
HIDDEN_SIZE = 64
THRESHOLD = 0.5


class LSTMChurnModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_size)
        _, (hn, _) = self.lstm(x)  # hn shape: (num_directions, batch, hidden_size)
        # concat last hidden states from both directions
        if hn.size(0) == 2:
            hn_cat = torch.cat((hn[0], hn[1]), dim=1)  # (batch, hidden_size*2)
        else:
            hn_cat = hn[-1]
        return torch.sigmoid(self.fc(hn_cat))  # (batch, 1)


def prepare_sequences(df: pd.DataFrame, scaler: StandardScaler) -> Tuple[np.ndarray, np.ndarray]:
    # Ensure date column
    if "date" not in df.columns:
        raise ValueError("Input dataframe must contain 'date' column")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = (df["date"] - pd.to_datetime("2024-01-01")).dt.days
    df = df[(df["day"] >= 0) & (df["day"] < SEQ_LEN)]

    agg = df.groupby(["user_id", "day"]).agg(
        amount_sum=("amount", "sum"),
        amount_count=("amount", "count")
    ).reset_index()

    all_days = pd.DataFrame({"day": range(SEQ_LEN)})
    user_ids = np.array(df["user_id"].unique())
    data = []

    for uid in user_ids:
        user_df = agg[agg["user_id"] == uid].merge(all_days, on="day", how="right").fillna(0)
        user_df = user_df.sort_values("day")
        data.append(user_df[["amount_sum", "amount_count"]].values)

    X = np.stack(data)  # shape: (n_users, seq_len, 2)

    # Normalize using saved scaler parameters (mean_, scale_)
    X_reshaped = X.reshape(-1, X.shape[-1])
    # scaler.mean_ and scaler.scale_ expected to be arrays of length n_features
    X_scaled = (X_reshaped - scaler.mean_) / scaler.scale_
    X_scaled = X_scaled.reshape(X.shape)

    return user_ids, X_scaled


def main(input_path: str, output_path: Optional[str] = None) -> Optional[pd.DataFrame]:
    # try read CSV with header; if it doesn't contain expected cols try without header
    try:
        df = pd.read_csv(input_path)
    except Exception:
        df = pd.read_csv(input_path, header=None, names=["user_id", "date", "amount"])

    # If file contains only three columns but first row is headerless numbers, ensure columns present
    expected_cols = {"user_id", "date", "amount"}
    if not expected_cols.issubset(set(df.columns)):
        # try fallback: assume no header and use three columns
        df = pd.read_csv(input_path, header=None, names=["user_id", "date", "amount"])

    # Load checkpoint and scaler params
    checkpoint = torch.load("model.pt", map_location=torch.device("cpu"))
    scaler = StandardScaler()
    # support both keys used previously
    if "scaler_mean" in checkpoint and "scaler_scale" in checkpoint:
        scaler.mean_ = checkpoint["scaler_mean"]
        scaler.scale_ = checkpoint["scaler_scale"]
    elif "scaler_mean_" in checkpoint and "scaler_scale_" in checkpoint:
        scaler.mean_ = checkpoint["scaler_mean_"]
        scaler.scale_ = checkpoint["scaler_scale_"]
    else:
        # Try keys used earlier during saves
        scaler.mean_ = checkpoint.get("scaler_mean", checkpoint.get("scaler_mean_", None))
        scaler.scale_ = checkpoint.get("scaler_scale", checkpoint.get("scaler_scale_", None))
        if scaler.mean_ is None or scaler.scale_ is None:
            raise KeyError("Scaler parameters not found in checkpoint (scaler_mean / scaler_scale)")

    # Instantiate and load model
    model = LSTMChurnModel(input_size=2, hidden_size=HIDDEN_SIZE)
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    user_ids, X = prepare_sequences(df, scaler)

    preds = []
    with torch.no_grad():
        for i in range(0, len(user_ids), BATCH_SIZE):
            batch_x = torch.tensor(X[i:i + BATCH_SIZE], dtype=torch.float32).to(device)
            outputs = model(batch_x).squeeze(1)  # (batch,)
            preds.extend(outputs.cpu().numpy())

    probs = np.array(preds)
    churn_bins = (probs >= THRESHOLD).astype(int)

    results = pd.DataFrame({
        "user_id": user_ids,
        "churn_probability": probs,
        "churn": churn_bins
    })

    # sort by user_id to keep deterministic output
    results = results.sort_values("user_id").reset_index(drop=True)

    if output_path:
        # write full result with header
        results.to_csv(output_path, index=False)
        print(f"[+] Предсказания сохранены в {output_path}")
        return None
    else:
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", "-i", required=True, help="Путь к входному CSV с сырыми транзакциями")
    parser.add_argument("--output-path", "-o", required=False, help="Путь для сохранения предсказаний (если не указан — вернёт DataFrame)")
    args = parser.parse_args()

    main(args.input_path, args.output_path)
