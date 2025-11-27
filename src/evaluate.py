import argparse
import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

SEQ_LEN = 335

class LSTMChurnModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        if hn.size(0) == 2:
            hn_cat = torch.cat((hn[0], hn[1]), dim=1)
        else:
            hn_cat = hn[-1]
        return torch.sigmoid(self.fc(hn_cat))


def prepare_sequences(df: pd.DataFrame, seq_len: int = SEQ_LEN):
    df['date'] = pd.to_datetime(df['date'])
    df['day'] = (df['date'] - pd.to_datetime('2024-01-01')).dt.days
    df = df[(df['day'] >= 0) & (df['day'] < seq_len)]
    agg = df.groupby(['user_id', 'day']).agg(amount_sum=('amount', 'sum'), amount_count=('amount', 'count')).reset_index()
    all_days = pd.DataFrame({'day': range(seq_len)})
    user_ids = np.array(agg['user_id'].unique())
    data = []
    labels = []
    for uid in user_ids:
        user_df = agg[agg['user_id'] == uid].merge(all_days, on='day', how='right').fillna(0)
        user_df = user_df.sort_values('day')
        data.append(user_df[['amount_sum', 'amount_count']].values)
        # label: get churn from original df (assumes same label for user)
        label = df[df['user_id'] == uid]['churn'].drop_duplicates()
        labels.append(int(label.iloc[0]) if len(label) > 0 else 0)
    X = np.stack(data)
    y = np.array(labels)
    return user_ids, X, y


def main(processed_path: str, model_path: str, out_json: str):
    if not os.path.exists(processed_path):
        print(f"Processed path {processed_path} does not exist")
        return
    df = pd.read_csv(processed_path)
    user_ids, X, y = prepare_sequences(df)

    # load scaler and model
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found")
        return
    ckpt = torch.load(model_path, map_location=torch.device('cpu'))

    scaler = StandardScaler()
    scaler.mean_ = ckpt.get('scaler_mean')
    scaler.scale_ = ckpt.get('scaler_scale')

    X_reshaped = X.reshape(-1, X.shape[-1])
    X_scaled = (X_reshaped - scaler.mean_) / scaler.scale_
    X_scaled = X_scaled.reshape(X.shape)

    model = LSTMChurnModel(input_size=2, hidden_size=64)
    model.load_state_dict(ckpt['model'])
    model.eval()

    with torch.no_grad():
        outputs = model(torch.tensor(X_scaled, dtype=torch.float32))
        probs = outputs.numpy().flatten()

    metric = float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else 0.0
    result = {'roc_auc': metric}
    with open(out_json, 'w') as f:
        json.dump(result, f)
    print(f"Evaluation saved to {out_json}: {result}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-path', default='data/processed/processed.csv')
    parser.add_argument('--model-path', default='model.pt')
    parser.add_argument('--out-json', default='data/result.json')
    args = parser.parse_args()
    main(args.processed_path, args.model_path, args.out_json)
