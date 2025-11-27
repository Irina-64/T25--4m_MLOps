import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path


def make_dummy_raw(path: str, n_users: int = 50, seq_len: int = 30):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    # Build small dataset where each user has seq_len entries
    rows = []
    for uid in range(1, n_users + 1):
        churn = int(np.random.rand() > 0.8)
        for day in range(seq_len):
            date = pd.to_datetime('2024-01-01') + pd.Timedelta(days=day)
            amount = float(np.abs(np.random.randn() * 20))
            rows.append({'user_id': uid, 'date': date.strftime('%Y-%m-%d'), 'amount': amount, 'churn': churn})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Dummy data written to {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-path', default='data/raw/churn_predict.csv')
    args = parser.parse_args()
    if not os.path.exists(args.raw_path):
        make_dummy_raw(args.raw_path)
    else:
        print('Raw data exists, skipping generation')
