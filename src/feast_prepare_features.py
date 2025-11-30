import pandas as pd
import os
from pathlib import Path
import argparse


def prepare_features(raw_path: str = 'data/raw/churn_predict.csv', out_path: str = 'feature_repo/data/features.csv'):
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(raw_path)
    # ensure date
    if 'date' not in df.columns and 'transaction_date' in df.columns:
        df['date'] = df['transaction_date']
    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df['date'].dt.floor('d')
    agg = df.groupby(['user_id', 'day']).agg(
        amount_sum=('amount', 'sum'),
        amount_count=('amount', 'count')
    ).reset_index()
    # rename day -> event_timestamp for Feast
    agg = agg.rename(columns={'day': 'event_timestamp'})
    agg.to_csv(out_path, index=False)
    print(f'Prepared features written to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-path', default='data/raw/churn_predict.csv')
    parser.add_argument('--out-path', default='feature_repo/data/features.csv')
    args = parser.parse_args()
    prepare_features(args.raw_path, args.out_path)
