from feast import FeatureStore
import pandas as pd
import argparse

def main(entity_df_path, repo_path, output_path):
    store = FeatureStore(repo_path=repo_path)
    entity_df = pd.read_csv(entity_df_path)
    features = [
        'user_features:amount_sum',
        'user_features:amount_count'
    ]
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=features
    ).to_df()

    training_df.to_csv(output_path, index=False)
    print(f"Training data with features saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--entity-csv', default='feature_repo/data/entities.csv')
    parser.add_argument('--repo', default='feature_repo')
    parser.add_argument('--out', default='data/training_with_features.csv')
    args = parser.parse_args()
    main(args.entity_csv, args.repo, args.out)
