from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from feast import FeatureStore
import pandas as pd
import argparse


def main(entity_csv: str = 'feature_repo/data/entities.csv', repo_path: str = 'feature_repo'):
    store = FeatureStore(repo_path=repo_path)

    entity_df = pd.read_csv(entity_csv)

    # We assume entity_df has event_timestamp, user_id and label churn
    features = [
        'user_features:amount_sum',
        'user_features:amount_count'
    ]

    training_df = store.get_historical_features(entity_df=entity_df, features=features).to_df()
    # store.get_historical_features returns a DataFrame with column names: feature views
    # Merge label
    # We assume churn is in entity_df
    # merge labels on user and event time
    merged = training_df.merge(entity_df[['user_id', 'event_timestamp', 'churn']], on=['user_id', 'event_timestamp'])

    X = merged[['amount_sum', 'amount_count']]
    y = merged['churn']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    print(f"Model ROC-AUC on validation: {auc:.4f}")

    # Save model
    import joblib
    joblib.dump(model, 'models/logreg_feast.pkl')
    print('Model saved to models/logreg_feast.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--entity-csv', default='feature_repo/data/entities.csv')
    parser.add_argument('--repo', default='feature_repo')
    args = parser.parse_args()
    main(args.entity_csv, args.repo)
