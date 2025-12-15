import json
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from pathlib import Path


def main():
    model = joblib.load('/opt/airflow/models/classifier_model.joblib')
    scaler = joblib.load('/opt/airflow/models/scaler.joblib')
    data = pd.read_csv('/opt/airflow/data/processed/processed.csv')
    
    features = [
        'Time_broken_spent_Alone',
        'Stage_fear',
        'Social_event_attendance',
        'Going_outside',
        'Drained_after_socializing',
        'Friends_circle_size',
        'Post_frequency'
    ]
    target = 'Personality_encoded'
    
    X = data[features]
    y = data[target]
    
    test_size = 0.2
    test_count = int(len(X) * test_size)
    X_test = X.iloc[-test_count:].values
    y_test = y.iloc[-test_count:].values
    
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
        'test_samples': int(len(y_test))
    }
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics.update({
        'confusion_matrix': cm.tolist(),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    })
    
    Path('/opt/airflow/reports').mkdir(exist_ok=True)
    with open('/opt/airflow/reports/eval.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    

if __name__ == "__main__":
    main()