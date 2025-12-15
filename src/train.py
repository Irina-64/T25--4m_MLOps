import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from typing import Dict, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import mlflow
import mlflow.sklearn


class Classifier:
    
    FEATURES = ['Time_broken_spent_Alone','Stage_fear','Social_event_attendance','Going_outside','Drained_after_socializing','Friends_circle_size','Post_frequency']
    TARGET = 'Personality_encoded'
    
    def __init__(self, model_type: str = 'rf', random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        df = pd.read_csv(data_path)

        return df
    
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple:
        X = df[self.FEATURES].copy()
        y = df[self.TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train: np.ndarray, y_train: pd.Series) -> None:
        if self.model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=150,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced'
            )
            
        elif self.model_type == 'lr':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                solver='liblinear',
                class_weight='balanced',
                C=1.0
            )

        self.model.fit(X_train, y_train)
    
    
    def evaluate_model(self, X_test: np.ndarray, y_test: pd.Series) -> Dict[str, float]:
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        metrics['classification_report'] = report
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'confusion_matrix': cm.tolist(),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,  # TNR
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
        })
        
        return metrics
    
    def log_to_mlflow(self, metrics: Dict, experiment_name: str) -> None:
        mlflow.set_experiment(experiment_name)
    
        with mlflow.start_run():
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("features", ", ".join(self.FEATURES))
            mlflow.log_param("random_state", self.random_state)
            mlflow.log_param("task", "binary_classification")
            
            for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
                mlflow.log_metric(metric, metrics[metric])
            
            mlflow.log_metric("specificity", metrics['specificity'])
            mlflow.log_dict({"confusion_matrix": metrics['confusion_matrix']}, 
                           "confusion_matrix.json")
            
            mlflow.sklearn.log_model(
                self.model,
                "model",
                registered_model_name=f"personality_{self.model_type}_classifier"
            )
    

    def save_artifacts(self, metrics: Dict, output_dir: str = 'models') -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model_path = Path(output_dir) / 'classifier_model.joblib'
        joblib.dump(self.model, model_path)
        
        scaler_path = Path(output_dir) / 'scaler.joblib'
        joblib.dump(self.scaler, scaler_path)
        
        metrics_path = Path(output_dir) / 'classification_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        features_info = {
            'features': self.FEATURES,
            'target': self.TARGET,
            'model_type': self.model_type,
        }
        
        features_path = Path(output_dir) / 'features_info.json'
        with open(features_path, 'w') as f:
            json.dump(features_info, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/opt/airflow/data/processed/processed.csv',
                       help='Путь к обработанным данным')
    parser.add_argument('--model', type=str, choices=['rf', 'lr'], default='rf',
                       help='Тип модели: rf (RandomForest), lr (LogisticRegression)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Доля тестовых данных')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Директория для сохранения модели')
    parser.add_argument('--experiment-name', type=str, 
                       default='personality_binary_classification',
                       help='Название эксперимента в MLflow')
    
    args = parser.parse_args()
    classifier = Classifier(model_type=args.model)
    
    try:
        df = classifier.load_data(args.data)
        X_train, X_test, y_train, y_test = classifier.prepare_data(
            df, test_size=args.test_size
        )
        classifier.train_model(X_train, y_train)
        metrics = classifier.evaluate_model(X_test, y_test)
        classifier.log_to_mlflow(metrics, args.experiment_name)
        classifier.save_artifacts(metrics, args.output_dir)

    except Exception as e:
        print(f"\n Ошибка: {str(e)}")
        raise


if __name__ == "__main__":
    main()