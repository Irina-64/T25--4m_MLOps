import argparse
import pandas as pd
import numpy as np
import joblib
import pickle
import json
import os

from datetime import datetime
from typing import List, Dict

from pathlib import Path
from feast import FeatureStore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report


class PersonalityClassifier:
    FEATURE_NAMES = ["time_broken_spent_alone","stage_fear", "social_event_attendance","going_outside","drained_after_socializing","friends_circle_size","post_frequency"]
    TARGET_NAME = "personality_encoded"
    
    def __init__(self, feature_store_path: str = "/opt/airflow/feature_repo"):
        self.fs = FeatureStore(repo_path=feature_store_path)
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data_from_feast(self, user_ids: List[int]) -> pd.DataFrame:
        entity_df = pd.DataFrame({
            "user_id": user_ids,
            "event_timestamp": pd.to_datetime([datetime.now()] * len(user_ids))
        })
        feature_refs = [
            f"personality_features:{feature}" for feature in self.FEATURE_NAMES
        ] + [f"personality_features:{self.TARGET_NAME}"]

        training_df = self.fs.get_historical_features(
            entity_df=entity_df,
            features=feature_refs
        ).to_df()

        training_df.columns = [col.replace("personality_features__", "") 
                              for col in training_df.columns]
        
        return training_df
    
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2) -> tuple:
        missing_features = [f for f in self.FEATURE_NAMES if f not in df.columns]
        if missing_features:
            raise ValueError(f"Отсутствуют фичи: {missing_features}")
        
        X = df[self.FEATURE_NAMES].copy()
        y = df[self.TARGET_NAME]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=42,
            stratify=y
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, X_train: np.ndarray, y_train: pd.Series) -> None:
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)    
   
    def evaluate(self, X_test: np.ndarray, y_test: pd.Series) -> Dict:
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
        }
        
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics['detailed_report'] = report
        
        print("\n Результаты оценки:")
        for metric, value in metrics.items():
            if metric != 'detailed_report':
                print(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def save_model(self, output_dir: str = "models") -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, f"{output_dir}/personality_model.joblib")
        joblib.dump(self.scaler, f"{output_dir}/personality_scaler.joblib")

        feature_info = {
            'features': self.FEATURE_NAMES,
            'target': self.TARGET_NAME,
            'model_type': 'RandomForestClassifier',
            'feature_store_used': True,
            'training_date': datetime.now().isoformat()
        }
        
        with open(f"{output_dir}/feature_info.json", 'w') as f:
            json.dump(feature_info, f, indent=2)
        

def main():
    parser = argparse.ArgumentParser(description='Обучение модели личности с Feast')
    parser.add_argument('--feature-store', type=str, default='/opt/airflow/feature_repo',
                       help='Путь к Feast Feature Store')
    parser.add_argument('--num-users', type=int, default=100,
                       help='Количество пользователей для загрузки')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Директория для сохранения модели')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Доля тестовых данных')
    
    args = parser.parse_args()  

    classifier = PersonalityClassifier(args.feature_store)
    
    try:
        user_ids = list(range(1, args.num_users + 1))
        df = classifier.load_data_from_feast(user_ids)

        X_train, X_test, y_train, y_test = classifier.prepare_data(
            df, test_size=args.test_size
        )

        classifier.train(X_train, y_train)

        metrics = classifier.evaluate(X_test, y_test)
        classifier.save_model(args.output_dir)

        reference_path = '/opt/airflow/data/processed/train_reference.pkl'
        os.makedirs(os.path.dirname(reference_path), exist_ok=True)

        with open(reference_path, 'wb') as f:
            pickle.dump(df[classifier.FEATURE_NAMES], f)
        

        metrics_path = f"{args.output_dir}/feast_training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        print(f"\n Обучение завершено!")
        print(f"   Модель: {args.output_dir}/personality_model.joblib")
        print(f"   Метрики: {metrics_path}")
        print(f"Эталонные данные для дрейфа сохранены: {reference_path}")
        
    except Exception as e:
        print(f"Ошибка: {str(e)}")
        raise

if __name__ == "__main__":
    main()