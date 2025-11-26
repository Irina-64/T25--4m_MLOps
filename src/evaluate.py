import pandas as pd
import numpy as np
import json
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, accuracy_score
)
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import prepare_features, get_feature_names

def load_test_data():
    """Загрузка и подготовка тестовых данных"""
    # Загружаем обработанные данные
    data = pd.read_csv('data/processed/processed.csv')
    data['date'] = pd.to_datetime(data['date'])
    
    # Применяем ту же подготовку признаков, что и при обучении
    data = prepare_features(data)
    
    # Получаем имена признаков
    try:
        feature_names = joblib.load('models/feature_names.joblib')
    except:
        # Если файл не существует, используем функцию для получения имен
        feature_names = get_feature_names()
        print("Использованы сгенерированные имена признаков")
    
    # Проверяем, что все признаки существуют
    missing_features = [f for f in feature_names if f not in data.columns]
    if missing_features:
        print(f"Предупреждение: отсутствуют признаки: {missing_features}")
        # Используем только существующие признаки
        feature_names = [f for f in feature_names if f in data.columns]
    
    # Проверяем наличие целевой переменной
    if 'USD_RUB_target' not in data.columns:
        raise KeyError("Целевая переменная 'USD_RUB_target' не найдена в данных")
    
    # Используем те же признаки
    X = data[feature_names]
    y = data['USD_RUB_target']
    
    # Берем последние 20% данных как тестовую выборку
    test_size = int(0.2 * len(data))
    X_test = X.iloc[-test_size:]
    y_test = y.iloc[-test_size:]
    
    print(f"Тестовая выборка: {X_test.shape}")
    print(f"Признаки: {len(feature_names)}")
    
    # Масштабируем
    scaler = joblib.load('models/scaler.joblib')
    X_test_scaled = scaler.transform(X_test)
    
    return X_test_scaled, y_test, feature_names

def generate_evaluation_report(model, X_test, y_test, model_name):
    """Генерация отчета с метриками"""
    
    # Предсказания
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Вычисление метрик
    metrics = {
        'model_name': model_name,
        'evaluation_date': datetime.now().isoformat(),
        'test_set_size': len(X_test),
        'metrics': {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, zero_division=0))
        },
        'class_distribution': {
            'class_0_count': int(sum(y_test == 0)),
            'class_1_count': int(sum(y_test == 1)),
            'class_0_ratio': float(sum(y_test == 0) / len(y_test)),
            'class_1_ratio': float(sum(y_test == 1) / len(y_test))
        }
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = {
        'true_negative': int(cm[0, 0]),
        'false_positive': int(cm[0, 1]),
        'false_negative': int(cm[1, 0]),
        'true_positive': int(cm[1, 1])
    }
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    metrics['classification_report'] = report
    
    return metrics, y_pred, y_pred_proba

def save_json_report(metrics, output_path):
    """Сохранение отчета в JSON"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

def save_html_report(metrics, y_test, y_pred_proba, output_path):
    """Сохранение отчета в HTML с визуализациями"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Создаем визуализации
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Confusion Matrix
    cm = np.array([
        [metrics['confusion_matrix']['true_negative'], metrics['confusion_matrix']['false_positive']],
        [metrics['confusion_matrix']['false_negative'], metrics['confusion_matrix']['true_positive']]
    ])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # 2. ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    ax2.plot(fpr, tpr, label=f"ROC AUC = {metrics['metrics']['roc_auc']:.3f}")
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend()
    
    # 3. Class Distribution
    classes = ['Class 0', 'Class 1']
    counts = [metrics['class_distribution']['class_0_count'], 
              metrics['class_distribution']['class_1_count']]
    ax3.bar(classes, counts, color=['skyblue', 'lightcoral'])
    ax3.set_title('Class Distribution')
    ax3.set_ylabel('Count')
    
    # 4. Metrics Comparison
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_values = [
        metrics['metrics']['accuracy'],
        metrics['metrics']['precision'], 
        metrics['metrics']['recall'],
        metrics['metrics']['f1_score']
    ]
    ax4.bar(metric_names, metric_values, color='lightgreen')
    ax4.set_title('Model Metrics')
    ax4.set_ylabel('Score')
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path.replace('.html', '_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Создаем HTML отчет
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .metric {{ margin: 10px 0; padding: 10px; background: #f5f5f5; }}
            .good {{ background: #d4edda; }}
            .warning {{ background: #fff3cd; }}
        </style>
    </head>
    <body>
        <h1>Model Evaluation Report</h1>
        <h2>Model: {metrics['model_name']}</h2>
        <p>Evaluation Date: {metrics['evaluation_date']}</p>
        <p>Test Set Size: {metrics['test_set_size']}</p>
        
        <h3>Key Metrics</h3>
        <div class="metric {'good' if metrics['metrics']['accuracy'] > 0.6 else 'warning'}">
            <strong>Accuracy:</strong> {metrics['metrics']['accuracy']:.3f}
        </div>
        <div class="metric {'good' if metrics['metrics']['roc_auc'] > 0.7 else 'warning'}">
            <strong>ROC AUC:</strong> {metrics['metrics']['roc_auc']:.3f}
        </div>
        <div class="metric {'good' if metrics['metrics']['precision'] > 0.6 else 'warning'}">
            <strong>Precision:</strong> {metrics['metrics']['precision']:.3f}
        </div>
        <div class="metric {'good' if metrics['metrics']['recall'] > 0.6 else 'warning'}">
            <strong>Recall:</strong> {metrics['metrics']['recall']:.3f}
        </div>
        <div class="metric {'good' if metrics['metrics']['f1_score'] > 0.6 else 'warning'}">
            <strong>F1-Score:</strong> {metrics['metrics']['f1_score']:.3f}
        </div>
        
        <h3>Visualizations</h3>
        <img src="{output_path.replace('.html', '_plots.png')}" alt="Model Evaluation Plots" width="800">
        
        <h3>Detailed Report</h3>
        <pre>{json.dumps(metrics, indent=2)}</pre>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

def main():
    """Основная функция оценки"""
    print("=== Model Evaluation ===")
    
    # Создаем директорию для отчетов
    os.makedirs('reports', exist_ok=True)
    
    try:
        # Загружаем тестовые данные
        print("Loading test data...")
        X_test, y_test, feature_names = load_test_data()
        print(f"Test set size: {X_test.shape}")
        
        # Загружаем лучшую модель
        print("Loading best model...")
        model = joblib.load('models/best_model.joblib')
        model_name = type(model).__name__
        print(f"Model: {model_name}")
        
        # Генерируем отчет
        print("Generating evaluation report...")
        metrics, y_pred, y_pred_proba = generate_evaluation_report(model, X_test, y_test, model_name)
        
        # Сохраняем отчеты
        save_json_report(metrics, 'reports/eval.json')
        save_html_report(metrics, y_test, y_pred_proba, 'reports/eval.html')
        
        # Логируем в MLflow
        print("Logging to MLflow...")
        mlflow.set_experiment("flight_delay")
        
        with mlflow.start_run(run_name="evaluation"):
            # Логируем метрики
            for metric_name, metric_value in metrics['metrics'].items():
                mlflow.log_metric(f"eval_{metric_name}", metric_value)
            
            # Логируем отчеты как артефакты
            mlflow.log_artifact('reports/eval.json', artifact_path="evaluation")
            mlflow.log_artifact('reports/eval.html', artifact_path="evaluation")
            if os.path.exists('reports/eval_plots.png'):
                mlflow.log_artifact('reports/eval_plots.png', artifact_path="evaluation")
            
            # Логируем модель
            mlflow.sklearn.log_model(model, "evaluated_model")
        
        print("=== Evaluation Complete ===")
        print(f"JSON report saved: reports/eval.json")
        print(f"HTML report saved: reports/eval.html")
        
        # Выводим основные метрики
        print("\n=== Key Metrics ===")
        for metric, value in metrics['metrics'].items():
            print(f"{metric}: {value:.3f}")
            
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
