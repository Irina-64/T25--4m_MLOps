import pandas as pd
import numpy as np
import json
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def evaluate_model(model_path: str = None, log_to_mlflow: bool = True):
    """
    Оценить модель на тестовом наборе и сгенерировать подробный отчет.
    
    Args:
        model_path: путь к сохраненной модели (если None, загружает из models/)
        log_to_mlflow: логировать ли метрики в MLflow
    
    Returns:
        dict: словарь с метриками оценки
    """
    
    print("=" * 80)
    print("ОЦЕНКА МОДЕЛИ")
    print("=" * 80)
    
    # Создание директории для отчетов
    os.makedirs('reports', exist_ok=True)
    
    # Загрузка обработанных данных
    print("\n📊 Загрузка данных...")
    df = pd.read_csv('data/processed/processed.csv')
    print(f"✓ Данные загружены. Размер: {df.shape}")
    
    # Разделение на features и target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✓ Train size: {X_train.shape}, Test size: {X_test.shape}")
    print(f"✓ Баланс классов в тесте: {y_test.value_counts().to_dict()}")
    
    # Загрузка модели
    print("\n🤖 Загрузка модели...")
    if model_path is None:
        # Пытаемся найти последнюю модель
        model_files = [
            f for f in os.listdir('models/') 
            if f.endswith('.joblib')
        ]
        if not model_files:
            raise FileNotFoundError("Нет сохраненных моделей в папке models/")
        model_path = f"models/{sorted(model_files)[-1]}"
    
    model = joblib.load(model_path)
    print(f"✓ Модель загружена из: {model_path}")
    print(f"  Тип модели: {type(model).__name__}")
    
    # Предсказания
    print("\n🔮 Генерирование предсказаний...")
    y_pred = model.predict(X_test)
    
    # Вероятности для ROC-AUC
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = y_pred
    
    # Вычисление метрик
    print("\n📈 Вычисление метрик...")
    
    # Основные метрики
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Classification Report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # ROC Curve data
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    # Подготовка результатов
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "model_type": type(model).__name__,
        "test_size": int(X_test.shape[0]),
        "n_features": int(X_test.shape[1]),
        "metrics": {
            "roc_auc": float(roc_auc),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "specificity": float(specificity)
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "class_distribution": {
            "class_0": int((y_test == 0).sum()),
            "class_1": int((y_test == 1).sum())
        },
        "classification_report": class_report
    }
    
    # Вывод метрик в консоль
    print("\n" + "="*80)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("="*80)
    print(f"\n📊 Основные метрики:")
    print(f"  • ROC-AUC:    {roc_auc:.4f}")
    print(f"  • Accuracy:   {accuracy:.4f}")
    print(f"  • Precision:  {precision:.4f}")
    print(f"  • Recall:     {recall:.4f}")
    print(f"  • F1-Score:   {f1:.4f}")
    print(f"  • Specificity: {specificity:.4f}")
    
    print(f"\n🎯 Confusion Matrix:")
    print(f"  True Negatives:  {tn:4d}")
    print(f"  False Positives: {fp:4d}")
    print(f"  False Negatives: {fn:4d}")
    print(f"  True Positives:  {tp:4d}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Сохранение отчета в JSON
    print("\n💾 Сохранение отчетов...")
    json_path = 'reports/eval.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"✓ JSON отчет сохранен: {json_path}")
    
    # Создание HTML отчета с визуализацией
    html_path = 'reports/eval.html'
    generate_html_report(
        html_path, metrics, X_test, y_test, y_pred, y_pred_proba
    )
    print(f"✓ HTML отчет сохранен: {html_path}")
    
    # Логирование в MLflow (если есть активный run)
    if log_to_mlflow:
        try:
            print("\n📤 Логирование в MLflow...")
            # Если нет активного run, создаем новый
            if mlflow.active_run() is None:
                mlflow.start_run(run_name=f"evaluation_{datetime.now().strftime('%H%M%S')}")
            
            # Логируем метрики
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("specificity", specificity)
            
            # Логируем артефакты
            mlflow.log_artifact(json_path)
            mlflow.log_artifact(html_path)
            
            print("✓ Метрики и артефакты залогированы в MLflow")
        except Exception as e:
            print(f"⚠ Не удалось залогировать в MLflow: {e}")
    
    print("\n" + "="*80)
    print("✅ ОЦЕНКА ЗАВЕРШЕНА")
    print("="*80)
    
    return metrics


def generate_html_report(
    html_path: str,
    metrics: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray
):
    """Генерирование HTML отчета с визуализацией."""
    
    # Создание визуализаций
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Evaluation Report - Model Performance', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix
    cm = metrics['confusion_matrix']
    cm_array = np.array([
        [cm['true_negatives'], cm['false_positives']],
        [cm['false_negatives'], cm['true_positives']]
    ])
    sns.heatmap(
        cm_array,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=axes[0, 0],
        cbar=False
    )
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')
    
    # 2. ROC Curve
    fpr, tpr, _ = __import__('sklearn.metrics', fromlist=['roc_curve']).roc_curve(
        y_test, y_pred_proba
    )
    roc_auc = metrics['metrics']['roc_auc']
    axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend(loc="lower right")
    
    # 3. Metrics Bar Chart
    metric_names = ['ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_values = [
        metrics['metrics']['roc_auc'],
        metrics['metrics']['accuracy'],
        metrics['metrics']['precision'],
        metrics['metrics']['recall'],
        metrics['metrics']['f1']
    ]
    colors = ['#1f77b4' if v >= 0.8 else '#ff7f0e' for v in metric_values]
    axes[1, 0].barh(metric_names, metric_values, color=colors)
    axes[1, 0].set_xlim([0, 1])
    axes[1, 0].set_xlabel('Score')
    axes[1, 0].set_title('Metrics Summary')
    for i, v in enumerate(metric_values):
        axes[1, 0].text(v + 0.02, i, f'{v:.4f}', va='center')
    
    # 4. Prediction Distribution
    axes[1, 1].hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7, label='Class 0', color='blue')
    axes[1, 1].hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7, label='Class 1', color='red')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Prediction Probability Distribution')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Сохранение графика
    plot_path = html_path.replace('.html', '_plot.png')
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    # Генерирование HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Model Evaluation Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            }}
            h1 {{
                color: #333;
                border-bottom: 3px solid #007bff;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #555;
                margin-top: 30px;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .metric-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
            }}
            .metric-card.high {{
                background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
                color: #333;
            }}
            .metric-card.low {{
                background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
                color: #333;
            }}
            .metric-value {{
                font-size: 32px;
                font-weight: bold;
                margin: 10px 0;
            }}
            .metric-label {{
                font-size: 14px;
                opacity: 0.9;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th {{
                background-color: #007bff;
                color: white;
                padding: 12px;
                text-align: left;
            }}
            td {{
                padding: 10px 12px;
                border-bottom: 1px solid #ddd;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .plot-container {{
                text-align: center;
                margin: 30px 0;
            }}
            .plot-container img {{
                max-width: 100%;
                height: auto;
                border-radius: 8px;
            }}
            .info-box {{
                background-color: #e7f3ff;
                border-left: 4px solid #007bff;
                padding: 15px;
                margin: 15px 0;
                border-radius: 4px;
            }}
            .info-box strong {{
                color: #007bff;
            }}
            footer {{
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                color: #666;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Model Evaluation Report</h1>
            
            <div class="info-box">
                <strong>Model Type:</strong> {metrics['model_type']}<br>
                <strong>Model Path:</strong> {metrics['model_path']}<br>
                <strong>Test Size:</strong> {metrics['test_size']} samples | 
                <strong>Features:</strong> {metrics['n_features']}<br>
                <strong>Generated:</strong> {metrics['timestamp']}
            </div>
            
            <h2>📊 Key Performance Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-card {'high' if metrics['metrics']['roc_auc'] >= 0.8 else 'low'}">
                    <div class="metric-label">ROC-AUC</div>
                    <div class="metric-value">{metrics['metrics']['roc_auc']:.4f}</div>
                </div>
                <div class="metric-card {'high' if metrics['metrics']['accuracy'] >= 0.8 else 'low'}">
                    <div class="metric-label">Accuracy</div>
                    <div class="metric-value">{metrics['metrics']['accuracy']:.4f}</div>
                </div>
                <div class="metric-card {'high' if metrics['metrics']['precision'] >= 0.8 else 'low'}">
                    <div class="metric-label">Precision</div>
                    <div class="metric-value">{metrics['metrics']['precision']:.4f}</div>
                </div>
                <div class="metric-card {'high' if metrics['metrics']['recall'] >= 0.8 else 'low'}">
                    <div class="metric-label">Recall</div>
                    <div class="metric-value">{metrics['metrics']['recall']:.4f}</div>
                </div>
                <div class="metric-card {'high' if metrics['metrics']['f1'] >= 0.8 else 'low'}">
                    <div class="metric-label">F1-Score</div>
                    <div class="metric-value">{metrics['metrics']['f1']:.4f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Specificity</div>
                    <div class="metric-value">{metrics['metrics']['specificity']:.4f}</div>
                </div>
            </div>
            
            <h2>🔍 Confusion Matrix</h2>
            <table>
                <tr>
                    <th></th>
                    <th>Predicted Negative</th>
                    <th>Predicted Positive</th>
                </tr>
                <tr>
                    <th>Actual Negative</th>
                    <td>{metrics['confusion_matrix']['true_negatives']}</td>
                    <td>{metrics['confusion_matrix']['false_positives']}</td>
                </tr>
                <tr>
                    <th>Actual Positive</th>
                    <td>{metrics['confusion_matrix']['false_negatives']}</td>
                    <td>{metrics['confusion_matrix']['true_positives']}</td>
                </tr>
            </table>
            
            <h2>📈 Visualizations</h2>
            <div class="plot-container">
                <img src="{plot_path.split('/')[-1]}" alt="Evaluation Plots">
            </div>
            
            <h2>📋 Classification Report</h2>
            <table>
                <tr>
                    <th>Class</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>Support</th>
                </tr>
    """
    
    # Добавление строк для каждого класса
    for class_name in ['0', '1']:
        class_data = metrics['classification_report'].get(class_name, {})
        html_content += f"""
                <tr>
                    <td><strong>Class {class_name}</strong></td>
                    <td>{class_data.get('precision', 0):.4f}</td>
                    <td>{class_data.get('recall', 0):.4f}</td>
                    <td>{class_data.get('f1-score', 0):.4f}</td>
                    <td>{int(class_data.get('support', 0))}</td>
                </tr>
        """
    
    html_content += """
            </table>
            
            <h2>💡 Key Insights</h2>
            <ul>
    """
    
    # Добавление выводов
    roc_auc = metrics['metrics']['roc_auc']
    accuracy = metrics['metrics']['accuracy']
    precision = metrics['metrics']['precision']
    recall = metrics['metrics']['recall']
    specificity = metrics['metrics']['specificity']
    
    insights = []
    if roc_auc >= 0.9:
        insights.append("✅ Excellent ROC-AUC score - model has excellent discrimination ability")
    elif roc_auc >= 0.8:
        insights.append("✅ Good ROC-AUC score - model shows good discrimination ability")
    else:
        insights.append("⚠️ Low ROC-AUC score - model discrimination ability needs improvement")
    
    if precision >= recall:
        insights.append(f"📌 Precision ({precision:.4f}) > Recall ({recall:.4f}) - model is conservative")
    else:
        insights.append(f"📌 Recall ({recall:.4f}) > Precision ({precision:.4f}) - model is aggressive")
    
    if specificity >= 0.8:
        insights.append(f"✅ Good Specificity ({specificity:.4f}) - model correctly identifies negatives")
    else:
        insights.append(f"⚠️ Low Specificity ({specificity:.4f}) - many false positives")
    
    for insight in insights:
        html_content += f"                <li>{insight}</li>\n"
    
    html_content += """
            </ul>
            
            <footer>
                <p>Generated automatically by src/evaluate.py</p>
                <p>For more details, see eval.json</p>
            </footer>
        </div>
    </body>
    </html>
    """
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


if __name__ == "__main__":
    # Запуск оценки
    metrics = evaluate_model()
