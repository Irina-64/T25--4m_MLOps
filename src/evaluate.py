import json
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score
import os

# Пути к данным и моделям
TEST_PATH = "/workspace/data/processed/test_data.csv"
MODEL_PATH = "/workspace/models/lgb_model.joblib"
REPORT_PATH = "/workspace/reports/eval.json"

'''
def main():
    # Проверим, что нужные файлы существуют
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Не найден тестовый файл: {TEST_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Не найдена модель: {MODEL_PATH}")

    # Загружаем тестовые данные
    data = pd.read_csv(TEST_PATH)
    if "churn" not in data.columns:
        raise ValueError("В тестовом наборе нет столбца 'churn' (целевая переменная).")

    X_test = data.drop(columns=["churn", "user_id"], errors="ignore")
    y_test = data["churn"]

    # Загружаем модель
    model = joblib.load(MODEL_PATH)

    # Предсказания
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Метрики
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if y_proba is not None else None,
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    # Создадим папку для отчётов, если её нет
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    # Сохраняем метрики в JSON
    with open(REPORT_PATH, "w") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print(f"✅ Отчёт сохранён в {REPORT_PATH}")
    print(json.dumps(metrics, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()'''

def main():
    # === 1. Проверка путей ===
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Модель не найдена: {MODEL_PATH}")
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"❌ Тестовый датасет не найден: {TEST_PATH}")

    # === 2. Загрузка модели и данных ===
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(TEST_PATH)
    print(f"✅ Загружен тестовый датасет: {df.shape}")

    # === 3. Разделение на признаки и цель ===
    X_test = df.drop(columns=["user_id", "churn"])
    y_test = df["churn"]

    # === 4. Приведение тестовых данных к обучающим признакам ===
    train_feature_count = model.n_features_in_
    print(f"🔍 Модель обучена на {train_feature_count} признаках, тест содержит {X_test.shape[1]}.")

    # Оставляем только те колонки, которые использовались при обучении
    if X_test.shape[1] != train_feature_count:
        print("⚙️ Приводим тестовые данные к нужным признакам...")
        model_features = model.feature_name_ if hasattr(model, "feature_name_") else None
        if model_features:
            missing_cols = [f for f in model_features if f not in X_test.columns]
            extra_cols = [f for f in X_test.columns if f not in model_features]

            if missing_cols:
                print(f"⚠️ В тесте нет признаков: {missing_cols}")
                for col in missing_cols:
                    X_test[col] = 0  # добавим пустые колонки

            if extra_cols:
                print(f"⚠️ Удаляем лишние признаки: {extra_cols}")
                X_test = X_test[model_features]
        else:
            raise ValueError("❌ Модель не содержит информации о признаках.")

    # === 5. Предсказания ===
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # === 6. Метрики ===
    roc_auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred).tolist()

    results = {
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": conf_mat,
    }

    # === 7. Сохранение ===
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print("✅ Отчёт сохранён в:", REPORT_PATH)
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()

