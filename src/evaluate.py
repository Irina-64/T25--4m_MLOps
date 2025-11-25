# src\evaluate.py
import sys, os

# Добавляет корневую директорию проекта в PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import mlflow
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from Core.agents import RLAgent


MODEL_PATH = "rl_agent_model.pth"
REPORT_DIR = "reports/eval.json"


def load_test_data(path="replays/test_replays.json"):
    """
    Загружает тестовый датасет реплеев.
    Ожидается формат:
    [
        {"state": [...], "action": int, "reward": float, "done": bool},
        ...
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_model(model, test_data):
    """
    Прогоняет модель через тестовые состояния и возвращает метрики.
    """

    y_true = []
    y_pred = []

    for sample in test_data:
        state = torch.tensor(sample["state"], dtype=torch.float32).unsqueeze(0)
        true_label = sample["label"]

        with torch.no_grad():
            logits = model(state)
            prob = torch.sigmoid(logits).item()

        y_pred.append(prob)
        y_true.append(true_label)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Бинаризация прогнозов
    y_bin = (y_pred > 0.5).astype(int)

    # Метрики
    return {
        "roc_auc": float(roc_auc_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_bin)),
        "recall": float(recall_score(y_true, y_bin)),
        "f1": float(f1_score(y_true, y_bin)),
        "confusion_matrix": confusion_matrix(y_true, y_bin).tolist(),
    }


def save_report(report: dict, path=REPORT_DIR):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    print(f"[INFO] Report saved to {path}")


def register_model_mlflow(run_id, name="durak_rl_model"):
    """
    Регистрирует модель в MLflow Model Registry.
    """
    model_uri = f"runs:/{run_id}/model"
    print(f"[MLFLOW] Registering model: {model_uri}")

    result = mlflow.register_model(model_uri, name)
    print(f"[MLFLOW] Registered as version: {result.version}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="replays/test_replays.json",
                        help="Путь к тестовому датасету")
    parser.add_argument("--register", action="store_true",
                        help="Зарегистрировать модель в MLflow Registry")
    args = parser.parse_args()

    # MLflow
    mlflow.set_experiment("durak-evaluation")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"[MLFLOW] Run ID: {run_id}")

        # Загружаем тестовые данные
        test_data = load_test_data(args.test)
        print(f"[INFO] Loaded {len(test_data)} test samples")

        # Загружаем модель
        agent = RLAgent(pid=0, state_size=len(test_data[0]["state"]), action_size=50)
        agent.model.load_state_dict(torch.load(MODEL_PATH))
        model = agent.model
        print(f"[INFO] RL model loaded")

        # ОЦЕНКА
        metrics = evaluate_model(model, test_data)

        # ЛОГИРУЕМ В MLflow
        for k, v in metrics.items():
            if k == "confusion_matrix":
                mlflow.log_dict({k: v}, f"confmat.json")
            else:
                mlflow.log_metric(k, v)

        # СОХРАНЯЕМ ОТЧЁТ
        save_report(metrics)

        # Логируем отчёт как артефакт
        mlflow.log_artifact(REPORT_DIR)

        # Регистрируем модель
        if args.register:
            mlflow.pytorch.log_model(model, "model")
            register_model_mlflow(run_id)

        print("\n=== Evaluation finished ===")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()



# python src/evaluate.py --test replays/replay_20251113_145258 copy.json