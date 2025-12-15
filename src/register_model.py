import mlflow
import mlflow.sklearn
import json
import sys
from datetime import datetime
import os

# mlflow.set_tracking_uri("sqlite:///mlflow.db")

def get_best_run(experiment_name: str = "telco_churn"):
    """
    Получить лучший run (с наивысшим ROC-AUC) из эксперимента.
    """
    client = mlflow.tracking.MlflowClient()
    
    # Получить эксперимент
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"❌ Эксперимент '{experiment_name}' не найден")
        # Список доступных экспериментов
        experiments = client.search_experiments()
        print("\nДоступные эксперименты:")
        for exp in experiments:
            print(f"  - {exp.name} (id: {exp.experiment_id})")
        return None
    
    print(f"✓ Эксперимент найден: {experiment_name} (id: {experiment.experiment_id})")
    
    # Поиск лучшего run по ROC-AUC
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.roc_auc DESC"],
        max_results=10
    )
    
    if not runs:
        print(f"❌ No runs found in experiment '{experiment_name}'")
        return None
    
    best_run = runs[0]
    roc_auc = best_run.data.metrics.get('roc_auc', 0)
    
    print(f"\n📊 Best run found:")
    print(f"  Run ID: {best_run.info.run_id}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  Model Type: {best_run.data.params.get('model_type', 'unknown')}")
    
    return best_run


def register_model_in_registry(
    run_id: str = None,
    model_name: str = "telco_churn_model",
    experiment_name: str = "telco_churn"
):
    """
    Зарегистрировать модель в MLflow Model Registry.
    
    Args:
        run_id: ID of the run to register (if None, will use best run)
        model_name: Name for the model in the registry
        experiment_name: Name of the experiment
    """
    
    print("="*80)
    print("РЕГИСТРАЦИЯ МОДЕЛИ В MLFLOW MODEL REGISTRY")
    print("="*80)
    
    client = mlflow.tracking.MlflowClient()
    
    # Если run_id не указан, ищем лучший
    if not run_id:
        print(f"\n🔍 Поиск лучшего run в эксперименте '{experiment_name}'...")
        best_run = get_best_run(experiment_name)
        if not best_run:
            return False
        run_id = best_run.info.run_id
    else:
        print(f"\n✓ Using specified run_id: {run_id}")
    
    # Проверка, что модель есть в run
    model_uri = f"runs:/{run_id}/model"
    print(f"\n📦 Attempting to register model from: {model_uri}")
    
    try:
        # Регистрация модели
        model_version = mlflow.register_model(model_uri, model_name)
        print(f"\n✅ Модель успешно зарегистрирована!")
        print(f"  Model Name: {model_name}")
        print(f"  Version: {model_version.version}")
        print(f"  Run ID: {run_id}")
        
        # Обновление описания версии
        description = f"""
        Telco Churn Prediction Model - Version {model_version.version}
        Registered: {datetime.now().isoformat()}
        Run ID: {run_id}
        """
        
        client.update_model_version(
            name=model_name,
            version=model_version.version,
            description=description.strip()
        )
        print(f"✓ Description updated")
        
        # Переведение в Staging
        print(f"\n📤 Transitioning model to Staging...")
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        print(f"✓ Model moved to Staging")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error registering model: {e}")
        return False


def list_registered_models():
    """Список всех зарегистрированных моделей."""
    client = mlflow.tracking.MlflowClient()
    
    print("\n" + "="*80)
    print("REGISTERED MODELS IN MLFLOW REGISTRY")
    print("="*80)
    
    try:
        models = client.search_registered_models()
        
        if not models:
            print("No models registered yet.")
            return
        
        for model in models:
            print(f"\n📦 Model: {model.name}")
            print(f"   Created: {model.creation_timestamp}")
            print(f"   Versions:")
            for version in model.latest_versions:
                print(f"     - Version {version.version}: {version.current_stage}")
                if version.description:
                    print(f"       Description: {version.description[:100]}...")
    except Exception as e:
        print(f"Error listing models: {e}")


def check_model_metrics(run_id: str):
    """Проверить метрики модели."""
    client = mlflow.tracking.MlflowClient()
    
    try:
        run = client.get_run(run_id)
        print(f"\n📊 Metrics for run {run_id}:")
        for metric_name, metric_value in run.data.metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
    except Exception as e:
        print(f"Error getting run metrics: {e}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Register model in MLflow Model Registry")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID to register")
    parser.add_argument("--model-name", type=str, default="telco_churn_model", help="Model name in registry")
    parser.add_argument("--experiment", type=str, default="telco_churn", help="Experiment name")
    parser.add_argument("--list", action="store_true", help="List all registered models")
    parser.add_argument("--check-metrics", type=str, default=None, help="Check metrics for a run")
    parser.add_argument("--auto", action="store_true", help="Auto-register best model")
    
    args = parser.parse_args()
    
    # Установить tracking URI перед созданием клиента
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    if args.list:
        list_registered_models()
    elif args.check_metrics:
        check_model_metrics(args.check_metrics)
    elif args.auto:
        # Автоматическая регистрация лучшей модели
        print("🔍 Поиск лучшей модели для автоматической регистрации...")
        
        # Пытаемся прочитать run_id из файла
        if os.path.exists("models/best_run_id.txt"):
            with open("models/best_run_id.txt", "r") as f:
                run_id = f.read().strip()
            print(f"✓ Найден run_id из файла: {run_id}")
        else:
            # Ищем лучший run по ROC-AUC
            best_run = get_best_run(args.experiment)
            if not best_run:
                print("❌ Не удалось найти лучшую модель")
                sys.exit(1)
            run_id = best_run.info.run_id
        
        success = register_model_in_registry(
            run_id=run_id,
            model_name=args.model_name,
            experiment_name=args.experiment
        )
        
        if success:
            print("\n" + "="*80)
            print("✅ МОДЕЛЬ УСПЕШНО ЗАРЕГИСТРИРОВАНА!")
            print("="*80)
        else:
            sys.exit(1)
    else:
        success = register_model_in_registry(
            run_id=args.run_id,
            model_name=args.model_name,
            experiment_name=args.experiment
        )


if __name__ == "__main__":
    main()