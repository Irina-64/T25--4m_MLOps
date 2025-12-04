# src\register_model.py
import mlflow
run_id = "5626118ec9a14daa83240604c61dc969"

result = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="durak_rl"
)

print("=== MODEL REGISTERED ===")
print("Name: durak_rl")
print("Version:", result.version)
