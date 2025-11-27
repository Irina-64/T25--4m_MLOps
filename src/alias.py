from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = "ufc_winner_model"

versions = client.search_model_versions(f"name='{model_name}'")

if not versions:
    raise ValueError(" Нет зарегистрированных моделей 'ufc_winner_model'")

latest = max(versions, key=lambda v: int(v.version))

client.set_registered_model_alias(model_name, "staging", latest.version)

print(f"Alias 'staging' → версия {latest.version}")
