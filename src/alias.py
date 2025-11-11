from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = "ufc_winner_model"


versions = client.search_model_versions(f"name='{model_name}'")
latest = max(versions, key=lambda v: int(v.version))

client.set_registered_model_alias(model_name, "staging", latest.version)

print(f"Alias 'staging' присвоен версии {latest.version} модели {model_name}")
