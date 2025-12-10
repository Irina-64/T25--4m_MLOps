# Run the FastAPI model API (Docker)

This project includes a small FastAPI app at `src/api.py` that exposes `/predict`.

Build the Docker image:

```bash
docker build -t flight-delay-api:lab6 .
```

Run the container (maps port 8080):

```bash
docker run -p 8080:8080 flight-delay-api:lab6
```

Test the endpoint (example payload — adjust features to match your model):

```bash
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{"tenure": 12, "MonthlyCharges": 70.5, "gender": 1, "Contract_Month-to-month": 1}'
```

Notes:
- The API will try to load `models/model.joblib` if present, otherwise it picks the latest `.joblib` file in `models/`.
- Make sure the payload keys match the features used during training (or the model's `feature_names_in_`).
- If you use a virtual environment locally, ensure Docker has access to the model files in the repository.
