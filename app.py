import uvicorn

from src.api import PredictApi, app

d = PredictApi()

uvicorn.run(app, host="0.0.0.0", port=8000)
