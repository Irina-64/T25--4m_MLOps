import pandas as pd
from mlflow.pyfunc.model import PythonModel
from tensorflow import keras


class DelayTextModel(PythonModel):
    model = None
    vectorizer = None

    def load_context(self, context):
        self.model = keras.models.load_model(context.artifacts["model"])
        self.vectorizer = keras.models.load_model(context.artifacts["vectorizer"])

    def predict(self, context, model_input):  # type: ignore[override]
        """
        model_input:
          - pd.Series[str]
          - pd.DataFrame с одной колонкой
          - list[str]
        """

        if isinstance(model_input, pd.DataFrame):
            texts = model_input.iloc[:, 0].astype(str).values
        elif isinstance(model_input, pd.Series):
            texts = model_input.astype(str).values
        else:
            texts = pd.Series(model_input).astype(str).values

        x = self.vectorizer(texts)
        preds = self.model.predict(x)

        return preds
