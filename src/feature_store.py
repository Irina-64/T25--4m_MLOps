# src/feature_store.py
import pandas as pd
import os

class SimpleFeatureStore:
    def __init__(self, csv_path="rl_state_features.csv"):
        self.csv_path = csv_path
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
            if "state_id" not in self.df.columns:
                self.df["state_id"] = range(len(self.df))
            self.df.set_index("state_id", inplace=True)
        else:
            # создаём пустой DataFrame
            self.df = pd.DataFrame()

    def log_features(self, state_id, features_dict):
        """
        Логируем фичи для конкретного state_id
        """
        row = pd.DataFrame([features_dict], index=[state_id])
        self.df = pd.concat([self.df, row])
        # optional: сохранять после каждой записи
        self.df.to_csv(self.csv_path)

    def get_features(self, state_id):
        """
        Получаем фичи для конкретного state_id
        """
        if state_id in self.df.index:
            return self.df.loc[state_id]
        else:
            # если нет, возвращаем нули
            return pd.Series([0]*self.df.shape[1], index=self.df.columns)
