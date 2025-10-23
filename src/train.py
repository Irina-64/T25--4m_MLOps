import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import Dataset, DataLoader
import mlflow
from sklearn.metrics import roc_auc_score

# Инициализируем MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("churn_prediction_test")

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    return roc_auc_score(all_labels, all_preds)
# --- Гиперпараметры ---
SEQ_LEN = 335  # от 2024-01-01 до 2024-11-30
BATCH_SIZE = 64
EPOCHS = 50
HIDDEN_SIZE = 64

df = pd.read_csv("C:/Users/Lenovo/go/RailScan/T25--4m_MLOps/data/processed/processed.csv")

# Агрегируем по пользователю и дню
agg = df.groupby(["user_id", "day"]).agg(
    amount_sum=("amount", "sum"),
    amount_count=("amount", "count")
).reset_index()

# Генерируем полные последовательности
all_days = pd.DataFrame({'day': range(SEQ_LEN)})
user_ids = df["user_id"].unique()
data = []

for uid in user_ids:
    user_df = agg[agg["user_id"] == uid].merge(all_days, on="day", how="right").fillna(0)
    user_df = user_df.sort_values("day")
    data.append(user_df[["amount_sum", "amount_count"]].values)

X = np.stack(data)
labels = df.drop_duplicates("user_id").set_index("user_id").loc[user_ids]["churn"].values

# Нормализация
scaler = StandardScaler()
X_reshaped = X.reshape(-1, X.shape[-1])
X_scaled = scaler.fit_transform(X_reshaped).reshape(X.shape)

# Train-test разделение
X_train, X_val, y_train, y_val = train_test_split(X_scaled, labels, test_size=0.2, random_state=42)

# --- Torch Dataset ---
class ChurnDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(ChurnDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(ChurnDataset(X_val, y_val), batch_size=BATCH_SIZE)

# --- Модель ---
class LSTMChurnModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)  # hn shape: (2, batch, hidden_size)
        hn_cat = torch.cat((hn[0], hn[1]), dim=1)  # (batch, hidden_size * 2)
        return torch.sigmoid(self.fc(hn_cat))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = LSTMChurnModel(input_size=2, hidden_size=HIDDEN_SIZE).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# --- Тренировка с логированием в MLflow ---
with mlflow.start_run():
    # Логируем параметры
    mlflow.log_param("SEQ_LEN", SEQ_LEN)
    mlflow.log_param("BATCH_SIZE", BATCH_SIZE)
    mlflow.log_param("EPOCHS", EPOCHS)
    mlflow.log_param("HIDDEN_SIZE", HIDDEN_SIZE)
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("val_size", len(X_val))
    mlflow.log_param("divice", str(device))

    for epoch in range(EPOCHS):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # Оценка на валидации
        val_roc_auc = evaluate_model(model, val_loader, device)
        mlflow.log_metric("val_roc_auc", val_roc_auc, step=epoch)
        mlflow.log_metric("train_loss", loss.item(), step=epoch)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}, Val ROC_AUC: {val_roc_auc:.4f}")

    # --- Сохраняем модель и скейлер ---
    torch.save({
        "model": model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_
    }, "model.pt")
    mlflow.log_artifact("model.pt")
    mlflow.log_artifact("C:/Users/Lenovo/go/RailScan/T25--4m_MLOps/data/processed/processed.csv")
