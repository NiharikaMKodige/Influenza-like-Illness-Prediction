import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

class LSTMRegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, dropout=0.2):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Last time-step output
        out = self.dropout(out)
        out = self.fc(out)
        return out

def create_sequences(data, seq_length=4):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def main():
    set_seed(42)
    
    # 1. Load Data
    data_path = 'final_processed_ili_weekly.csv'
    df = pd.read_csv(data_path)
    ili_series = df['target_ili_pct'].values.reshape(-1, 1)
    
    # 2. Train/Test Split (First 132 weeks train, last 33 weeks test)
    train_size = 132
    test_size = 33
    
    train_data = ili_series[:train_size]
    test_data = ili_series[train_size:]
    
    print(f"Total samples: {len(ili_series)}")
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    
    # 3. Scaling (Fit ONLY on training data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    # 4. Sequence Creation (seq_length = 4)
    seq_length = 4
    
    X_train, y_train = create_sequences(train_scaled, seq_length=seq_length)
    
    # Prepend last seq_length values of train_scaled to test_scaled to get predictions for all 33 test weeks
    full_test_scaled = np.vstack([train_scaled[-seq_length:], test_scaled])
    X_test, y_test = create_sequences(full_test_scaled, seq_length=seq_length)
    
    # Reshape for PyTorch (batch_size, seq_len, num_features)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # 5. Model Architecture & Setup
    model = LSTMRegressor(input_size=1, hidden_size=32, dropout=0.2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    # 6. Training (80 epochs, batch size 16)
    epochs = 80
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    print("Training finished.")
    
    # 7. Evaluation & Inverse Transform
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test_t).numpy()
        
    y_true = scaler.inverse_transform(y_test).flatten()
    y_pred = scaler.inverse_transform(preds_scaled).flatten()
    
    # Calculate RMSE, MAE, MAPE
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    
    print(f"Test Metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}%")
    
    # 8. Future Forecasting (Recursive multi-step forecasting)
    full_dataset_scaled = np.vstack([train_scaled, test_scaled])
    last_seq = full_dataset_scaled[-seq_length:].copy().reshape(1, seq_length, 1)
    
    # t+1 (7 days ahead)
    with torch.no_grad():
        t1_input = torch.tensor(last_seq, dtype=torch.float32)
        pred_t1_scaled = model(t1_input).numpy()[0, 0]
        
    future_pred_7_days = float(scaler.inverse_transform([[pred_t1_scaled]])[0, 0])
    
    # t+2 (14 days ahead)
    seq_t2 = np.append(last_seq[:, 1:, :], [[[pred_t1_scaled]]], axis=1)
    with torch.no_grad():
        t2_input = torch.tensor(seq_t2, dtype=torch.float32)
        pred_t2_scaled = model(t2_input).numpy()[0, 0]
        
    future_pred_14_days = float(scaler.inverse_transform([[pred_t2_scaled]])[0, 0])
    
    print(f"Future Forecast (7 days / t+1): {future_pred_7_days:.4f}")
    print(f"Future Forecast (14 days / t+2): {future_pred_14_days:.4f}")
    
    # 9. Save Structured Outputs
    metrics_data = {
        "model_name": "LSTM",
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mape": round(mape, 4),
        "future_pred_7_days": round(future_pred_7_days, 4),
        "future_pred_14_days": round(future_pred_14_days, 4)
    }
    
    with open("metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=4)
    print("Saved metrics.json")
    
    predictions_df = pd.DataFrame({
        "week_index": np.arange(1, len(y_true) + 1),
        "y_true": np.round(y_true, 4),
        "y_pred": np.round(y_pred, 4)
    })
    predictions_df.to_csv("predictions.csv", index=False)
    print("Saved predictions.csv")
    
    # Plot Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.plot(predictions_df["week_index"], predictions_df["y_true"], label="Actual ILI %", color="#1f77b4", linewidth=2, marker='o')
    plt.plot(predictions_df["week_index"], predictions_df["y_pred"], label="Predicted ILI %", color="#ff7f0e", linewidth=2, linestyle='--', marker='s')
    plt.title("LSTM: Actual vs Predicted ILI Percentage", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Week Index (Test Period)", fontsize=12)
    plt.ylabel("ILI Percentage (%)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("lstm_actual_vs_predicted.png", dpi=300)
    plt.close()
    print("Saved lstm_actual_vs_predicted.png")

if __name__ == "__main__":
    main()
