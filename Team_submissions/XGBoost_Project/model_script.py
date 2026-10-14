import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json


df = pd.read_csv("final_processed_ili_weekly.csv")

# Quick check - print shape and column names
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

feature_cols = [
    "temperature", "humidity", "pressure", "wind_speed",
    "ili_cases", "total patients",
    "ili_pct_lag_1", "ili_pct_lag_2", "ili_pct_lag_3"
]
target_col = "target_ili_pct"

X = df[feature_cols]
y = df[target_col]

split_index = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print("Train size:", len(X_train))
print("Test size:", len(X_test))


model = XGBRegressor(
    n_estimators=200,
    max_depth=4,        
    learning_rate=0.05,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("First 5 predictions:", y_pred[:5])
print("First 5 actual values:", y_test.values[:5])


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test.values - y_pred) / y_test.values)) * 100

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.4f}%")

# Get the last row of the dataset (most recent week)
last_row = df.iloc[-1]

# --- Forecast t+1 (7 days ahead) ---
future_features_1 = pd.DataFrame([{
    "temperature": last_row["temperature"],
    "humidity": last_row["humidity"],
    "pressure": last_row["pressure"],
    "wind_speed": last_row["wind_speed"],
    "ili_cases": last_row["ili_cases"],
    "total patients": last_row["total patients"],
    "ili_pct_lag_1": last_row["target_ili_pct"],
    "ili_pct_lag_2": last_row["ili_pct_lag_1"],
    "ili_pct_lag_3": last_row["ili_pct_lag_2"],
}])

pred_t1 = model.predict(future_features_1)[0]

# --- Forecast t+2 (14 days ahead), using predicted t+1 as new lag_1 ---
future_features_2 = pd.DataFrame([{
    "temperature": last_row["temperature"],
    "humidity": last_row["humidity"],
    "pressure": last_row["pressure"],
    "wind_speed": last_row["wind_speed"],
    "ili_cases": last_row["ili_cases"],
    "total patients": last_row["total patients"],
    "ili_pct_lag_1": pred_t1,
    "ili_pct_lag_2": last_row["target_ili_pct"],
    "ili_pct_lag_3": last_row["ili_pct_lag_1"],
}])

pred_t2 = model.predict(future_features_2)[0]

print(f"7-day ahead forecast (t+1): {pred_t1:.4f}")
print(f"14-day ahead forecast (t+2): {pred_t2:.4f}")

# --- Save metrics.json ---
metrics = {
    "model_name": "XGBoost",
    "rmse": round(float(rmse), 4),
    "mae": round(float(mae), 4),
    "mape": round(float(mape), 4),
    "future_pred_7_days": round(float(pred_t1), 4),
    "future_pred_14_days": round(float(pred_t2), 4)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved metrics.json")

# --- Save predictions.csv ---
predictions_df = pd.DataFrame({
    "week_index": range(split_index, split_index + len(y_test)),
    "y_true": y_test.values,
    "y_pred": y_pred
})
predictions_df.to_csv("predictions.csv", index=False)

print("Saved predictions.csv")

# --- Save actual_vs_predicted.png ---
plt.figure(figsize=(10, 5))
plt.plot(predictions_df["week_index"], predictions_df["y_true"], label="Actual", marker="o")
plt.plot(predictions_df["week_index"], predictions_df["y_pred"], label="Predicted", marker="x")
plt.xlabel("Week Index")
plt.ylabel("ILI %")
plt.title("XGBoost: Actual vs Predicted (Test Set)")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.close()

print("Saved actual_vs_predicted.png")

# --- Save future_forecast.png ---
plt.figure(figsize=(8, 5))
weeks = ["t+1 (7 days)", "t+2 (14 days)"]
values = [pred_t1, pred_t2]
plt.plot(weeks, values, marker="o", color="red")
plt.ylabel("Forecasted ILI %")
plt.title("XGBoost: Future Forecast (7-day and 14-day)")
plt.tight_layout()
plt.savefig("future_forecast.png")
plt.close()

print("Saved future_forecast.png")