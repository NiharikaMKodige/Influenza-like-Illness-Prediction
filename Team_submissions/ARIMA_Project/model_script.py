import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("final_processed_ili_weekly.csv")

# Target Variable
y = df["target_ili_pct"]
print("Total Rows:", len(df))

# -----------------------------
# Train Test Split (80:20)
# -----------------------------
train_size = 132

train = y.iloc[:132]
test = y.iloc[132:]

print("Training Weeks:", len(train))
print("Testing Weeks :", len(test))

# -----------------------------
# ADF Stationarity Test
# -----------------------------
print("\n==============================")
print("ADF Stationarity Test")
print("==============================")

adf_result = adfuller(train)

print("ADF Statistic :", adf_result[0])
print("P-value       :", adf_result[1])

if adf_result[1] < 0.05:
    print("Series is Stationary")
else:
    print("Series is NOT Stationary")

# -----------------------------
# Auto ARIMA Model
# -----------------------------
print("\nSearching Best ARIMA Model...\n")

model = auto_arima(
    train,
    seasonal=False,
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
    stepwise=True
)

print("\nBest Model:")
print(model.summary())

# -----------------------------
# Predict Test Set
# -----------------------------
predictions = model.predict(n_periods=len(test))

# -----------------------------
# Evaluation Metrics
# -----------------------------
rmse = np.sqrt(mean_squared_error(test, predictions))
mae = mean_absolute_error(test, predictions)
mape = np.mean(np.abs((test - predictions) / test)) * 100

print("\n==============================")
print("Evaluation Metrics")
print("==============================")

print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"MAPE : {mape:.2f}%")

# -----------------------------
# Future Forecast
# -----------------------------
future = model.predict(n_periods=2)

forecast_7 = future.iloc[0]
forecast_14 = future.iloc[1]

print("\nFuture Forecasts")
print("------------------------")
print("7-Day Forecast :", forecast_7)
print("14-Day Forecast:", forecast_14)

# -----------------------------
# Save Metrics JSON
# -----------------------------
metrics = {
    "model_name": "ARIMA",
    "rmse": float(rmse),
    "mae": float(mae),
    "mape": float(mape),
    "future_pred_7_days": float(forecast_7),
    "future_pred_14_days": float(forecast_14)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("metrics.json saved.")

# -----------------------------
# Save Predictions CSV
# -----------------------------
pred_df = pd.DataFrame({
    "week_index": np.arange(133, 165),   # 32 values
    "y_true": test.values,
    "y_pred": predictions
})

pred_df.to_csv("predictions.csv", index=False)

print("predictions.csv saved.")

# -----------------------------
# Plot Actual vs Predicted
# -----------------------------
plt.figure(figsize=(12,6))

# X-axis for the test period
weeks = range(133, 133 + len(test))

plt.plot(
    weeks,
    test.values,
    marker='o',
    linewidth=2,
    label='Actual'
)

plt.plot(
    weeks,
    np.array(predictions),
    marker='s',
    linewidth=2,
    label='Predicted'
)

plt.title("ARIMA: Actual vs Predicted")
plt.xlabel("Week Index")
plt.ylabel("ILI Percentage")

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=300)
plt.show()

# -----------------------------
# Future Forecast Plot
# -----------------------------
plt.figure(figsize=(8,5))

plt.plot(
    [1,2],
    [forecast_7, forecast_14],
    marker="o",
    linewidth=2
)

plt.xticks([1,2], ["7 Days","14 Days"])

plt.title("Future Forecast")
plt.ylabel("Predicted ILI %")

plt.grid(True)

plt.tight_layout()
plt.savefig("future_forecast.png", dpi=300)

plt.show()

print("\nAll files saved successfully.")