import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. Define File Paths
    paths = {
        "LSTM": {
            "metrics": "metrics.json",
            "predictions": "predictions.csv"
        },
        "ARIMA": {
            "metrics": os.path.join("Team_submissions", "ARIMA_Project", "metrics.json"),
            "predictions": os.path.join("Team_submissions", "ARIMA_Project", "predictions.csv")
        },
        "XGBoost": {
            "metrics": os.path.join("Team_submissions", "XGBoost_Project", "metrics.json"),
            "predictions": os.path.join("Team_submissions", "XGBoost_Project", "predictions.csv")
        }
    }

    # 2. Load Metrics and Build Summary Table
    metrics_list = []
    for model_name, files in paths.items():
        with open(files["metrics"], "r") as f:
            data = json.load(f)
            # Ensure model_name field is present
            if "model_name" not in data:
                data["model_name"] = model_name
            metrics_list.append({
                "model_name": data.get("model_name", model_name),
                "rmse": round(float(data["rmse"]), 4),
                "mae": round(float(data["mae"]), 4),
                "mape": round(float(data["mape"]), 4),
                "future_pred_7_days": round(float(data["future_pred_7_days"]), 4),
                "future_pred_14_days": round(float(data["future_pred_14_days"]), 4)
            })

    summary_df = pd.DataFrame(metrics_list)
    
    # Save summary table to final_model_comparison.csv
    summary_csv_path = "final_model_comparison.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print("=" * 65)
    print("MODEL PERFORMANCE COMPARISON SUMMARY")
    print("=" * 65)
    print(summary_df.to_string(index=False))
    print("=" * 65)
    print(f"\nSaved summary table to '{summary_csv_path}' successfully.\n")

    # 3. Load Predictions for Master Plot
    lstm_preds = pd.read_csv(paths["LSTM"]["predictions"])
    arima_preds = pd.read_csv(paths["ARIMA"]["predictions"])
    xgb_preds = pd.read_csv(paths["XGBoost"]["predictions"])

    # Align predictions on common test length (last 32 test weeks)
    num_points = 32
    actual_vals = lstm_preds["y_true"].values[-num_points:]
    lstm_y_pred = lstm_preds["y_pred"].values[-num_points:]
    arima_y_pred = arima_preds["y_pred"].values[-num_points:]
    xgb_y_pred = xgb_preds["y_pred"].values[-num_points:]

    week_indices = np.arange(1, num_points + 1)

    # 4. Generate Master Plot
    plt.figure(figsize=(12, 6))

    # Plot Actual vs Predictions
    plt.plot(week_indices, actual_vals, label="Actual (y_true)", color="black", linewidth=2.5, marker="o", markersize=6)
    plt.plot(week_indices, lstm_y_pred, label="LSTM", color="blue", linewidth=2, linestyle="--", marker="s", markersize=5)
    plt.plot(week_indices, arima_y_pred, label="ARIMA", color="green", linewidth=2, linestyle="-.", marker="^", markersize=5)
    plt.plot(week_indices, xgb_y_pred, label="XGBoost", color="orange", linewidth=2, linestyle=":", marker="D", markersize=5)

    plt.title("Master Model Comparison: Actual vs Predicted ILI Percentage", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Week Index (Test Period)", fontsize=12, fontweight="bold")
    plt.ylabel("ILI Percentage (%)", fontsize=12, fontweight="bold")
    plt.xticks(np.arange(1, num_points + 1, 2))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11, loc="upper left", frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
    plt.tight_layout()

    master_plot_path = "master_model_comparison.png"
    plt.savefig(master_plot_path, dpi=300)
    plt.close()
    print(f"Saved master comparison plot to '{master_plot_path}' successfully.")

if __name__ == "__main__":
    main()
