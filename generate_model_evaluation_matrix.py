import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. Read final_model_comparison.csv
    csv_path = "final_model_comparison.csv"
    print(f"Reading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 2. Rename and format columns cleanly
    # Target columns: Model, RMSE, MAE, MAPE, 7-Day Forecast, 14-Day Forecast
    column_mapping = {
        "model_name": "Model",
        "rmse": "RMSE",
        "mae": "MAE",
        "mape": "MAPE",
        "future_pred_7_days": "7-Day Forecast",
        "future_pred_14_days": "14-Day Forecast"
    }
    df = df.rename(columns=column_mapping)

    # Format numeric values to exactly 4 decimal places
    for col in ["RMSE", "MAE", "MAPE", "7-Day Forecast", "14-Day Forecast"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{float(x):.4f}")

    columns = ["Model", "RMSE", "MAE", "MAPE", "7-Day Forecast", "14-Day Forecast"]
    df = df[columns]

    print("\nFormatted Table Data:")
    print(df.to_string(index=False))

    # 3. Create matplotlib figure and render publication-quality table
    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=300)
    ax.axis("off")
    ax.axis("tight")

    # Colors
    header_bg = "#1f3c88"      # Slate / dark blue header
    header_fg = "#ffffff"      # White header text
    row_bg_even = "#f8f9fa"    # Light gray/blue
    row_bg_odd = "#ffffff"     # Pure white
    text_color = "#212529"     # Dark text

    # Render table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header cells and data cells
    for i, key in enumerate(table.get_celld().keys()):
        row, col = key
        cell = table[key]
        cell.set_edgecolor("#d0d7de")
        cell.set_linewidth(1.0)
        
        if row == 0:
            cell.set_facecolor(header_bg)
            cell.set_text_props(color=header_fg, weight="bold", fontsize=11)
        else:
            bg_color = row_bg_even if row % 2 == 0 else row_bg_odd
            cell.set_facecolor(bg_color)
            cell.set_text_props(color=text_color, fontsize=10.5)

    plt.title("Model Performance Evaluation Matrix", fontsize=14, fontweight="bold", pad=15, color="#1a252c")
    plt.tight_layout()

    # 4. Save table graphic as model_evaluation_matrix.png at 300 DPI
    output_path = "model_evaluation_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved publication-quality evaluation matrix to '{output_path}' successfully.")

if __name__ == "__main__":
    main()
