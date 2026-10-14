import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load comparison data
df = pd.read_csv('final_model_comparison.csv')

# 2. Select numeric metrics to plot in matrix
metrics = ['rmse', 'mae', 'mape', 'future_pred_7_days', 'future_pred_14_days']
matrix_data = df.set_index('model_name')[metrics]

# Rename columns for presentation
matrix_data.columns = ['RMSE', 'MAE', 'MAPE (%)', '7-Day Pred', '14-Day Pred']

# 3. Render Heatmap Matrix
plt.figure(figsize=(10, 4))
sns.heatmap(
    matrix_data, 
    annot=True, 
    fmt=".4f", 
    cmap="YlGnBu", 
    cbar=True, 
    linewidths=1.5,
    linecolor='white',
    annot_kws={"size": 11, "weight": "bold"}
)

plt.title("Model Performance & Forecast Heatmap Matrix", fontsize=14, pad=15, weight='bold')
plt.ylabel("Model Architecture", fontsize=12, labelpad=10)
plt.tight_layout()

# Save image
plt.savefig("model_performance_heatmap_matrix.png", dpi=300)
print("Saved model_performance_heatmap_matrix.png successfully!")