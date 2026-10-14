import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # 1. Load cleaned_ili_master.csv
    file_path = "cleaned_ili_master.csv"
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    # 2. Select specified columns
    selected_cols = ["temperature", "humidity", "pressure", "wind_speed", "target_ili_pct"]
    
    # Ensure numeric types and drop missing values if any
    subset_df = df[selected_cols].apply(pd.to_numeric, errors='coerce').dropna()

    # 3. Calculate Pearson correlation matrix
    corr_matrix = subset_df.corr(method='pearson')
    print("\nPearson Correlation Matrix:")
    print(corr_matrix)

    # 4. Render high-resolution heatmap
    plt.figure(figsize=(8, 6), dpi=300)
    
    # Custom formatting for publication aesthetics (IEEE paper)
    sns.set_theme(style="white")
    ax = sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".3f", 
        cmap="coolwarm", 
        vmin=-1, 
        vmax=1, 
        center=0,
        square=True, 
        linewidths=1, 
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 11, "weight": "bold"}
    )

    plt.title("Pearson Correlation Matrix of Weather & ILI Features", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()

    # 5. Save the image as correlation_matrix.png in project root
    output_path = "correlation_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved correlation heatmap to '{output_path}' successfully.")

if __name__ == "__main__":
    main()
