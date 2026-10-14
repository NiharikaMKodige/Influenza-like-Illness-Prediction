import pandas as pd
import numpy as np

def preprocess_master():
    # Read input dataset
    input_path = 'cleaned_ili_master.csv'
    output_path = 'final_processed_ili_weekly.csv'
    
    df = pd.read_csv(input_path)
    
    # Define aggregation columns and target data types
    num_cols = ['target_ili_pct', 'temperature', 'humidity', 'pressure', 'wind_speed', 'ili_cases', 'total patients']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Aggregate nationally by grouping by year and week
    agg_dict = {
        'target_ili_pct': 'mean',
        'temperature': 'mean',
        'humidity': 'mean',
        'pressure': 'mean',
        'wind_speed': 'mean',
        'ili_cases': 'sum',
        'total patients': 'sum'
    }
    
    df_aggregated = df.groupby(['year', 'week'], as_index=False).agg(agg_dict)
    
    # Sort chronologically by year and week
    df_sorted = df_aggregated.sort_values(by=['year', 'week']).reset_index(drop=True)
    
    # Handle missing values using forward fill followed by backward fill
    df_filled = df_sorted.ffill().bfill()
    
    # Create lag features based on target_ili_pct
    df_filled['ili_pct_lag_1'] = df_filled['target_ili_pct'].shift(1)
    df_filled['ili_pct_lag_2'] = df_filled['target_ili_pct'].shift(2)
    df_filled['ili_pct_lag_3'] = df_filled['target_ili_pct'].shift(3)
    
    # Drop remaining NaN rows (from the initial shift lags)
    df_final = df_filled.dropna().reset_index(drop=True)
    
    # Save the final processed dataset
    df_final.to_csv(output_path, index=False)
    print(f"Preprocessing complete. Saved processed dataset to '{output_path}'.")
    print(f"Final Dataset Shape: {df_final.shape}")
    print("\nFirst 5 rows:")
    print(df_final.head())

if __name__ == '__main__':
    preprocess_master()
