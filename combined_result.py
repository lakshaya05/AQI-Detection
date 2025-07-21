import pandas as pd

# Define mapping of pollutant name and respective file
pollutant_files = {
    "PM2.5": "hybrid_pm25_metrics.csv",
    "PM10": "hybrid_pm10_metrics.csv",
    "NO2": "hybrid_no2_metrics.csv"
}

# Initialize an empty list to collect rows
rows = []

# Iterate and read each file
for pollutant, filename in pollutant_files.items():
    df = pd.read_csv(filename)
    
    # Assuming the CSV has single-row metrics with RMSE, MAE, R2 columns
    row = df.iloc[0]
    rows.append({
        "Model": "Hybrid (AE+GAT+RL+LightGBM)",
        "Pollutant": pollutant,
        "RMSE": row['RMSE'],
        "MAE": row['MAE'],
        "R2": row['R2']
    })

# Combine into a unified DataFrame
hybrid_metrics_df = pd.DataFrame(rows)

# Save to a unified CSV
hybrid_metrics_df.to_csv("hybrid_evaluation_report.csv", index=False)

print("Unified hybrid metrics saved as 'hybrid_evaluation_report.csv'")
