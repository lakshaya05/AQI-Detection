import os
import pandas as pd
from glob import glob

# === Step 1: Load & Preprocess AQI Data === #
def load_and_combine_aqi(folder_path):
    files = glob(os.path.join(folder_path, "*.csv"))
    dataframes = []

    for f in files:
        try:
            df = pd.read_csv(f)
            if {'From Date', 'PM2.5 (ug/m3)', 'PM10 (ug/m3)', 'NO2 (ug/m3)'}.issubset(df.columns):
                df['Timestamp'] = pd.to_datetime(df['From Date'], errors='coerce')
                df = df[['Timestamp', 'PM2.5 (ug/m3)', 'PM10 (ug/m3)', 'NO2 (ug/m3)']]
                df.columns = ['Timestamp', 'PM2.5', 'PM10', 'NO2']
                df = df.dropna(subset=['Timestamp'])
                df = df[(df['Timestamp'] >= "2019-01-01") & (df['Timestamp'] <= "2023-03-31")]
                dataframes.append(df)
                print(f"✅ Loaded AQI file: {os.path.basename(f)} with {len(df)} rows")
            else:
                print(f"⚠️ Skipping AQI file: {os.path.basename(f)} (missing required columns)")
        except Exception as e:
            print(f"❌ Error reading AQI file {f}: {e}")

    if not dataframes:
        raise ValueError("No valid AQI data found.")
    
    combined = pd.concat(dataframes, ignore_index=True)
    daily_aqi = combined.set_index('Timestamp').resample('D').mean().reset_index()
    return daily_aqi

# === Step 2: Load & Combine Two Weather CSVs === #
def load_and_combine_weather(file1, file2):
    try:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        combined = pd.concat([df1, df2], ignore_index=True)

        if 'datetime' not in combined.columns or 'windspeed' not in combined.columns or 'humidity' not in combined.columns:
            raise ValueError("Missing required columns in weather files")

        combined['Timestamp'] = pd.to_datetime(combined['datetime'], errors='coerce')
        combined = combined[['Timestamp', 'windspeed', 'humidity']]
        combined.columns = ['Timestamp', 'Wind Speed', 'Humidity']
        combined = combined.dropna(subset=['Timestamp'])
        combined = combined[(combined['Timestamp'] >= "2019-01-01") & (combined['Timestamp'] <= "2023-03-31")]

        # Group by date and calculate mean
        combined['Date'] = combined['Timestamp'].dt.date
        daily_weather = combined.groupby('Date')[['Wind Speed', 'Humidity']].mean().reset_index()
        daily_weather.rename(columns={'Date': 'Timestamp'}, inplace=True)
        daily_weather['Timestamp'] = pd.to_datetime(daily_weather['Timestamp'])

        return daily_weather

    except Exception as e:
        raise RuntimeError(f"Weather processing failed: {e}")


# === Step 3: Merge === #
def merge_datasets(aqi_df, weather_df):
    merged = pd.merge(aqi_df, weather_df, on='Timestamp', how='inner')
    merged = merged.sort_values('Timestamp').reset_index(drop=True)
    return merged

# === Step 4: Run the Pipeline === #

aqi_folder = r"C:\Users\Dr. Swati Borse\Desktop\minor project fml\aqi_data"
weather_file1 = r"C:\Users\Dr. Swati Borse\Desktop\minor project fml\chennai 2021-07-01 to 2023-03-31.csv"
weather_file2 = r"C:\Users\Dr. Swati Borse\Desktop\minor project fml\Chennai, TN, India 2019-01-01 to 2021-06-30.csv"

print("🔄 Starting AQI and Weather Processing...")

aqi_df = load_and_combine_aqi(aqi_folder)
print(f"✅ Daily AQI rows: {len(aqi_df)}")

weather_df = load_and_combine_weather(weather_file1, weather_file2)
print(f"✅ Daily Weather rows: {len(weather_df)}")

final_df = merge_datasets(aqi_df, weather_df)
print(f"✅ Final merged dataset: {len(final_df)} rows")

# Save result
os.makedirs("processed_data", exist_ok=True)
final_df.to_csv("processed_data/final_daily_merged.csv", index=False)
print("✅ Saved merged dataset to 'processed_data/final_daily_merged.csv'")
