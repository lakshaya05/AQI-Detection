import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

# Load the dataset
df = pd.read_csv("processed_data/final_daily_merged.csv")
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Interpolate short gaps
for col in ['PM2.5', 'PM10', 'NO2']:
    df[col] = df[col].interpolate(method='linear', limit=3, limit_direction='forward')

# Seasonal decomposition for long gaps
for col in ['PM2.5', 'PM10', 'NO2']:
    series = df[col].dropna()
    decomposition = seasonal_decompose(series, period=365, model='additive', extrapolate_trend='freq')
    trend_filled = decomposition.trend.ffill().bfill()
    df.loc[trend_filled.index, f'{col}_filled'] = trend_filled

# Add festival flags
bhogi_dates = ['2019-01-14', '2020-01-14', '2021-01-14', '2022-01-14', '2023-01-14']
diwali_dates = ['2019-10-27', '2020-11-14', '2021-11-04', '2022-10-24']
df['Bhogi'] = df['Timestamp'].dt.strftime('%Y-%m-%d').isin(bhogi_dates).astype(int)
df['Diwali'] = df['Timestamp'].dt.strftime('%Y-%m-%d').isin(diwali_dates).astype(int)

# Feature engineering
for col in ['PM2.5', 'PM10', 'NO2']:
    df[f'{col}_lag1'] = df[col].shift(1)
    df[f'{col}_lag7'] = df[col].shift(7)
    df[f'{col}_roll7'] = df[col].rolling(window=7).mean()



# Handle missing values before IsolationForest
iso_cols = ['PM2.5', 'PM10', 'NO2', 'Wind Speed', 'Humidity']
df[iso_cols] = df[iso_cols].interpolate(limit_direction='forward').bfill()    

# IsolationForest to flag errors
iso = IsolationForest(contamination=0.05, random_state=42)
df['is_error'] = iso.fit_predict(df[['PM2.5', 'PM10', 'NO2', 'Wind Speed', 'Humidity']])

for col in ['PM2.5', 'PM10', 'NO2']:
    df[f'{col}_clean'] = np.where(df['is_error'] == -1, np.nan, df[col])
    df[f'{col}_clean'] = df[f'{col}_clean'].interpolate()

# Split data
train = df[df['Timestamp'] < '2023-01-01'].copy()
test = df[df['Timestamp'] >= '2023-01-01'].copy()

# Scaling
pollutant_scaler = RobustScaler()
weather_scaler = RobustScaler()

train_pollutants = pollutant_scaler.fit_transform(train[['PM2.5_clean', 'PM10_clean', 'NO2_clean']])
test_pollutants = pollutant_scaler.transform(test[['PM2.5_clean', 'PM10_clean', 'NO2_clean']])

train_weather = weather_scaler.fit_transform(train[['Wind Speed', 'Humidity']])
test_weather = weather_scaler.transform(test[['Wind Speed', 'Humidity']])

train_scaled = pd.DataFrame(train_pollutants, columns=['PM2.5_scaled', 'PM10_scaled', 'NO2_scaled'], index=train.index)
train_scaled[['WindSpeed_scaled', 'Humidity_scaled']] = train_weather

test_scaled = pd.DataFrame(test_pollutants, columns=['PM2.5_scaled', 'PM10_scaled', 'NO2_scaled'], index=test.index)
test_scaled[['WindSpeed_scaled', 'Humidity_scaled']] = test_weather

# Combine and save
# Combine train and test sets along with their scaled versions
combined_df = pd.concat([
    pd.concat([train, train_scaled], axis=1),
    pd.concat([test, test_scaled], axis=1)
], axis=0)

# Save the combined dataset
combined_df.to_csv("combined_preprocessed.csv", index=False)
