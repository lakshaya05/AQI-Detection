import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import lightgbm as lgb
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

# ------------------- DATA LOADING AND PREPROCESSING -------------------
def load_data(file_path="combined_preprocessed.csv"):
    """Load and preprocess Chennai air quality data"""
    # Load data
    df = pd.read_csv(file_path, parse_dates=["Timestamp"])
    df["year"] = df["Timestamp"].dt.year
    df["month"] = df["Timestamp"].dt.month
    
    # Extract features and target
    features = ['PM10', 'NO2', 'Wind Speed', 'Humidity', 'month']
    X = df[features].values
    y = df['PM2.5'].values
    
    # Split train/test by year
    train_mask = df['year'] == 2022
    test_mask = df['year'] == 2023
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_mask])
    X_test = scaler.transform(X[test_mask])
    
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    # Prepare time series data for ARIMA/LSTM
    ts_train = pd.Series(y_train, index=df[train_mask]['Timestamp'])
    ts_test = pd.Series(y_test, index=df[test_mask]['Timestamp'])
    
    return X_train, X_test, y_train, y_test, ts_train, ts_test, df[train_mask], df[test_mask]

# ------------------- MODEL IMPLEMENTATIONS -------------------

def train_arima(ts_train, order=(5,1,0)):
    """ARIMA model for time series prediction"""
    start_time = time.time()
    
    # Fit model
    model = ARIMA(ts_train, order=order)
    model_fit = model.fit()
    
    train_time = time.time() - start_time
    return model_fit, train_time

def train_random_forest(X_train, y_train):
    """Random Forest regression model"""
    start_time = time.time()
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=4,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    return model, train_time

def train_xgboost(X_train, y_train):
    """XGBoost regression model"""
    start_time = time.time()
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    return model, train_time

def train_lightgbm(X_train, y_train, X_test, y_test):
    """LightGBM regression model (optimized for Chennai data)"""
    start_time = time.time()
    
    model = lgb.LGBMRegressor(
        num_leaves=31,
        max_depth=5,
        learning_rate=0.05,
        n_estimators=200,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='rmse',
        callbacks=[
    lgb.early_stopping(stopping_rounds=50),
    lgb.log_evaluation(0)  # Set to 0 for silent, or 10 for logging every 10 rounds
]
    )
    
    train_time = time.time() - start_time
    return model, train_time

def train_lstm(X_train, y_train):
    """LSTM deep learning model"""
    start_time = time.time()
    
    # Reshape data for LSTM [samples, time steps, features]
    X_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    
    # Build model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(1, X_train.shape[1])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Train
    model.fit(
        X_reshaped, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    train_time = time.time() - start_time
    return model, train_time

# ------------------- EVALUATION FUNCTION -------------------

def evaluate_pollutant(target='NO2', include_hybrid=True):
    """Chennai-optimized evaluation for any pollutant"""
    # Load and preprocess data
    df = pd.read_csv("combined_preprocessed.csv", parse_dates=["Timestamp"])

    df["month"] = df["Timestamp"].dt.month
    df["is_monsoon"] = df["month"].isin([6, 7, 8, 9]).astype(int)

    


    # Common features for all pollutants
    base_features = ['Wind Speed', 'Humidity', 'month', 'is_monsoon', 'Bhogi', 'Diwali']
    
    # Target-specific features
    if target == 'PM2.5':
        features = base_features + ['PM10', 'NO2']
    elif target == 'PM10':
        features = base_features + ['PM2.5', 'NO2']
    elif target == 'NO2':
        # 1. Remove correlated features causing overfitting
        base_features = ['Wind Speed', 'Humidity', 'month', 'is_monsoon']
        features = base_features + ['PM2.5', 'PM10']
        
        # 2. Add traffic-related features
        if 'Traffic_Index' not in df.columns:
            df['Traffic_Index'] = df['Bhogi'] + df['Diwali']  # Festival 
    else:
        features = base_features

    # Prepare data
    X = df[features].values
    y = df[target].values
    if 'year' not in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['year'] = df['Timestamp'].dt.year
    train_mask = df['year'] == 2022
    test_mask = df['year'] == 2023
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_mask])
    X_test = scaler.transform(X[test_mask])
    
    # Train models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=5),
        'LightGBM': lgb.LGBMRegressor(num_leaves=31, max_depth=5),
        'Hybrid (Yours)': None  # Placeholder for your metrics
    }
    
    # Your hybrid model metrics (replace with actual values)
    hybrid_metrics = {
        'PM2.5': {'RMSE': 6.98, 'MAE': 4.38, 'R²': 0.85, 'Train Time': 2700},
        'PM10': {'RMSE': 1.00, 'MAE': 0.80, 'R²': 0.78, 'Train Time': 2900},
        'NO2': {'RMSE': 8.21, 'MAE': 6.54, 'R²': 0.72, 'Train Time': 3100}
    }
    
    results = {}
    
    # Evaluate standard models
    for name, model in models.items():
        if model is not None:  # Skip hybrid placeholder
            start_time = time.time()
            model.fit(X_train, y[train_mask])
            train_time = time.time() - start_time
            
            preds = model.predict(X_test)
            
            results[name] = {
                'RMSE': np.sqrt(mean_squared_error(y[test_mask], preds)),
                'MAE': mean_absolute_error(y[test_mask], preds),
                'R²': r2_score(y[test_mask], preds),
                'Train Time': train_time
            }
    
    # Add hybrid model results
    if include_hybrid:
        try:
            hybrid_df = pd.read_csv("hybrid_evaluation_report.csv")  # replace with your actual path

        # Inside your evaluation loop for each pollutant (e.g., target = "PM2.5", "NO2", etc.)
            pollutant_row = hybrid_df[(hybrid_df['Model'] == 'Hybrid (AE+GAT+RL+LightGBM)') &  (hybrid_df['Pollutant'] == target)]

            if not pollutant_row.empty:
                row = pollutant_row.iloc[0]
                results['Hybrid (Yours)'] = {
                    'RMSE': row['RMSE'],
                    'MAE': row['MAE'],
                    'R²': row['R2'],  # Notice: 'R2' not 'R²' in the CSV
                    'Train Time': None  # Fill in if available in future
                }
        except Exception as e:
            print(f"[Warning] Could not load hybrid model metrics from CSV for {target}: {e}")

    
    return pd.DataFrame(results).T

# Generate comparison reports
pm25_report = evaluate_pollutant('PM2.5')
pm10_report = evaluate_pollutant('PM10')
no2_report = evaluate_pollutant('NO2')

# Save reports
pm25_report.to_csv(f"chennai_pm25_report_{pd.Timestamp.now().strftime('%Y%m%d')}.csv")
pm10_report.to_csv(f"chennai_pm10_report_{pd.Timestamp.now().strftime('%Y%m%d')}.csv") 
no2_report.to_csv(f"chennai_no2_report_{pd.Timestamp.now().strftime('%Y%m%d')}.csv")


def visualize_results(results_df):
    """Create comparison visualizations from properly formatted DataFrame"""
    plt.figure(figsize=(16, 10))
    
    # 1. RMSE Comparison
    plt.subplot(2, 2, 1)
    results_df['RMSE'].sort_values().plot(kind='barh', color='steelblue')
    plt.title('RMSE Comparison (Lower is Better)')
    plt.xlabel('μg/m³')
    
    # 2. MAE Comparison
    plt.subplot(2, 2, 2)
    results_df['MAE'].sort_values().plot(kind='barh', color='lightgreen')
    plt.title('MAE Comparison (Lower is Better)')
    plt.xlabel('μg/m³')
    
    # 3. R² Comparison
    plt.subplot(2, 2, 3)
    results_df['R²'].sort_values().plot(kind='barh', color='salmon')
    plt.title('R² Comparison (Higher is Better)')
    plt.xlabel('Score')
    


# ------------------- MAIN EXECUTION -------------------

if __name__ == "__main__":
    # Load data
    X_train, X_test, y_train, y_test, ts_train, ts_test, df_train, df_test = load_data()
    
    print("Training models for Chennai PM2.5 prediction...")
    
    # Train each model
    arima_model, arima_time = train_arima(ts_train)
    rf_model, rf_time = train_random_forest(X_train, y_train)
    xgb_model, xgb_time = train_xgboost(X_train, y_train)
    lgbm_model, lgbm_time = train_lightgbm(X_train, y_train, X_test, y_test)
    lstm_model, lstm_time = train_lstm(X_train, y_train)
    
    # Store models and training times
    models = {
        'ARIMA': arima_model,
        'Random Forest': rf_model,
        'XGBoost': xgb_model,
        'LightGBM': lgbm_model,
        'LSTM': lstm_model
    }
    
    models_training_time = {
        'ARIMA': arima_time,
        'Random Forest': rf_time,
        'XGBoost': xgb_time,
        'LightGBM': lgbm_time,
        'LSTM': lstm_time
    }
    
    # Add your hybrid model results
    # models['Hybrid (AE+GAT+RL+LightGBM)'] = your_hybrid_model  # Add your model here
    # models_training_time['Hybrid (AE+GAT+RL+LightGBM)'] = your_training_time  # Add your time here
    
    # Evaluate all models
    

# Helper function to print clean tables
def print_report(df, title):
    df['Model'] = df.index  # Assuming the index of df corresponds to model names
    # Reorder columns to have 'Model' first
    df = df[['Model', 'RMSE', 'MAE', 'R²', 'Train Time']]
    print(f"\n{title}")
    print("=" * 80)
    print(tabulate(df.round(2), headers='keys', tablefmt='grid', showindex=False))
    print("=" * 80)

# Helper function to plot bar graph
def visualize_results(report_df, title="Model Comparison"):
    # Create the plot
    ax = report_df[['RMSE', 'MAE', 'R²']].plot(kind='bar', figsize=(10, 6), rot=0)
    plt.title(title)
    plt.xlabel("Models")
    plt.ylabel("Score")
    plt.legend(loc='upper right')
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9)
    
    plt.tight_layout()
    plt.show()  # <- Display the plot
    return report_df

# Evaluate all 3 pollutants
pm25_report = evaluate_pollutant('PM2.5')
pm10_report = evaluate_pollutant('PM10')
no2_report = evaluate_pollutant('NO2')
# Print all 3 reports
print_report(pm25_report, "Chennai PM2.5 Prediction Model Comparison")
print_report(pm10_report, "Chennai PM10 Prediction Model Comparison")
print_report(no2_report, "Chennai NO2 Prediction Model Comparison")

# Visualize all 3 results
visualize_results(pm25_report, title="PM2.5 Model Performance")
visualize_results(pm10_report, title="PM10 Model Performance")
visualize_results(no2_report, title="NO2 Model Performance")



    # Save all reports with timestamp
'''current_date = pd.Timestamp.now().strftime('%Y%m%d')
    pm25_report.to_csv(f"chennai_pm25_report_{current_date}.csv")
    pm10_report.to_csv(f"chennai_pm10_report_{current_date}.csv") 
    no2_report.to_csv(f"chennai_no2_report_{current_date}.csv")'''