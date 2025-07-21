import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
import lightgbm as lgb
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import DQN
from auto_model import EnhancedAutoencoder
from auto_model import weighted_mse_loss
from stable_baselines3 import SAC
# ---------------------
# 1. Enhanced Data Pipeline with Chennai Features
# ---------------------
def load_and_preprocess():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv("combined_preprocessed.csv", parse_dates=["Timestamp"])

    # Chennai temporal features
    df["year"] = df["Timestamp"].dt.year
    df["month"] = df["Timestamp"].dt.month
    df['is_monsoon'] = df['month'].between(6, 12).astype(int)
    
    ae_model = EnhancedAutoencoder().to(device)
    ae_model.load_state_dict(torch.load("best_autoencoder.pt"))
    ae_model.eval()
    pm25_scaler = StandardScaler()
    df['PM2.5_scaled'] = pm25_scaler.fit_transform(df[['PM2.5']])
    
    # Generate AE features dynamically
    feature_scaler = StandardScaler()
    ae_features = ['PM2.5','PM10', 'NO2', 'Wind Speed', 'Humidity']  # Exclude PM2.5
    
    # Fit ONLY on training data
    feature_scaler.fit(df.loc[df['year'] == 2022, ae_features])
    
    # Transform all data
    df[ae_features] = feature_scaler.transform(df[ae_features])
    
    with torch.no_grad():
        scaled_data = feature_scaler.transform(df[ae_features])
        inputs = torch.tensor(scaled_data, dtype=torch.float32).to(device)
        reconstructed, latent = ae_model(inputs)
        ae_latent = latent.cpu().numpy()
        ae_recon_loss = F.mse_loss(reconstructed, inputs, reduction='none').mean(dim=1).cpu().numpy()
        df['ae_latent'] = latent.cpu().numpy().tolist()

    # Festival dates (Example - verify actual dates)
    festival_dates = {
        'diwali': pd.Timestamp('2023-11-12'),
        'bhogi': pd.Timestamp('2023-01-14')
    }
    
    for fest, date in festival_dates.items():
        df[f'days_until_{fest}'] = (date - df['Timestamp']).dt.days.clip(lower=0)
    
    return df, ae_latent, ae_recon_loss, feature_scaler, ae_model, pm25_scaler

# ---------------------
# 2. Improved GAT Model with Chennai Relationships
# ---------------------
class GATAnomaly(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gat1 = GATConv(21, 128, heads=4)  # 21 input features
        self.gat2 = GATConv(512, 64)
        self.gat3 = GATConv(64, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        return self.gat3(x, edge_index).mean(dim=0)

def create_graphs(df):
    num_nodes = len(df)
    node_features = []
    edge_index = []

    # 1. Collect all node features
    for i in range(num_nodes):
        row = df.iloc[i]
        features = np.concatenate([
            row['ae_latent'],
            [
                row['PM2.5'], row['PM10'], row['NO2'],
                row['Wind Speed'], row['Humidity'],
                row['month'], row['Bhogi'], row['Diwali'],
                row['days_until_diwali']
            ]
        ])
        node_features.append(features)

    # 2. Efficient tensor conversion
    x = torch.tensor(np.array(node_features), dtype=torch.float32)

    # 3. Chennai temporal edges (connect to previous 3 days)
    for i in range(num_nodes):
        if i > 0: edge_index.append([i-1, i])
        if i > 1: edge_index.append([i-2, i])
        if i > 2: edge_index.append([i-3, i])
    
    edge_index = torch.tensor(edge_index).t().contiguous()
    edge_index = edge_index[:, (edge_index < num_nodes).all(dim=0)]

    # 4. Single graph for entire dataset
    full_graph = Data(x=x, edge_index=edge_index, y=torch.tensor(ae_recon_loss))
    return [full_graph]

# ---------------------
# 3. RL Threshold Adaptation with Chennai-Specific Rewards
# ---------------------
class AQIThresholdEnv(gym.Env):
    def __init__(self, data, scores):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.scores = scores
        self.action_space = gym.spaces.Discrete(5)  # [-2, -1, 0, +1, +2]
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(12,))
        self.threshold = np.percentile(scores, 95)
        self.initial_threshold = self.threshold

    def _get_state(self, idx):
        row = self.data.iloc[idx]
        return [
            row['PM2.5'], row['PM10'], row['NO2'],
            row['Wind Speed'], row['Humidity'],
            row['month']/12, 
            row['Bhogi'], row['Diwali'],
            row['days_until_diwali']/365,
            self.scores[idx]/10,
            row['is_monsoon'],
            row['days_until_bhogi']/365
        ]

    def step(self, action):
        delta = [-2, -1, 0, 1, 2][action]
        self.threshold = np.clip(self.threshold + delta, 50, 300)
        
        row = self.data.iloc[self.current_step]
        score = self.scores[self.current_step]
        is_anomaly = score > self.threshold

        # Base reward components
        detection_reward = 3.0 * score if is_anomaly else -0.1 * score
        threshold_penalty = -0.05 * abs(self.threshold - 150)  # Encourage thresholds around 150
        
        # Monsoon season adjustments
        if row['is_monsoon']:
            monsoon_bonus = 2.0 * (1 - abs(self.threshold - 100)/100)
        else:
            monsoon_bonus = 0
        
        # Festival bonuses
        if row['Bhogi'] or row['Diwali']:
            festival_bonus = 4.0 if (150 <= self.threshold <= 200) else -2.0
        else:
            festival_bonus = 0
        
        # Combine all reward components
        reward = detection_reward + threshold_penalty + monsoon_bonus + festival_bonus
        
        self.current_step += 1
        done = self.current_step >= len(self.data)
        next_state = self._get_state(self.current_step) if not done else np.zeros(12)
        return np.array(next_state, dtype=np.float32), reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.threshold = self.initial_threshold
        return np.array(self._get_state(0), dtype=np.float32), {}

# ---------------------
# 4. Future Prediction Functions
# ---------------------
def prepare_features(current_data, historical_data, ae_model, scaler):
    """Prepare features for forecasting with all required features"""
    # Combine historical and current data
    combined = pd.concat([historical_data, current_data])
    
    # Define required features (same order as scaler was trained on)
    required_features = ['PM2.5', 'PM10', 'NO2', 'Wind Speed', 'Humidity']
    
    # Handle missing features (especially PM2.5 in future predictions)
    for feat in required_features:
        if feat not in combined.columns:
            # Use mean value from training if feature is missing
            idx = required_features.index(feat)
            combined[feat] = scaler.mean_[idx]
    
    # Normalize features
    scaled_data = scaler.transform(combined[required_features])
    if len(scaled_data.shape) != 2:
        raise ValueError(f"Expected 2D array, got {scaled_data.shape}")
    
    try:
        # Generate AE latent vectors - process only current data point
        with torch.no_grad():
            current_scaled = scaled_data[-1:]  # Get only the current time step
            ae_input = torch.tensor(current_scaled, dtype=torch.float32)
            latent = ae_model.encoder(ae_input).numpy()
        
        # Extract temporal features for current data point
        temporal_features = current_data[['month', 'is_monsoon', 'Bhogi', 'Diwali']].values
        
        # Combine features
        return np.hstack([latent, temporal_features])
    except Exception as e:
        print(f"Feature preparation error: {str(e)}")
        return None

def forecast_no2_with_monthly_averages(df, lgbm_no2, feature_scaler, n_days=60):
    """
    Chennai-specific PM10 forecasting using monthly averages
    
    Args:
        df: Historical dataframe with all required features
        lgbm_pm10: Trained LightGBM model for PM10 prediction
        feature_scaler: Fitted StandardScaler for features
        n_days: Number of days to forecast
    
    Returns:
        DataFrame with PM10 predictions
    """
    # Get monthly averages from training data (Chennai-specific)
    training_data = df[df['year'] == 2022]
    monthly_means = training_data.groupby('month')[['PM10', 'NO2', 'Wind Speed', 'Humidity']].mean()
    
    # Get last known values
    last_known = df[df['year'] == 2023].tail(30)
    last_date = last_known['Timestamp'].max()
    
    # Create future dates
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_days)
    
    # Initialize forecast dataframe
    forecast_df = pd.DataFrame({
        'Timestamp': future_dates,
        'month': future_dates.month,
        'is_monsoon': [1 if 6 <= m <= 12 else 0 for m in future_dates.month],
        'Bhogi': (future_dates == pd.Timestamp('2024-01-14')).astype(int),
        'Diwali': (future_dates == pd.Timestamp('2024-11-01')).astype(int)
    })
    
    # Initialize with last known values
    for feat in ['PM10', 'NO2', 'Wind Speed', 'Humidity', 'PM2.5']:
        if feat in last_known.columns:
            forecast_df[feat] = last_known[feat].iloc[-1]
    
    # Store PM10 predictions
    no2_predictions = []
    
    # Calculate PM10 AE reconstruction loss (if used in your model)
    ae_recon_loss_no2 = df['ae_recon_loss_no2'].values if 'ae_recon_loss_no2' in df.columns else np.zeros(len(df))
    
    for i in range(n_days):
        # Get current row and create new row with updated features
        current_row = forecast_df.iloc[i:i+1].copy()
        new_row = current_row.copy()
        
        # Update features with Chennai monthly averages
        month = current_row['month'].values[0]
        for feat in ['PM10', 'NO2', 'Wind Speed', 'Humidity']:
            new_row[feat] = monthly_means.loc[month, feat]
        
        # Prepare feature vector for PM10 prediction (similar to PM2.5 features)
        X = np.hstack([
            ae_recon_loss_no2[-1:].reshape(-1,1),  # Use last known value
            new_row[['NO2', 'Wind Speed', 'Humidity', 'month']].values,
            new_row[['Bhogi', 'Diwali', 'is_monsoon']].values
        ])
        
        # Predict PM10
        no2_pred = lgbm_no2.predict(X)[0]
        no2_predictions.append(no2_pred)
        
        # Update for next prediction
        if i < n_days - 1:
            forecast_df.at[i+1, 'NO2'] = no2_pred
    
    # Add predictions to dataframe
    forecast_df['no2_predicted'] = no2_predictions
    return forecast_df


# ---------------------
# 5. Main Execution Pipeline
# ---------------------
if __name__ == "__main__":
    # Data loading
    df, ae_latent, ae_recon_loss, feature_scaler, ae_model, pm25_scaler = load_and_preprocess()

    train_idx = df[df['year'] == 2022].index
    test_idx = df[df['year'] == 2023].index
    
    # GAT Training
    graphs = create_graphs(df)
    full_graph = graphs[0]
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gat_model = GATAnomaly(in_channels=21).to(device)
    optimizer = torch.optim.Adam(gat_model.parameters(), lr=0.0001)

    # Train GAT
    for epoch in range(30):
        total_loss = 0
        batch = full_graph.to(device)
        optimizer.zero_grad()
        pred = gat_model(batch.x, batch.edge_index)
        loss = F.mse_loss(pred.view(-1, 1), batch.y.view(-1, 1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gat_model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")
    
    # RL Training
    val_data = df[df['year'] == 2022]
    env = AQIThresholdEnv(val_data, ae_recon_loss[val_data.index])
    
    rl_model = DQN(
    "MlpPolicy",
    env,
    learning_rate=0.0005,
    buffer_size=200000,
    batch_size=512,
    tau=0.001,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1000,
    exploration_fraction=0.4,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    verbose=1
)


    
    rl_model.learn(total_timesteps=50_000)
    
    # Dynamic Threshold Inference
    test_env = AQIThresholdEnv(df[df['year'] == 2023], ae_recon_loss[df['year'] == 2023])
    obs, _ = test_env.reset()
    thresholds_2023 = []
    
    for _ in range(len(test_env.data)):
        action, _ = rl_model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = test_env.step(action)
        thresholds_2023.append(test_env.threshold)
        if terminated or truncated:
            break
    
  
def train_no2_model(df, ae_latent, ae_recon_loss):
  
    # LightGBM Training
    X_no2 = np.hstack([
        ae_recon_loss.reshape(-1,1),
        df[['NO2', 'Wind Speed', 'Humidity', 'month']].values,
        df[['Bhogi', 'Diwali', 'is_monsoon']].values
    ])
    
    y_no2 = df['NO2'].values
    
    # Train-test split by year (Chennai-specific temporal split)
    train_idx = df[df['year'] == 2022].index
    test_idx = df[df['year'] == 2023].index
    
    # Initialize LightGBM with optimal parameters for PM10 (based on research)
    lgbm_no2 = LGBMRegressor(
        num_leaves=63,           # Increased for Chennai PM10 patterns
        max_depth=9,
        min_gain_to_split=0.001,              
        reg_alpha=0.005,          # Reduced regularization
        reg_lambda=0.005,
        min_child_samples=15,    
        learning_rate=0.02,      # Slower learning
        early_stopping_min_delta=0.001,  # Add this parameter
        n_estimators=1000
    )
    
    # Train with early stopping
    lgbm_no2.fit(
        X_no2[train_idx], 
        y_no2[train_idx],
        eval_set=[(X_no2[test_idx], y_no2[test_idx])],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(stopping_rounds=5, min_delta=0.001)]
    )

    y_pred = lgbm_no2.predict(X_no2[test_idx])
    rmse = np.sqrt(mean_squared_error(y_no2[test_idx], y_pred))
    mae = mean_absolute_error(y_no2[test_idx], y_pred)
    
    print(f"NO2 Model - RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    return lgbm_no2, X_no2
    
try:
    
    lgbm_no2, X_no2 = train_no2_model(df, ae_latent, ae_recon_loss)
    #  Generate future predictions
    features = ['PM2.5','PM10', 'NO2', 'Wind Speed', 'Humidity']
    last_known = df[df['year'] == 2023].tail(30)
    no2_forecast = forecast_no2_with_monthly_averages(
    df=df,
    lgbm_no2=lgbm_no2,
    feature_scaler=feature_scaler,
    n_days=60
    )


    # Debug: Print the forecast columns

    


    # Visualization

    

    df_filtered = df[df['year'].isin([2022, 2023])]

    X_no2_filtered = X_no2[df_filtered.index]  # Use your PM10 feature matrix
    y_no2_filtered = df_filtered['NO2'].values  # PM10 actual values
    predicted_no2_filtered = lgbm_no2.predict(X_no2_filtered)  # Using your PM10 model
    
    plt.figure(figsize=(16, 8))

# Historical data
    plt.plot(df_filtered['Timestamp'], y_no2_filtered, 
            label='Actual NO2', color='blue')
    plt.plot(df_filtered['Timestamp'], predicted_no2_filtered, 
            label='Predicted NO2', color='orange')

# You can keep the RL threshold if relevant for PM10, or remove this line
# plt.plot(df_filtered[df_filtered['year'] == 2023]['Timestamp'], 
#          thresholds_2023, label='RL Threshold', linestyle='--', color='red')

# PM10 forecast

    plt.title("Chennai NO2 Air Quality Monitoring & Forecasting (2022-2024)")
    plt.legend()
    plt.xlabel("Timestamp")
    plt.ylabel("NO2 (μg/m³)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('chennai_NO2_forecast.png')  # Save the figure
    plt.show()


    rmse = np.sqrt(mean_squared_error(y_no2_filtered, predicted_no2_filtered))
    mae = mean_absolute_error(y_no2_filtered,predicted_no2_filtered)
    r2 = r2_score(y_no2_filtered, predicted_no2_filtered)

    # Create a DataFrame for export
    no2_hybrid_results = pd.DataFrame({
        'Model': ['Hybrid (AE+GAT+RL+LightGBM)'],
        'Pollutant': ['NO2'],
        'RMSE': [rmse],
        'MAE': [mae],
        'R2': [r2]
    })

    print("\nHybrid NO2 Evaluation Metrics:")
    print(no2_hybrid_results.round(4))

    # Save as CSV (to merge later)
    no2_hybrid_results.to_csv("hybrid_no2_metrics.csv", index=False)


except Exception as e:
    print(f"Forecasting failed: {str(e)}") 