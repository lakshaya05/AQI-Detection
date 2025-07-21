import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import DQN
from auto_model import EnhancedAutoencoder
from auto_model import weighted_mse_loss

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
def prepare_features(current_data, historical_data, model, scaler):
    """Create time-aware features with rolling statistics"""
    # Combine last 7 days of data
    window = pd.concat([historical_data, current_data]).tail(7)
    
    # Calculate rolling features
    features = {
        'current_pm25': window['PM2.5'].iloc[-1],
        'mean_3day': window['PM2.5'].rolling(3).mean().iloc[-1],
        'mean_7day': window['PM2.5'].mean(),
        'month': current_data['month'].iloc[0],
        'is_monsoon': current_data['is_monsoon'].iloc[0],
        'wind_speed': window['Wind Speed'].iloc[-1],
        'humidity': window['Humidity'].iloc[-1],
        'bhogi': current_data['Bhogi'].iloc[0],
        'diwali': current_data['Diwali'].iloc[0]
    }
    
    # Convert to array in correct order
    return np.array([features[k] for k in model.booster_.feature_name()]).reshape(1, -1)

def forecast_future(model, last_known_data, ae_model, scaler, n_days=30):
    # Verify model expectations
    print(f"Model expects {model.n_features_in_} features")
    
    forecast_dates = pd.date_range(
        start=last_known_data['Timestamp'].max() + pd.Timedelta(days=1),
        periods=n_days
    )
    
    forecast_df = pd.DataFrame({
        'Timestamp': forecast_dates,
        'month': forecast_dates.month,
        'is_monsoon': [1 if 6 <= m <= 12 else 0 for m in forecast_dates.month],
        'Bhogi': (forecast_dates == pd.Timestamp('2024-01-14')).astype(int),
        'Diwali': (forecast_dates == pd.Timestamp('2024-11-01')).astype(int)
    })
    
    # Initialize with last known values
    last_values = last_known_data.iloc[-1]
    forecast_df['PM2.5'] = last_values['PM2.5']
    forecast_df['PM10'] = last_values['PM10']
    forecast_df['NO2'] = last_values['NO2']
    forecast_df['Wind Speed'] = last_values['Wind Speed']
    forecast_df['Humidity'] = last_values['Humidity']
    
    predictions = []
    for i in range(n_days):
        try:
            # Prepare exactly 9 features
            X = np.array([
                forecast_df.at[i, 'PM2.5'],
                forecast_df.at[i, 'PM10'],
                forecast_df.at[i, 'NO2'],
                forecast_df.at[i, 'Wind Speed'],
                forecast_df.at[i, 'Humidity'],
                forecast_df.at[i, 'month'],
                forecast_df.at[i, 'is_monsoon'],
                forecast_df.at[i, 'Bhogi'],
                forecast_df.at[i, 'Diwali']
            ]).reshape(1, -1)
            
            pred = model.predict(X)[0]
            predictions.append(pred)
            
            # Update PM2.5 for next prediction
            if i < n_days - 1:
                forecast_df.at[i+1, 'PM2.5'] = pred
                
        except Exception as e:
            print(f"Day {i+1} failed: {str(e)}")
            predictions.append(np.nan)
    
    forecast_df['PM2.5_predicted'] = predictions
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
    
    rl_model = model = DQN(
    "MlpPolicy", 
    env,
    learning_rate=0.0003,  # Increased from 0.0001
    buffer_size=100000,    # Larger replay buffer
    learning_starts=10000, # More exploration before learning
    batch_size=128,        # Increased batch size
    tau=0.005,             # For soft target update
    gamma=0.99,            # Discount factor
    train_freq=4,          # Update every 4 steps
    gradient_steps=1,
    target_update_interval=500,
    exploration_fraction=0.2,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,  # Lower final exploration
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
    
    # LightGBM Training
    X = np.hstack([
        ae_recon_loss.reshape(-1,1),
        df[['PM10', 'NO2', 'Wind Speed', 'Humidity', 'month']].values,
        df[['Bhogi', 'Diwali', 'is_monsoon']].values
    ])
    y = pm25_scaler.inverse_transform(df[['PM2.5_scaled']]).ravel()
    
    lgbm = LGBMRegressor(
        num_leaves=31,
        max_depth=3,
        reg_alpha=0.05,
        reg_lambda=0.05,
        min_data_in_leaf=12,
        early_stopping_rounds=50,
        colsample_bytree=0.7,
        subsample=0.8,
        subsample_freq=5,
        learning_rate=0.03,
        n_estimators=200,
        random_state=42,
        verbose=10
    )

    lgbm.fit(
        X[train_idx], 
        y[train_idx],
        eval_set=[(X[test_idx], y[test_idx])],
        eval_metric='rmse'
    )
    
try:
    #  Generate future predictions
    features = ['PM2.5','PM10', 'NO2', 'Wind Speed', 'Humidity']
    last_known = df[df['year'] == 2023].tail(30)
    future_forecast = forecast_future(
        model=lgbm,
        last_known_data=last_known,
        ae_model=ae_model,
        scaler=feature_scaler,
        n_days=60
    )
     

    print("\nSuccessful Forecast:")
    print(future_forecast[['Timestamp', 'PM2.5_predicted']].head(10))
    
    # After forecast generation
    
    #print(future_forecast[['Timestamp', 'PM2.5_predicted']].tail(5))
    print("Last historical date:", df['Timestamp'].max())
    print("Forecast starts:", future_forecast['Timestamp'].min())

    # Visualization
    plt.figure(figsize=(16, 8))
    
    # Historical data
    df_filtered = df[df['year'].isin([2022, 2023])]
    X_filtered = X[df_filtered.index]
    y_filtered = y[df_filtered.index]
    predicted_filtered = lgbm.predict(X_filtered)
    
    plt.plot(df_filtered['Timestamp'], y_filtered, label='Actual PM2.5', color='blue')
    plt.plot(df_filtered['Timestamp'], predicted_filtered, label='Predicted', color='orange')
    
    
    
    plt.title("Chennai Air Quality Monitoring & Forecasting (2022-2024)")
    plt.legend()
    plt.xlabel("Timestamp")
    plt.ylabel("PM2.5 (μg/m³)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()




# Evaluate in-sample predictions
    rmse = np.sqrt(mean_squared_error(y_filtered, predicted_filtered))
    mae = mean_absolute_error(y_filtered, predicted_filtered)
    r2 = r2_score(y_filtered, predicted_filtered)

    # Create a DataFrame for export
    pm25_hybrid_results = pd.DataFrame({
        'Model': ['Hybrid (AE+GAT+RL+LightGBM)'],
        'Pollutant': ['PM2.5'],
        'RMSE': [rmse],
        'MAE': [mae],
        'R2': [r2]
    })

    # Print for quick reference
    print("\nHybrid PM2.5 Evaluation Metrics:")
    print(pm25_hybrid_results.round(4))

    # Save as CSV (to merge later)
    pm25_hybrid_results.to_csv("hybrid_pm25_metrics.csv", index=False)




except Exception as e:
    print(f"Forecasting failed: {str(e)}")