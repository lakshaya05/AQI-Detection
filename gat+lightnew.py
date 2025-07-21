import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import DQN

# ---------------------
# 1. Enhanced Data Pipeline with Chennai Features
# ---------------------
def load_and_preprocess():
    df = pd.read_csv("combined_preprocessed.csv", parse_dates=["Timestamp"])
    ae_latent = np.load("ae_latent_vectors.npy")
    ae_recon_loss = np.load("ae_reconstruction_loss.npy")

    # Chennai temporal features
    df["year"] = df["Timestamp"].dt.year
    df["month"] = df["Timestamp"].dt.month
    df['is_monsoon'] = df['month'].between(6, 12).astype(int)
    
    # Festival dates (Example - verify actual dates)
    festival_dates = {
        'diwali': pd.Timestamp('2023-11-12'),
        'bhogi': pd.Timestamp('2023-01-14')
    }
    
    for fest, date in festival_dates.items():
        df[f'days_until_{fest}'] = (date - df['Timestamp']).dt.days.clip(lower=0)
    
    # Normalization
    scaler = StandardScaler()
    features = ['PM2.5', 'PM10', 'NO2', 'Wind Speed', 'Humidity']
    df[features] = scaler.fit_transform(df[features])
    
    return df, ae_latent, ae_recon_loss, scaler

# ---------------------
# 2. Improved GAT Model with Chennai Relationships
# ---------------------
class GATAnomaly(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gat1 = GATConv(in_channels, 64, heads=4)
        self.gat2 = GATConv(64*4, 32)
        self.gat3 = GATConv(32, 1)

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        return self.gat3(x, edge_index).mean(dim=0)

def create_graphs(df, ae_latent):
    graphs = []
    for i in range(len(df)):
        row = df.iloc[i]
        latent = ae_latent[i]
        
        # Chennai-specific node features
        node_feats = np.array([
            np.concatenate([latent, [row[f]]]) 
            for f in ['PM2.5', 'PM10', 'NO2', 'Wind Speed', 'Humidity', 
                     'month', 'Bhogi', 'Diwali', 'days_until_diwali']
        ], dtype=np.float32)
        
        x = torch.tensor(node_feats, dtype=torch.float32)
        
        # Enhanced Chennai pollution relationships
        edge_index = torch.tensor([
            [0,3], [3,0], [0,1], [1,0],   # Core pollutants
            [2,4], [4,2], [5,0], [0,5],   # Monthly trends
            [6,0], [0,6], [7,0], [0,7],   # Festival impacts
            [8,0], [0,8], [1,4], [4,1]    # Temporal relationships
        ]).t().contiguous()
        
        graphs.append(Data(x=x, edge_index=edge_index, 
                         y=torch.tensor([ae_recon_loss[i]], dtype=torch.float32)))
    return graphs

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

        # Base reward with Chennai weighting
        reward = (2.5 * score * is_anomaly) - (0.2 * score * (not is_anomaly))
        
        # Monsoon adaptation (Jun-Dec)
        if row['is_monsoon']:
            if 80 <= self.threshold <= 120:
                reward += 4.0  # Strong incentive for optimal range
            elif self.threshold > 120:
                reward -= 0.5  # Reduced penalty
                
        # Festival handling (Diwali/Bhogi)
        if row['Bhogi'] or row['Diwali']:
            if 150 <= self.threshold <= 200:
                reward += 8.0 if is_anomaly else 3.0
            else:
                reward -= 2.0  # Reduced penalty

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
# 4. Optimized Training Pipeline
# ---------------------
if __name__ == "__main__":
    # Data loading
    df, ae_latent, ae_recon_loss, scaler = load_and_preprocess()
    
    # GAT Training
    graphs = create_graphs(df, ae_latent)
    train_idx = df[df['year'] == 2022].index
    train_loader = DataLoader([graphs[i] for i in train_idx], batch_size=32, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GATAnomaly(ae_latent.shape[1]+1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train with proper shape matching
    for epoch in range(30):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index)
            loss = F.mse_loss(pred.view(-1, 1), batch.y.view(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")
    
    # RL Training with extended exploration
    val_data = df[df['year'] == 2022]
    env = AQIThresholdEnv(val_data, ae_recon_loss[val_data.index])
    
    model = DQN(
        "MlpPolicy", env,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.2,
        exploration_fraction=0.6,
        target_update_interval=1000,
        verbose=1
    )
    model.learn(total_timesteps=50_000)
    
    # Dynamic Threshold Inference
    test_env = AQIThresholdEnv(df[df['year'] == 2023], ae_recon_loss[df['year'] == 2023])
    obs, _ = test_env.reset()
    thresholds_2023 = []
    
    for _ in range(len(test_data)):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)  # 5 values!
        thresholds_2023.append(test_env.threshold)
        if terminated or truncated:
            break
    
    # LightGBM Integration with Proper Scaling
    X = np.hstack([
        ae_recon_loss.reshape(-1,1),
        df[['PM10', 'NO2', 'Wind Speed', 'Humidity', 'month']]
    ])
    y = scaler.inverse_transform(df[['PM2.5']]).ravel()
    
    lgbm = LGBMRegressor(
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=200,
        min_child_samples=20
    )
    lgbm.fit(X[train_idx], y[train_idx])
    
    # Visualization with Thresholds
    plt.figure(figsize=(14,7))
    plt.plot(df['Timestamp'], y, label='Actual PM2.5')
    plt.plot(df['Timestamp'], lgbm.predict(X), label='Predicted')
    plt.plot(df['Timestamp'], thresholds, label='RL Threshold', linestyle='--')
    plt.title("Chennai Air Quality Monitoring")
    plt.legend()
    plt.show()
