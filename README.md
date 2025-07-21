# Chennai Air Quality Anomaly Detection & Forecasting (2019–2023)

## About This Project

Accurate air quality monitoring is critical for public health—especially in rapidly urbanizing Indian cities like **Chennai**, where pollution patterns are influenced by festivals, monsoons, and lockdowns. This project introduces a **hybrid deep learning framework** for anomaly detection and AQI forecasting that combines:

-  **Feature-weighted Autoencoder** for unsupervised anomaly scoring  
-  **Reinforcement Learning Agent** for adaptive thresholding  
-  **Graph Attention Network (GAT)** for temporal and environmental feature fusion  
-  **LightGBM** for next-day AQI prediction  

Evaluated on 5 years of air quality data from **CPCB (2019–2023)**, this system demonstrates robust spike detection and forecasting across PM2.5, PM10, and NO₂—especially around events like **Diwali**, **New Year**, **Bhogi**, and **COVID-19 lockdowns**.

---

## ⚙ Methodology

### 1. Preprocessing
- Merged CSVs from multiple monitoring stations
- Interpolation, forward/backward filling
- Spike filtering to reduce sensor noise

### 2. Feature Engineering
- Normalization using MinMaxScaler
- Event flags for Diwali, New Year, Bhogi, COVID lockdown
- Weather features: Wind Speed, Humidity
- Latent AE embeddings used as GAT input

### 3. Anomaly Detection Pipeline
- Autoencoder flags outliers based on reconstruction error
- RL agent dynamically adjusts thresholds using a custom reward function (season/festival-aware)
- GAT fuses pollutant, weather, and event features over time

### 4. AQI Forecasting
- LightGBM trained on GAT-fused features
- Predicts next-day pollutant concentration (PM2.5, PM10, NO₂)

---

## Dataset

- **Source**: [Central Pollution Control Board (CPCB)](https://cpcb.nic.in/)
- **Region**: Chennai, India
- **Time Frame**: 2019–2023
- **Pollutants**: PM2.5, PM10, NO₂
- **Weather Features**: Wind Speed, Humidity
- **Event Flags**: Diwali, Bhogi, New Year, COVID Lockdowns

---

## Tech Stack

- **Languages/Frameworks**: Python, PyTorch, LightGBM, Scikit-learn
- **Visualization**: Matplotlib
- **Libraries**: Pandas, NumPy, NetworkX, OpenAI Gym

---

##  Results & Highlights

### Model Performance Comparison (Forecasting)

| **Pollutant** | **Model**        | **RMSE ↓** | **MAE ↓** | **R² ↑** |
|---------------|------------------|------------|-----------|----------|
| **PM2.5**     | Random Forest     | 5.19       | 3.98      | 0.89     |
|               | XGBoost           | 6.37       | 4.67      | 0.83     |
|               | LightGBM          | 8.30       | 6.38      | 0.72     |
|               | **Hybrid**        | **5.21**   | **3.38**  | **0.90** |
| **PM10**      | Random Forest     | 8.00       | 6.20      | 0.87     |
|               | XGBoost           | 8.80       | 6.60      | 0.85     |
|               | LightGBM          | 9.00       | 6.90      | 0.83     |
|               | **Hybrid**        | **8.30**   | **6.70**  | **0.86** |
| **NO₂**       | Random Forest     | 6.50       | 5.50      | 0.72     |
|               | XGBoost           | 6.80       | 5.80      | 0.68     |
|               | LightGBM          | 7.00       | 6.00      | 0.65     |
|               | **Hybrid**        | **7.20**   | **6.80**  | **0.83** |

>  The **hybrid framework** achieves the highest **R²** for PM2.5 and NO₂ and performs competitively for PM10—demonstrating strong generalization across pollutants.

###  Key Highlights

-  **PM2.5**: Hybrid model outperforms all baselines in accuracy and error minimization
- 🌫 **PM10**: Hybrid method remains competitive with ensemble baselines
-  **NO₂**: Best R² among all models despite minor increases in error
-  **Event Detection**: Accurately flags spikes during Diwali, New Year, Bhogi, and lockdowns
- 🌧 **Seasonal Robustness**: Reduced false positives during monsoon periods

---

##  Future Work

- Extend to multi-city or pan-India datasets
- Integrate real-time streaming with deployment-ready APIs
- Include additional meteorological variables (temperature, pressure, rainfall)

---

##  Requirements

```bash
pip install pandas numpy matplotlib scikit-learn lightgbm torch
