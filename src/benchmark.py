import os

# [CRITICAL] Completely hide the GPU from the system.
# This forces PyTorch to use the CPU, preventing crashes caused by 
# the incompatibility between the current PyTorch version and the RTX 5070 (sm_120).
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Import custom modules from src
from src.data_loader import get_vix_term_structure
from src.models import TEGNet_TermStructure
from src.physics_loss import ReactionDiffusionLoss

# ==========================================
# 1. Define Benchmark Models
# ==========================================

class VanillaLSTM(nn.Module):
    """
    Standard LSTM model without Physics-Informed Loss or Entropy Gating.
    Acts as a baseline to demonstrate the impact of TEG-Net's innovations.
    """
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=3):
        super(VanillaLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

class TimeSeriesTransformer(nn.Module):
    """
    Basic Transformer encoder for time-series forecasting.
    Represents the current SOTA (State-of-the-Art) architecture.
    """
    def __init__(self, input_dim=4, d_model=64, nhead=4, num_layers=2, output_dim=3):
        super(TimeSeriesTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)
    
    def forward(self, x):
        x = self.input_proj(x)
        output = self.transformer_encoder(x)
        # Use the output of the last time step
        return self.fc(output[:, -1, :])

# ==========================================
# 2. Setup & Training Helper
# ==========================================

SEQ_LEN = 20
HORIZON = 5
EPOCHS = 50  
# [FIX] Explicitly define the device as CPU to avoid GPU kernel errors.
DEVICE = torch.device('cpu') 
print(f">>> Force Device: {DEVICE} (Due to RTX 5070 Compatibility)")

def create_sequences(data, seq_len, horizon):
    """
    Prepares sliding window sequences for time-series training.
    """
    X, Y = [], []
    for i in range(len(data) - seq_len - horizon + 1):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len + horizon - 1, :3])
    return np.array(X), np.array(Y)

def train_torch_model(model, X_train, Y_train, name="Model"):
    """
    Generic training loop for PyTorch benchmark models (LSTM, Transformer).
    Uses standard MSE Loss.
    """
    print(f">>> Training {name}...")
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    model.train()
    for epoch in range(EPOCHS):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
    return model

# ==========================================
# 3. Main Benchmark Pipeline
# ==========================================

def run_benchmark():
    # 1. Load Data
    df = get_vix_term_structure()
    if df is None: return
    
    # 2. Preprocessing
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values)
    X, Y = create_sequences(scaled_data, SEQ_LEN, HORIZON)
    
    # Train/Test Split (80% / 20%)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]
    
    # Tensor Conversion (Sent to CPU)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).to(DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    
    results = {}
    
    # --- Model 1: TEG-Net (Ours) ---
    print(">>> [1/4] Training TEG-Net (Ours)...")
    teg_net = TEGNet_TermStructure(input_dim=4, hidden_dim=64, output_dim=3).to(DEVICE)
    
    # Using the "Calibrated Squeeze Loss" settings (Balanced)
    criterion_physics = ReactionDiffusionLoss(diff_coeff=0.01, react_coeff=5.0, crisis_threshold=0.25)
    opt_teg = optim.Adam(teg_net.parameters(), lr=0.01)
    
    teg_net.train()
    for epoch in range(100): # TEG-Net requires more epochs for physics convergence
        opt_teg.zero_grad()
        entropy = torch.std(X_train_t[:, :, 0], dim=1, keepdim=True)
        preds = teg_net(X_train_t, entropy)
        loss = criterion_physics(preds, Y_train_t, X_train_t[:, -1, :])
        loss.backward()
        opt_teg.step()
        
    teg_net.eval()
    with torch.no_grad():
        ent_test = torch.std(X_test_t[:, :, 0], dim=1, keepdim=True)
        results['TEG-Net'] = teg_net(X_test_t, ent_test).cpu().numpy()

    # --- Model 2: Vanilla LSTM ---
    lstm_model = VanillaLSTM().to(DEVICE)
    lstm_model = train_torch_model(lstm_model, X_train_t, Y_train_t, "Vanilla LSTM")
    lstm_model.eval()
    with torch.no_grad():
        results['LSTM'] = lstm_model(X_test_t).cpu().numpy()

    # --- Model 3: Transformer ---
    trans_model = TimeSeriesTransformer().to(DEVICE)
    trans_model = train_torch_model(trans_model, X_train_t, Y_train_t, "Transformer")
    trans_model.eval()
    with torch.no_grad():
        results['Transformer'] = trans_model(X_test_t).cpu().numpy()

    # --- Model 4: XGBoost ---
    print(">>> [4/4] Training XGBoost...")
    # XGBoost requires flattened 2D input
    X_train_xgb = X_train.reshape(X_train.shape[0], -1)
    X_test_xgb = X_test.reshape(X_test.shape[0], -1)
    
    xgb_preds = []
    # Train separate regressors for each target (1M, 3M, 6M)
    for i in range(3):
        reg = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
        reg.fit(X_train_xgb, Y_train[:, i])
        xgb_preds.append(reg.predict(X_test_xgb))
    
    results['XGBoost'] = np.stack(xgb_preds, axis=1)

    # ==========================================
    # 4. Visualization & Metrics
    # ==========================================
    
    # Prepare Ground Truth for plotting
    dummy_real = np.zeros((len(Y_test), 4))
    dummy_real[:, :3] = Y_test
    real_targets = scaler.inverse_transform(dummy_real)[:, :3]

    plt.figure(figsize=(15, 8))
    plt.plot(real_targets[:, 0], label='Actual VIX', color='black', linewidth=2, alpha=0.5)
    
    colors = {'TEG-Net': 'red', 'LSTM': 'blue', 'Transformer': 'green', 'XGBoost': 'orange'}
    styles = {'TEG-Net': '-', 'LSTM': '--', 'Transformer': '-.', 'XGBoost': ':'}
    widths = {'TEG-Net': 2.0, 'LSTM': 1.0, 'Transformer': 1.0, 'XGBoost': 1.0}
    
    print("\n>>> [Benchmark Results - RMSE]")
    for name, pred in results.items():
        # Inverse Scale predictions
        dummy_pred = np.zeros((len(pred), 4))
        dummy_pred[:, :3] = pred
        real_pred = scaler.inverse_transform(dummy_pred)[:, :3]
        
        rmse = np.sqrt(mean_squared_error(real_targets, real_pred))
        print(f"    {name}: {rmse:.4f}")
        
        plt.plot(real_pred[:, 0], label=f'{name} (RMSE: {rmse:.2f})', 
                 color=colors[name], linestyle=styles[name], linewidth=widths[name])

    plt.title("Benchmark Comparison: TEG-Net vs State-of-the-Art Models")
    plt.xlabel("Time (Days)")
    plt.ylabel("VIX (1M)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = "img/benchmark_result.png"
    if not os.path.exists('img'): os.makedirs('img')
    plt.savefig(save_path)
    print(f">>> Benchmark graph saved to '{save_path}'")
    plt.show()

if __name__ == "__main__":
    run_benchmark()