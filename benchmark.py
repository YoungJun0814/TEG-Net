import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import os

# Import our TEG-Net components
from src.data_loader import get_vix_term_structure
from src.models import TEGNet_TermStructure
from src.physics_loss import ReactionDiffusionLoss

# ==========================================
# 1. Define Benchmark Models
# ==========================================

# A. Vanilla LSTM (No Physics, No Gate)
class VanillaLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=3):
        super(VanillaLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# B. Time-Series Transformer (Simple Version)
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=4, d_model=64, nhead=4, num_layers=2, output_dim=3):
        super(TimeSeriesTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)
    
    def forward(self, x):
        x = self.input_proj(x)
        # Add simple positional encoding (optional but good for transformer)
        output = self.transformer_encoder(x)
        # Use the last time step's output
        return self.fc(output[:, -1, :])

# ==========================================
# 2. Setup & Training Helper
# ==========================================

SEQ_LEN = 20
HORIZON = 5
EPOCHS = 50  # Comparison models train faster
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_sequences(data, seq_len, horizon):
    X, Y = [], []
    for i in range(len(data) - seq_len - horizon + 1):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len + horizon - 1, :3])
    return np.array(X), np.array(Y)

def train_torch_model(model, X_train, Y_train, name="Model"):
    print(f">>> Training {name}...")
    criterion = nn.MSELoss() # Benchmarks utilize standard MSE
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    model.train()
    for epoch in range(EPOCHS):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            if name == "Transformer": # Transformers need dummy mask handling sometimes
                preds = model(batch_x)
            else:
                preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
    return model

# ==========================================
# 3. Main Benchmark Pipeline
# ==========================================

def run_benchmark():
    # Load Data
    df = get_vix_term_structure()
    if df is None: return
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values)
    X, Y = create_sequences(scaled_data, SEQ_LEN, HORIZON)
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]
    
    # Tensor Conversion
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).to(DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    
    results = {}
    
    # --- 1. TEG-Net (Ours) ---
    print(">>> [1/4] Training TEG-Net (Ours)...")
    teg_net = TEGNet_TermStructure(input_dim=4, hidden_dim=64, output_dim=3).to(DEVICE)
    # Using the Calibrated Squeeze Loss settings
    criterion_physics = ReactionDiffusionLoss(diff_coeff=0.01, react_coeff=5.0, crisis_threshold=0.25)
    opt_teg = optim.Adam(teg_net.parameters(), lr=0.01)
    
    for epoch in range(100): # Ours needs more epochs for physics
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

    # --- 2. Vanilla LSTM ---
    lstm_model = VanillaLSTM().to(DEVICE)
    lstm_model = train_torch_model(lstm_model, X_train_t, Y_train_t, "Vanilla LSTM")
    lstm_model.eval()
    with torch.no_grad():
        results['LSTM'] = lstm_model(X_test_t).cpu().numpy()

    # --- 3. Transformer ---
    trans_model = TimeSeriesTransformer().to(DEVICE)
    trans_model = train_torch_model(trans_model, X_train_t, Y_train_t, "Transformer")
    trans_model.eval()
    with torch.no_grad():
        results['Transformer'] = trans_model(X_test_t).cpu().numpy()

    # --- 4. XGBoost ---
    print(">>> [4/4] Training XGBoost...")
    # XGBoost requires 2D input (flatten time steps)
    X_train_xgb = X_train.reshape(X_train.shape[0], -1)
    X_test_xgb = X_test.reshape(X_test.shape[0], -1)
    
    # Train separate models for VIX 1M, 3M, 6M
    xgb_preds = []
    for i in range(3):
        reg = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
        reg.fit(X_train_xgb, Y_train[:, i])
        xgb_preds.append(reg.predict(X_test_xgb))
    
    results['XGBoost'] = np.stack(xgb_preds, axis=1)

    # ==========================================
    # 4. Visualization & Metrics
    # ==========================================
    
    # Prepare Ground Truth
    dummy_real = np.zeros((len(Y_test), 4))
    dummy_real[:, :3] = Y_test
    real_targets = scaler.inverse_transform(dummy_real)[:, :3]

    plt.figure(figsize=(15, 8))
    plt.plot(real_targets[:, 0], label='Actual VIX', color='black', linewidth=2, alpha=0.6)
    
    colors = {'TEG-Net': 'red', 'LSTM': 'blue', 'Transformer': 'green', 'XGBoost': 'orange'}
    styles = {'TEG-Net': '-', 'LSTM': '--', 'Transformer': '-.', 'XGBoost': ':'}
    
    print("\n>>> [Benchmark Results - RMSE]")
    for name, pred in results.items():
        # Inverse Scale
        dummy_pred = np.zeros((len(pred), 4))
        dummy_pred[:, :3] = pred
        real_pred = scaler.inverse_transform(dummy_pred)[:, :3]
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(real_targets, real_pred))
        print(f"    {name}: {rmse:.4f}")
        
        # Plot
        plt.plot(real_pred[:, 0], label=f'{name} (RMSE: {rmse:.2f})', 
                 color=colors[name], linestyle=styles[name], linewidth=1.5)

    plt.title("Benchmark Comparison: TEG-Net vs State-of-the-Art Models")
    plt.xlabel("Time (Days)")
    plt.ylabel("VIX (1M)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if not os.path.exists('img'): os.makedirs('img')
    plt.savefig("img/benchmark_result.png")
    print(">>> Benchmark graph saved to 'img/benchmark_result.png'")
    plt.show()

if __name__ == "__main__":
    run_benchmark()