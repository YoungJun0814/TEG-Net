import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Import custom modules from src
from src.data_loader import get_vix_term_structure
from src.models import TEGNet_TermStructure
from src.physics_loss import reaction_diffusion_loss_with_inputs

# ==========================================
# Hyperparameters (Calibrated Tuning)
# ==========================================
SEQ_LEN = 20        
HORIZON = 5         
EPOCHS = 100        
LEARNING_RATE = 0.01 
HIDDEN_DIM = 64
BATCH_SIZE = 32


def create_sequences(data, seq_len, horizon):
    """
    Generates temporal sequences for LSTM training.
    """
    X, Y = [], []
    for i in range(len(data) - seq_len - horizon + 1):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len + horizon - 1, :3]) 
    return np.array(X), np.array(Y)


def run_pipeline():
    print(">>> [System] Initializing VIX Reaction-Diffusion Project (PyTorch)...")
    
    # ==========================================
    # Device Setup (Force CPU)
    # ==========================================
    # Current PyTorch version does not support RTX 5070 (sm_120) yet.
    # Forcing CPU usage to prevent CUDA kernel errors.
    device = torch.device('cpu')
    print(">>> [Device] Force CPU Mode (GPU disabled due to compatibility issues)")
    
    # 1. Data Loading
    df = get_vix_term_structure()
    if df is None:
        print(">>> [Error] Failed to load data. Exiting.")
        return

    # 2. Preprocessing & Scaling
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values)
    
    X, Y = create_sequences(scaled_data, SEQ_LEN, HORIZON)
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]
    
    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32).to(device)
    
    print(f">>> [Data] Train Samples: {X_train.shape[0]}, Test Samples: {X_test.shape[0]}")

    # 3. Model Initialization
    model = TEGNet_TermStructure(input_dim=4, hidden_dim=HIDDEN_DIM, output_dim=3).to(device)
    
    # [Calibrated Physics Parameters]
    # diff_coeff=0.01: Keeps the prediction sharp (prevent lagging).
    # react_coeff=5.0: Reduced from 20.0 to prevent overshooting (75 -> 55 range).
    # crisis_threshold=0.25: Trigger point for asymmetric loss.
    loss_fn = reaction_diffusion_loss_with_inputs(
        diff_coeff=0.01, 
        react_coeff=5.0, 
        crisis_threshold=0.25
    )
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. Training Loop
    print(">>> [Training] Starting optimization...")
    
    # Create dataset for batching
    train_dataset = torch.utils.data.TensorDataset(X_train_t, Y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        num_batches = 0
        
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # Calculate entropy (std of VIX sequence)
            vix_seq = x_batch[:, :, 0]  # [Batch, Seq_Len]
            entropy_input = torch.std(vix_seq, dim=1, keepdim=True)  # [Batch, 1]
            
            # Forward pass
            preds = model(x_batch, entropy_input)
            
            # Loss calculation with physics constraints
            loss = loss_fn(preds, y_batch, x_batch[:, -1, :])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.6f}")

    # 5. Evaluation & Visualization
    print(">>> [Evaluation] Generating Forecasts...")
    
    model.eval()
    with torch.no_grad():
        # Calculate test entropy
        test_vix_seq = X_test_t[:, :, 0]
        test_entropy = torch.std(test_vix_seq, dim=1, keepdim=True)
        
        # Get predictions
        test_preds = model(X_test_t, test_entropy).cpu().numpy()
    
    # Inverse transform to original scale
    dummy_pred = np.zeros((len(test_preds), 4))
    dummy_real = np.zeros((len(Y_test), 4))
    
    dummy_pred[:, :3] = test_preds
    dummy_real[:, :3] = Y_test
    
    real_preds = scaler.inverse_transform(dummy_pred)[:, :3]
    real_targets = scaler.inverse_transform(dummy_real)[:, :3]
    
    rmse = np.sqrt(mean_squared_error(real_targets, real_preds))
    print(f">>> [Result] Test RMSE: {rmse:.4f}")

    # 6. Plotting
    if not os.path.exists('img'):
        os.makedirs('img')
        
    plt.figure(figsize=(14, 7))
    
    # Plot Actual
    plt.plot(real_targets[:, 0], label='Actual VIX (1M)', color='black', alpha=0.6, linewidth=1)
    
    # Plot Predicted
    plt.plot(real_preds[:, 0], label='TEG-Net Forecast (1M)', color='red', linewidth=1.5)
    
    # Highlight Crisis
    crisis_mask = real_targets[:, 0] > 30
    if np.any(crisis_mask):
        plt.scatter(np.where(crisis_mask)[0], real_targets[crisis_mask, 0], 
                    color='orange', s=15, label='High Volatility Regime')

    plt.title("VIX Term Structure Forecast: PyTorch Physics Model")
    plt.xlabel("Time (Days)")
    plt.ylabel("Volatility Index")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = "img/forecast_result_pytorch.png"
    plt.savefig(save_path)
    print(f">>> [Output] Graph saved to '{save_path}'")
    plt.show()


if __name__ == "__main__":
    run_pipeline()
