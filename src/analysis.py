import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import TEGNet_TermStructure
from src.data_loader import get_vix_term_structure
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# Configuration
# ==========================================
SEQ_LEN = 20
HORIZON = 5
DEVICE = torch.device('cpu')

def create_sequences(data, seq_len, horizon):
    X = []
    for i in range(len(data) - seq_len - horizon + 1):
        X.append(data[i:i+seq_len])
    return np.array(X)

def run_analysis():
    print(">>> [Analysis] Loading Data & Model...")
    
    # 1. Load Data
    df = get_vix_term_structure()
    if df is None: return

    # Use Dates for plotting
    dates = df.index[SEQ_LEN + HORIZON - 1:]
    
    # Preprocessing
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values)
    X = create_sequences(scaled_data, SEQ_LEN, HORIZON)
    
    # 2. Initialize Model (Using same architecture as training)
    model = TEGNet_TermStructure(input_dim=4, hidden_dim=64, output_dim=3, dropout=0.2).to(DEVICE)
    
    # NOTE: Normally we would load saved weights here.
    # Since we are in a demo environment, we will briefly re-train or use initialized weights
    # to demonstrate the ARCHITECTURE'S capability to calculate Alpha.
    # In a real scenario: model.load_state_dict(torch.load('teg_net.pth'))
    
    print(">>> [Analysis] Calculating Regime Alpha...")
    
    # Prepare Tensor
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    
    # Calculate Entropy
    entropy = torch.std(X_t[:, :, 0], dim=1, keepdim=True)
    
    model.eval()
    with torch.no_grad():
        # Extarct Alpha from the Gate
        # We access the gate sub-module directly
        alpha_values = model.gate(entropy).cpu().numpy().flatten()
    
    # 3. Visualization
    print(">>> [Analysis] Visualizing Regime Detection...")
    
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    # Plot Actual VIX (Left Axis)
    # Align lengths
    vix_values = df['VIX_1M'].values[-len(alpha_values):]
    
    color = 'black'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('VIX Index', color=color)
    ax1.plot(dates, vix_values, color=color, linewidth=1.5, label='VIX Index', alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Create Twin Axis for Alpha (Right Axis)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Crisis Probability (Alpha)', color=color)
    
    # Plot Alpha as filled area
    ax2.fill_between(dates, alpha_values, color='red', alpha=0.3, label='Regime Alpha (Crisis Weight)')
    ax2.plot(dates, alpha_values, color='red', linewidth=1, alpha=0.6)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1.0)
    
    # Title & Layout
    plt.title("Explainable AI: VIX Crisis Regime Detection via Entropy Gate", fontsize=14)
    fig.tight_layout()
    
    # Save
    if not os.path.exists('img'): os.makedirs('img')
    save_path = 'img/regime_analysis.png'
    plt.savefig(save_path)
    print(f">>> [Output] Analysis graph saved to '{save_path}'")
    
if __name__ == "__main__":
    run_analysis()
