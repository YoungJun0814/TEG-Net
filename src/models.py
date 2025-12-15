import torch
import torch.nn as nn

class TEGNet_TermStructure(nn.Module):
    """
    TEG-Net (Trend-Entropy Gated Network) adapted for Term Structure Forecasting.
    
    This architecture integrates a regime-switching mechanism based on Entropy
    to dynamically weight the contribution of Trend (Stable) and Chaos (Spike) streams.
    
    Attributes:
        lstm_trend: Captures long-term mean-reverting patterns.
        lstm_chaos: Captures short-term shock propagation and noise.
        gate: An entropy-based neural gate to determine the market regime alpha.
    """
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=3):
        super(TEGNet_TermStructure, self).__init__()
        
        # Stream 1: Trend Expert (Stable market dynamics)
        self.lstm_trend = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Stream 2: Chaos Expert (Crisis dynamics)
        self.lstm_chaos = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Output Layers: Predicts the entire curve [VIX_1M, VIX_3M, VIX_6M]
        self.fc_trend = nn.Linear(hidden_dim, output_dim) 
        self.fc_chaos = nn.Linear(hidden_dim, output_dim)
        
        # Entropy Gate: 
        # Maps market entropy to a weight scalar alpha within [0, 1]
        self.gate = nn.Sequential(
            nn.Linear(1, 16), 
            nn.ReLU(), 
            nn.Linear(16, 1), 
            nn.Sigmoid()
        )

    def forward(self, x, entropy):
        """
        Args:
            x (torch.Tensor): Input sequence [Batch, Seq_Len, Features]
            entropy (torch.Tensor): Calculated entropy of the sequence [Batch, 1]
        """
        # LSTM Forward pass
        _, (h_t, _) = self.lstm_trend(x)
        _, (h_c, _) = self.lstm_chaos(x)
        
        # Calculate Regime Weight (Alpha)
        # alpha -> 1 implies Chaos/High-Risk regime
        # alpha -> 0 implies Trend/Stable regime
        alpha = self.gate(entropy)
        
        # Adaptive Fusion
        # The final prediction is a weighted average of both experts
        out = alpha * self.fc_chaos(h_c[-1]) + (1-alpha) * self.fc_trend(h_t[-1])
        return out