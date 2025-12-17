import torch
import torch.nn as nn


class TEGNet_TermStructure(nn.Module):
    """
    TEG-Net (Trend-Entropy Gated Network) - Improved Version
    
    Improvements over v1:
    1. 2-Layer LSTM for deeper temporal pattern learning
    2. Enhanced Entropy Gate (32→16→1) for better regime detection
    3. Residual Connection from last input timestep
    4. Dropout for regularization
    """
    
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=3, dropout=0.2):
        super(TEGNet_TermStructure, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Stream 1: Trend Expert (Stable market dynamics)
        # Upgraded to 2-layer LSTM with dropout
        self.lstm_trend = nn.LSTM(
            input_dim, hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            dropout=dropout
        )
        
        # Stream 2: Chaos Expert (Crisis dynamics)
        self.lstm_chaos = nn.LSTM(
            input_dim, hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            dropout=dropout
        )
        
        # Output Layers
        self.fc_trend = nn.Linear(hidden_dim, output_dim)
        self.fc_chaos = nn.Linear(hidden_dim, output_dim)
        
        # Enhanced Entropy Gate: Deeper network for better regime sensitivity
        self.gate = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Residual Connection: Directly use last timestep input
        self.residual_fc = nn.Linear(input_dim, output_dim)
        self.residual_weight = 0.1  # Small weight to avoid dominating
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence."""
        for name, param in self.named_parameters():
            if 'weight' in name and 'lstm' not in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, entropy):
        """
        Args:
            x: Input sequence [Batch, Seq_Len, Features]
            entropy: Calculated entropy of the sequence [Batch, 1]
        
        Returns:
            out: Predicted values [Batch, output_dim]
        """
        # LSTM Forward pass (2-layer)
        _, (h_t, _) = self.lstm_trend(x)  # h_t: [2, Batch, hidden_dim]
        _, (h_c, _) = self.lstm_chaos(x)  # h_c: [2, Batch, hidden_dim]
        
        # Use the last layer's hidden state
        h_t = h_t[-1]  # [Batch, hidden_dim]
        h_c = h_c[-1]  # [Batch, hidden_dim]
        
        # Calculate Regime Weight (Alpha)
        alpha = self.gate(entropy)  # [Batch, 1]
        
        # Expert outputs
        trend_out = self.fc_trend(h_t)  # [Batch, output_dim]
        chaos_out = self.fc_chaos(h_c)  # [Batch, output_dim]
        
        # Adaptive Fusion
        out = alpha * chaos_out + (1 - alpha) * trend_out
        
        # Residual Connection: Add scaled contribution from last input
        last_input = x[:, -1, :]  # [Batch, input_dim]
        residual = self.residual_fc(last_input)  # [Batch, output_dim]
        out = out + self.residual_weight * residual
        
        return out