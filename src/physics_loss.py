import torch
import torch.nn as nn

class ReactionDiffusionLoss(nn.Module):
    """
    [Balanced Squeeze Loss]
    
    Adjusted to prevent overshooting while maintaining reactivity.
    The previous 'Nuclear Option' (200x) caused the model to predict unrealistic spikes (>75).
    We calibrate the penalties to a 'Strong but Reasonable' level.
    
    Strategy:
    1. Crisis (Underestimation) -> Penalty x50.0 (Strong enough to catch spikes, stops overshooting)
    2. Stable (Overestimation)  -> Penalty x10.0 (Standard suppression for noise)
    """
    def __init__(self, diff_coeff=0.01, react_coeff=5.0, crisis_threshold=0.25):
        super(ReactionDiffusionLoss, self).__init__()
        # Fixed Physics Coefficients
        self.D = diff_coeff
        self.R = react_coeff
        self.threshold = crisis_threshold

    def forward(self, pred, target, inputs):
        """
        pred: [Batch, 3]
        target: [Batch, 3]
        inputs: [Batch, Seq, Feat]
        """
        
        # --- 1. Balanced Asymmetric Data Loss ---
        residual = target - pred
        
        # Detect Regimes
        is_crisis = target > self.threshold
        is_stable = ~is_crisis
        
        # Detect Errors
        is_under = residual > 0 # Prediction < Actual
        is_over = residual < 0  # Prediction > Actual
        
        weights = torch.ones_like(residual)
        
        # [Crisis Zone]
        # Tuned Down: 200.0 -> 50.0
        # 50x is the "Goldilocks" zone: catches the spike but prevents flying to the moon.
        weights = torch.where(is_crisis & is_under, 50.0, weights)
        
        # [Stable Zone]
        # Tuned Down: 20.0 -> 10.0
        # Keeps the baseline clean without suppressing recovery too much.
        weights = torch.where(is_stable & is_over, 10.0, weights)
        
        # Calculate Weighted MSE
        data_loss = torch.mean(weights * (residual ** 2))
        
        # --- 2. Physics Constraints ---
        
        # A. Diffusion (Smoothing)
        v1, v3, v6 = pred[:, 0], pred[:, 1], pred[:, 2]
        laplacian = v1 - 2*v3 + v6
        diffusion_term = self.D * torch.mean(laplacian**2)
        
        # B. Reaction (Excitation)
        # SKEW-based trigger for "Shape Guidance"
        skew_scaled = inputs[:, -1]
        skew_trigger = torch.relu(skew_scaled - 0.6)
        
        # Reaction Force
        # Guided by 'react_coeff' from main.py
        reaction_force = self.R * skew_trigger * v1
        reaction_term = torch.mean(reaction_force**2) * 0.01 
        
        return data_loss + diffusion_term + reaction_term