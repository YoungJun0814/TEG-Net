import torch
import torch.nn as nn


class ReactionDiffusionLoss(nn.Module):
    """
    Physics-Informed Loss Function - Improved Version
    
    Improvements:
    1. Softened asymmetric weights (10x/3x instead of 50x/10x)
    2. Huber Loss for robustness against outliers
    3. Reduced physics constraint contribution
    """
    
    def __init__(self, diff_coeff=0.01, react_coeff=1.0, crisis_threshold=0.25):
        super(ReactionDiffusionLoss, self).__init__()
        self.D = diff_coeff
        self.R = react_coeff
        self.threshold = crisis_threshold
        self.huber = nn.SmoothL1Loss(reduction='none')
    
    def forward(self, y_pred, y_true, inputs=None):
        """
        Args:
            y_pred: [Batch, 3] - Predicted values
            y_true: [Batch, 3] - Target values
            inputs: [Batch, Features] - Last timestep input (for reaction term)
        
        Returns:
            loss: Scalar tensor
        """
        # --- 1. Robust Asymmetric Data Loss ---
        residual = y_true - y_pred
        
        # Detect Regimes
        is_crisis = (y_true > self.threshold).float()
        is_stable = 1.0 - is_crisis
        
        # Detect Errors
        is_under = (residual > 0).float()  # Prediction < Actual
        is_over = (residual < 0).float()   # Prediction > Actual
        
        # Base weights
        weights = torch.ones_like(residual)
        
        # [Crisis Zone] - Softened from 50x to 10x
        crisis_mask = (is_crisis > 0.5) & (is_under > 0.5)
        weights = torch.where(crisis_mask, torch.tensor(10.0, device=weights.device), weights)
        
        # [Stable Zone] - Softened from 10x to 3x
        stable_mask = (is_stable > 0.5) & (is_over > 0.5)
        weights = torch.where(stable_mask, torch.tensor(3.0, device=weights.device), weights)
        
        # Huber Loss: Robust to outliers
        huber_loss = self.huber(y_pred, y_true)
        data_loss = torch.mean(weights * huber_loss)
        
        # --- 2. Physics Constraints (Reduced weight) ---
        
        # A. Diffusion (Smoothing) - Reduced contribution
        v1 = y_pred[:, 0]
        v3 = y_pred[:, 1]
        v6 = y_pred[:, 2]
        laplacian = v1 - 2 * v3 + v6
        diffusion_term = self.D * torch.mean(laplacian ** 2) * 0.1
        
        # B. Reaction (Excitation)
        reaction_term = 0.0
        if inputs is not None:
            skew_scaled = inputs[:, -1]
            skew_trigger = torch.relu(skew_scaled - 0.6)
            reaction_force = self.R * skew_trigger * v1
            reaction_term = torch.mean(reaction_force ** 2) * 0.001
        
        return data_loss + diffusion_term + reaction_term


def reaction_diffusion_loss_with_inputs(diff_coeff=0.01, react_coeff=1.0, crisis_threshold=0.25):
    """
    Factory function returning a loss function for custom training loops.
    """
    huber = nn.SmoothL1Loss(reduction='none')
    
    def loss_fn(y_pred, y_true, inputs=None):
        residual = y_true - y_pred
        
        # Detect Regimes
        is_crisis = (y_true > crisis_threshold).float()
        is_stable = 1.0 - is_crisis
        
        # Detect Errors
        is_under = (residual > 0).float()
        is_over = (residual < 0).float()
        
        # Softened asymmetric weights
        weights = torch.ones_like(residual)
        
        crisis_mask = (is_crisis > 0.5) & (is_under > 0.5)
        weights = torch.where(crisis_mask, torch.tensor(10.0, device=weights.device), weights)
        
        stable_mask = (is_stable > 0.5) & (is_over > 0.5)
        weights = torch.where(stable_mask, torch.tensor(3.0, device=weights.device), weights)
        
        # Huber Loss
        huber_loss = huber(y_pred, y_true)
        data_loss = torch.mean(weights * huber_loss)
        
        # Physics Constraints (reduced)
        v1, v3, v6 = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
        laplacian = v1 - 2 * v3 + v6
        diffusion_term = diff_coeff * torch.mean(laplacian ** 2) * 0.1
        
        reaction_term = 0.0
        if inputs is not None:
            skew_trigger = torch.relu(inputs[:, -1] - 0.6)
            reaction_force = react_coeff * skew_trigger * v1
            reaction_term = torch.mean(reaction_force ** 2) * 0.001
        
        return data_loss + diffusion_term + reaction_term
    
    return loss_fn