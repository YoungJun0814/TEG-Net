# Physics-Informed Reaction-Diffusion Network for VIX Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Research-green.svg)]()

## 📌 Project Overview

This project implements **TEG-Net (Trend-Entropy Gated Network)**, a novel deep learning architecture for forecasting the **VIX Term Structure** (VIX 1M, 3M, 6M).

Unlike traditional models (LSTM, Transformer) that blindly minimize error, this model incorporates **Physics-Informed Neural Networks (PINNs)** principles based on **Reaction-Diffusion** dynamics to model market fear propagation.

### Key Innovation: Explainability over Black-box
While traditional models like XGBoost may achieve slightly lower RMSE, TEG-Net offers **interpretable insights** into market regimes:
- **Trend Expert**: Captures stable, mean-reverting market behavior.
- **Chaos Expert**: Specialized in modeling explosive volatility spikes (Crisis).
- **Entropy Gate**: Automatically detects market regime switches ($\alpha \in [0, 1]$) based on information entropy.

---

## 🏗 Model Architecture

The architecture consists of two specialized LSTM streams gated by an Entropy mechanism.

```mermaid
graph TD
    Input[VIX Term Structure Input] --> Trend[Trend Expert (LSTM)]
    Input --> Chaos[Chaos Expert (LSTM)]
    Input --> Entropy[Entropy Calculation]
    
    Entropy --> Gate[Entropy Gate (Neural Network)]
    Gate --> Alpha[Regime Weight Alpha]
    
    Trend --> |Linear| OutT[Trend Prediction]
    Chaos --> |Linear| OutC[Chaos Prediction]
    
    Alpha --> Fusion((Adaptive Fusion))
    OutT --> Fusion
    OutC --> Fusion
    
    Fusion --> Output[Final VIX Forecast]
    
    Input -.-> |Residual Connection| Output
```

---

## 📊 Results & Performance

| Model | RMSE (Test) | Characteristics |
|-------|-------------|-----------------|
| **TEG-Net (Ours)** | **3.66** | **Physically Consistent, Explainable Regime Detection** |
| XGBoost | 3.10 | High Accuracy, Poor Extrapolation |
| LSTM Base | 3.50 | Generic Baseline |

> **Note**: While TEG-Net prioritizes physical plausibility (preventing unrealistic negative predictions or impossible curve shapes) via its **Reaction-Diffusion Loss**, it maintains competitive accuracy while offering superior interpretability.

---

## 🧪 Physics-Informed Loss Function

We utilize a custom loss function derived from the **Fisher-KPP Equation**:

$$ \mathcal{L} = \mathcal{L}_{Data} + \lambda_D \mathcal{L}_{Diffusion} + \lambda_R \mathcal{L}_{Reaction} $$

1. **Balanced Data Loss**: Uses asymmetric weights to penalize underestimation of crises (10x penalty) more than overestimation.
2. **Diffusion Term**: Enforces smoothness across the term structure (1M → 3M → 6M).
3. **Reaction Term**: Models the "excitation" of volatility when SKEW index (tail risk) is high.

---

## 🚀 Usage

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Benchmark Training
Train TEG-Net alongside baselines (LSTM, Transformer, XGBoost):
```bash
python benchmark.py
```

### 3. Regime Analysis (Explainability)
Visualize how the model detects crisis regimes:
```bash
python src/analysis.py
```
*Output saved to `img/regime_analysis.png`*

---

## 📂 Project Structure
```
.
├── src/
│   ├── models.py          # TEG-Net PyTorch Implementation
│   ├── physics_loss.py    # Custom Physics-Informed Loss
│   ├── data_loader.py     # Yahoo Finance Data Pipeline
│   └── analysis.py        # Explainability Visualization
├── benchmark.py           # Training & Comparison Script
├── main.py                # Single Model Execution Script
└── requirements.txt       # Dependencies
```
