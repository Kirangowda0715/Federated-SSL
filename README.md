# Federated Self-Supervised Learning (FedSSL) for TB Detection

A production-grade Federated Learning system for detecting Tuberculosis (TB) using Self-Supervised Pre-training (Masked Autoencoders) on large-scale Chest X-ray datasets.

## 🌟 Key Features
- **Federated SSL**: Privacy-preserving training across 5 simulated hospitals (FedAvg/FedProx).
- **MAE Backbone**: Self-supervised learning from unlabeled data (ResNet50 / ViT).
- **Few-Shot TB Detection**: Prototypical Networks capable of high accuracy with minimal labeled data.
- **GPU Optimized**: Full support for NVIDIA CUDA (tested on RTX 2050/3050).
- **Scale**: Verified on the **46,718 image** NIH Clinical Center dataset.

## 📊 Current Milestone Results
| Metric | Score (Round 3) |
| :--- | :--- |
| **AUC** | **0.8942** |
| **Accuracy** | **0.8421** |
| **Sensitivity** | **0.8800** |

## 🚀 Getting Started

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/Kirangowda0715/Federated-SSL
cd Federated-SSL

# Install dependencies
pip install -r requirements.txt

# For GPU support (NVIDIA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Prepare Data
Download the NIH, Shenzhen, and Montgomery datasets and place them in `data/raw/` as described in the [Documentation](PROJECT_DOCUMENTATION.md).

### 3. Training
To start a new simulation:
```bash
python src/federated/simulation.py --config configs/default.yaml --parallel
```

### 4. Resuming Training
To continue from your last saved round:
```bash
python src/federated/simulation.py --config configs/default.yaml --resume --parallel
```

### 5. Evaluation
To run a detailed performance report on the latest model:
```bash
python evaluate_trained_model.py
```

## 📖 Learn More
For a deep dive into the architecture, federated strategies, and physics of the model, see:
👉 **[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)**

---
Developed as a Final Year Project for TB Detection across Federated Hospital Networks.
