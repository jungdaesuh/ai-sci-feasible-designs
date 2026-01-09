#!/bin/bash
# RunPod Training Script for StellarForge Diffusion Model
# Usage: bash scripts/runpod_train.sh
#
# Environment variable overrides (for different GPU configurations):
#   BATCH_SIZE  - Training batch size (default: 4096, use 2048 for A100 40GB)
#   HIDDEN_DIM  - Model hidden dimension (default: 2048)
#   N_LAYERS    - Number of transformer layers (default: 6)
#   EPOCHS      - Training epochs (default: 250)
#   PCA_COMPONENTS - PCA dimensions (default: 32)
#
# Example for A100 40GB:
#   BATCH_SIZE=2048 ./scripts/runpod_train.sh

set -e

# Configuration with environment variable overrides
BATCH_SIZE=${BATCH_SIZE:-4096}
HIDDEN_DIM=${HIDDEN_DIM:-2048}
N_LAYERS=${N_LAYERS:-6}
EPOCHS=${EPOCHS:-250}
PCA_COMPONENTS=${PCA_COMPONENTS:-32}
TIMESTEPS=${TIMESTEPS:-1000}
LOG_INTERVAL=${LOG_INTERVAL:-10}

echo "============================================================"
echo "StellarForge Diffusion Model - RunPod Training"
echo "============================================================"
echo "Config: batch_size=$BATCH_SIZE hidden_dim=$HIDDEN_DIM n_layers=$N_LAYERS"
echo "============================================================"

# 1. System setup
echo "[1/5] Installing system dependencies..."
apt-get update && apt-get install -y git

# 2. Clone repo
echo "[2/5] Cloning repository..."
cd /workspace
if [ ! -d "ai-sci-feasible-designs" ]; then
    git clone https://github.com/jungdaesuh/ai-sci-feasible-designs.git
fi
cd ai-sci-feasible-designs

# 3. Install Python dependencies
echo "[3/5] Installing Python dependencies..."
pip install -e . --quiet
pip install torch --upgrade --quiet

# 4. Verify GPU
echo "[4/5] Verifying GPU..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# 5. Run training
echo "[5/5] Starting training..."
echo "============================================================"

python scripts/train_generative_offline.py \
    --batch-size $BATCH_SIZE \
    --hidden-dim $HIDDEN_DIM \
    --n-layers $N_LAYERS \
    --epochs $EPOCHS \
    --pca-components $PCA_COMPONENTS \
    --timesteps $TIMESTEPS \
    --device cuda \
    --output /workspace/diffusion_paper_spec.pt \
    --log-interval $LOG_INTERVAL

echo "============================================================"
echo "Training complete!"
echo "Checkpoint saved to: /workspace/diffusion_paper_spec.pt"
echo "============================================================"
