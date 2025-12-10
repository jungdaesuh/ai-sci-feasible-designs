#!/bin/bash
# RunPod Training Script for StellarForge Diffusion Model
# Usage: bash scripts/runpod_train.sh

set -e

echo "============================================================"
echo "StellarForge Diffusion Model - RunPod Training"
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
    --batch-size 4096 \
    --hidden-dim 2048 \
    --n-layers 6 \
    --epochs 250 \
    --pca-components 32 \
    --timesteps 1000 \
    --device cuda \
    --output /workspace/diffusion_paper_spec.pt \
    --log-interval 10

echo "============================================================"
echo "Training complete!"
echo "Checkpoint saved to: /workspace/diffusion_paper_spec.pt"
echo "============================================================"
