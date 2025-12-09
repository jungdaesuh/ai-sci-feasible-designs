#!/usr/bin/env python3
"""Offline training script for StellarForge Diffusion Model.

This script trains the DiffusionDesignModel on the ConStellaration dataset
from Hugging Face for pre-training before the optimization loop.

Usage:
    python scripts/train_generative_offline.py --epochs 250 --batch-size 4096

Reference:
    Padidar et al. (2025) - StellarForge architecture specifications
"""

# ruff: noqa: E402
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add current directory to path
sys.path.insert(0, os.getcwd())

from ai_scientist.optim.data_loader import load_constellaration_dataset
from ai_scientist.optim.generative import DiffusionDesignModel

_LOGGER = logging.getLogger(__name__)


def clean_matrix(val):
    """Convert numpy arrays (potentially object-arrays) to list of lists."""
    if hasattr(val, "tolist"):
        val = val.tolist()
    if isinstance(val, list):
        return [x.tolist() if hasattr(x, "tolist") else x for x in val]
    return val


def prepare_diffusion_candidates(df):
    """Extract candidates with params and metrics for diffusion training.

    Returns:
        List[Dict] with keys:
            - params: {r_cos, z_sin, n_field_periods}
            - metrics: {iota, aspect_ratio, nfp, is_qa}
    """
    candidates = []
    skipped = 0

    for _, row in df.iterrows():
        try:
            r_cos = clean_matrix(row["boundary.r_cos"])
            z_sin = clean_matrix(row["boundary.z_sin"])

            # Validate shapes are consistent
            r_arr = np.asarray(r_cos, dtype=float)
            z_arr = np.asarray(z_sin, dtype=float)

            if r_arr.ndim != 2 or z_arr.ndim != 2:
                skipped += 1
                continue

            nfp = int(row.get("boundary.n_field_periods", 3))

            params = {
                "r_cos": r_cos,
                "z_sin": z_sin,
                "n_field_periods": nfp,
            }

            # Metrics for conditioning (matching DiffusionDesignModel.METRIC_KEYS)
            metrics = {
                "edge_rotational_transform_over_n_field_periods": float(
                    row.get("edge_rotational_transform_over_n_field_periods", 0.42)
                ),
                "aspect_ratio": float(row.get("aspect_ratio", 8.0)),
                "number_of_field_periods": float(nfp),
                "is_quasihelical": float(row.get("is_quasihelical", 0.0)),
            }

            candidates.append({"params": params, "metrics": metrics})

        except (ValueError, TypeError, KeyError) as e:
            _LOGGER.debug(f"Skipping row due to error: {e}")
            skipped += 1
            continue

    print(f"Prepared {len(candidates)} candidates ({skipped} skipped due to errors)")
    return candidates


def main():
    parser = argparse.ArgumentParser(
        description="Train StellarForge Diffusion Model on ConStellaration dataset"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=250,
        help="Number of training epochs (default: 250, paper spec)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size (default: 4096, paper spec)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect cuda/mps/cpu)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/diffusion_paper_spec.pt",
        help="Output checkpoint path",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=2048,
        help="Hidden dimension (default: 2048, paper spec)",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=4,
        help="Number of MLP layers (default: 4, paper spec)",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=50,
        help="PCA components for latent space (default: 50)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200,
        help="Diffusion timesteps (default: 200, paper spec)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Print progress every N epochs (default: 10)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="proxima-fusion/constellaration",
        help="Hugging Face dataset name (default: proxima-fusion/constellaration)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Auto-detect device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("=" * 60)
    print("StellarForge Diffusion Model - Offline Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"N layers: {args.n_layers}")
    print(f"PCA components: {args.pca_components}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Log interval: {args.log_interval}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print("=" * 60)

    # 1. Load dataset
    print("\n[1/4] Loading ConStellaration dataset...")
    try:
        df = load_constellaration_dataset(filter_geometry=True)
        print(f"Loaded {len(df)} valid samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you have internet access for HuggingFace datasets")
        return 1

    # 2. Prepare training data
    print("\n[2/4] Preparing training candidates...")
    candidates = prepare_diffusion_candidates(df)

    if not candidates:
        print("Error: No valid candidates for training")
        return 1

    if len(candidates) < 1000:
        print(f"Warning: Only {len(candidates)} candidates. Consider using more data.")

    # 3. Initialize and train model
    print("\n[3/4] Training Diffusion Model...")
    print(f"Architecture: MLP with {args.n_layers} layers x {args.hidden_dim} hidden")

    model = DiffusionDesignModel(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        timesteps=args.timesteps,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        pca_components=args.pca_components,
        log_interval=args.log_interval,
    )

    # Custom training loop to print progress
    # Note: DiffusionDesignModel.fit() handles the training internally

    # Train with progress logging
    model.fit(candidates)

    if not model._trained:
        print("Error: Training did not complete successfully")
        return 1

    # 4. Save checkpoint
    print("\n[4/4] Saving checkpoint...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = model.state_dict()
    torch.save(checkpoint, output_path)

    # Print summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Checkpoint saved: {output_path}")
    print(f"Checkpoint size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Validation: Generate a few samples to verify
    print("\nValidation: Generating 3 test samples...")
    try:
        test_metrics = {
            "edge_rotational_transform_over_n_field_periods": 0.42,
            "aspect_ratio": 8.0,
            "number_of_field_periods": 3.0,
            "is_quasihelical": 0.0,
        }
        test_samples = model.sample(3, target_metrics=test_metrics, seed=42)
        print(f"Generated {len(test_samples)} test samples successfully")

        if test_samples:
            sample = test_samples[0]
            params = sample.get("params", {})
            r_cos = np.array(params.get("r_cos", []))
            print(f"Sample shape: r_cos={r_cos.shape}")
    except Exception as e:
        print(f"Warning: Validation sampling failed: {e}")

    print("\nDone! You can now use this checkpoint in your experiment config:")
    print("  generative:")
    print("    enabled: true")
    print("    backend: diffusion")
    print(f"    checkpoint_path: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
