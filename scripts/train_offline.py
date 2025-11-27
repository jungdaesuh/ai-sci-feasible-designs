import argparse
import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import torch

# Add current directory to path
sys.path.append(os.getcwd())

from ai_scientist.optim.data_loader import load_constellaration_dataset, LogRobustScaler, save_scaler
from ai_scientist.optim.surrogate_v2 import NeuralOperatorSurrogate

# Define Physics Columns to Scale
PHYSICS_COLS = [
    "qi",
    "edge_magnetic_mirror_ratio",
    "minimum_normalized_magnetic_gradient_scale_length",
    "edge_rotational_transform_over_n_field_periods", # iota
]

def clean_matrix(val):
    """Convert numpy arrays (potentially object-arrays) to list of lists."""
    if hasattr(val, 'tolist'):
        val = val.tolist()
    # Now val is list
    if isinstance(val, list):
        return [x.tolist() if hasattr(x, 'tolist') else x for x in val]
    return val

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Offline Data Pipeline for AI Scientist V2")
    parser.add_argument("--output_dir", type=str, default="checkpoints/v2_1", help="Directory to save artifacts")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)

    print("1. Loading and Cleaning Data...")
    try:
        df = load_constellaration_dataset(filter_geometry=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Loaded {len(df)} valid samples.")
    
    available_physics_cols = [c for c in PHYSICS_COLS if c in df.columns]
    well_cols = [c for c in df.columns if 'well' in c.lower() or 'magnetic_well' in c.lower()]
    available_physics_cols.extend(well_cols)
    available_physics_cols = list(set(available_physics_cols))
    
    print(f"Physics columns selected for scaling: {available_physics_cols}")
    
    if available_physics_cols:
        # Filter out rows with NaNs in these specific physics columns
        initial_count = len(df)
        df_clean = df.dropna(subset=available_physics_cols)
        
        if len(df_clean) < initial_count:
             print(f"Dropped {initial_count - len(df_clean)} rows with missing physics values for scaling.")

        print("2. Fitting LogRobustScaler...")
        scaler = LogRobustScaler()
        X_physics = df_clean[available_physics_cols].values
        scaler.fit(X_physics)
        
        scaler_path = output_dir / "scaler.pkl"
        save_scaler(scaler, str(scaler_path))
        print(f"Scaler saved to {scaler_path}")
        
        cols_path = output_dir / "physics_cols.txt"
        with open(cols_path, "w") as f:
            for col in available_physics_cols:
                f.write(f"{col}\n")

    print("3. Training Physics Surrogate...")
    
    metrics_list = []
    target_values = []
    
    if 'boundary.r_cos' not in df.columns:
        print("Error: boundary.r_cos not found in dataframe. Cannot train surrogate.")
        return

    target_col = "minimum_normalized_magnetic_gradient_scale_length"
    if target_col not in df.columns:
        print(f"Warning: {target_col} not found. Using aspect_ratio.")
        target_col = "aspect_ratio"
    
    print(f"Training target: {target_col}")

    df_train = df.dropna(subset=[target_col, "boundary.r_cos", "boundary.z_sin"])
    
    print("Preparing training data...")
    skipped_count = 0
    
    for _, row in df_train.iterrows():
        try:
            r_cos = clean_matrix(row["boundary.r_cos"])
            z_sin = clean_matrix(row["boundary.z_sin"])
            
            # Ensure they are valid rectangular matrices for numpy
            np.asarray(r_cos, dtype=float)
            np.asarray(z_sin, dtype=float)
            
            params = {
                "r_cos": r_cos,
                "z_sin": z_sin,
                "n_field_periods": row.get("boundary.n_field_periods", 1),
                "nfp": row.get("boundary.n_field_periods", 1)
            }
            
            metric_payload = {
                "candidate_params": params,
                "metrics": {
                    "vacuum_well": row.get("vacuum_well", -1.0),
                    "qi": row.get("qi", 1.0),
                    target_col: row[target_col]
                }
            }
            metrics_list.append(metric_payload)
            target_values.append(float(row[target_col]))
        except (ValueError, TypeError):
            skipped_count += 1
            continue

    print(f"Training on {len(metrics_list)} samples (Skipped {skipped_count} invalid/jagged).")
    
    if not metrics_list:
        print("No valid samples found for training.")
        return

    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    print(f"Using device: {device}")

    surrogate = NeuralOperatorSurrogate(
        epochs=args.epochs,
        batch_size=64,
        learning_rate=1e-3,
        n_ensembles=3,
        device=device
    )
    
    surrogate.fit(
        metrics_list, 
        target_values, 
        minimize_objective=False 
    )
    
    model_path = checkpoints_dir / "surrogate_physics_v2.pt"
    torch.save(surrogate, model_path)
    print(f"Surrogate model saved to {model_path}")
            
    print("Pipeline complete.")

if __name__ == "__main__":
    main()
