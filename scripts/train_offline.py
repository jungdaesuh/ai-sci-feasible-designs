import argparse
import os
import sys
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import torch

# Add current directory to path
sys.path.append(os.getcwd())

from ai_scientist.optim.data_loader import load_constellaration_dataset, LogRobustScaler, save_scaler
from ai_scientist.optim.surrogate_v2 import NeuralOperatorSurrogate
from ai_scientist.optim.generative import DiffusionDesignModel

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

def save_seeds(seeds_df, filename):
    """Save selected seeds to a JSON file in configs/seeds."""
    output_path = Path("configs/seeds") / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    seeds_list = []
    for _, row in seeds_df.iterrows():
        cleaned = {}
        # Extract boundary parameters
        for col in row.index:
            if col.startswith("boundary."):
                key = col.replace("boundary.", "")
                val = clean_matrix(row[col])

                # Skip null/NaN values to avoid writing "null" in JSON
                if val is None:
                    continue
                if isinstance(val, float) and np.isnan(val):
                    continue

                cleaned[key] = val
            elif col in ["vacuum_well", "qi", "aspect_ratio"]: # Optional: Keep some metrics for reference
                cleaned[f"meta_{col}"] = float(row[col])
        
        # Ensure vital keys exist
        if "r_cos" in cleaned and "z_sin" in cleaned:
            # Fill defaults if missing
            if "n_field_periods" not in cleaned:
                 cleaned["n_field_periods"] = int(row.get("boundary.n_field_periods", 1))
            seeds_list.append(cleaned)
            
    with open(output_path, "w") as f:
        json.dump(seeds_list, f, indent=2)
    print(f"Saved {len(seeds_list)} seeds to {output_path}")

def select_best_seeds(df):
    """Filter dataset for Best-of-Failure seeds for P1, P2, P3."""
    print("4. Generating 'Best-of-Failure' Seeds...")
    
    # Ensure columns exist (handling potential missing ones gracefully)
    def has_cols(cols):
        return all(c in df.columns for c in cols)

    # P1: Geometric Optimization
    # Target: A <= 4.0, delta <= -0.5, iota >= 0.3. Minimize Elongation.
    # Relaxed filters to ensure candidates
    if has_cols(["aspect_ratio", "average_triangularity", "edge_rotational_transform_over_n_field_periods", "max_elongation"]):
        p1_mask = (
            (df["aspect_ratio"] <= 4.5) & 
            (df["average_triangularity"] <= -0.4) &
            (df["edge_rotational_transform_over_n_field_periods"] >= 0.25)
        )
        p1_candidates = df[p1_mask].sort_values("max_elongation").head(20)
        
        if len(p1_candidates) > 0:
            save_seeds(p1_candidates, "p1_seeds.json")
        else:
            print("No strict P1 candidates found. Using 'Best-of-Failure' (closest to geometry targets).")
            # Fallback: Minimize distance to target A=4.0, delta=-0.5, iota=0.3
            # Distance = |A - 4| + |delta - (-0.5)| + |iota - 0.3|
            
            dist = (
                (df["aspect_ratio"] - 4.0).abs() + 
                (df["average_triangularity"] - (-0.5)).abs() +
                (df["edge_rotational_transform_over_n_field_periods"] - 0.3).abs()
            )
            
            # Take top 20 closest
            p1_fallback = df.iloc[dist.argsort()].head(20)
            save_seeds(p1_fallback, "p1_seeds.json")
    
    # P2: Simple QI
    # Constraints: A <= 10, iota >= 0.25, M <= 0.2, E <= 5. Minimize QI.
    cols_p2 = ["aspect_ratio", "edge_rotational_transform_over_n_field_periods", 
               "edge_magnetic_mirror_ratio", "max_elongation", "qi"]
    if has_cols(cols_p2):
        p2_mask = (
            (df["aspect_ratio"] <= 10.0) &
            (df["edge_rotational_transform_over_n_field_periods"] >= 0.25) &
            (df["edge_magnetic_mirror_ratio"] <= 0.2) &
            (df["max_elongation"] <= 5.0)
        )
        p2_candidates = df[p2_mask].sort_values("qi").head(20)
        if len(p2_candidates) > 0:
            save_seeds(p2_candidates, "p2_seeds.json")
        else:
             print("No valid P2 candidates found. Skipping p2_seeds.json.")

    # P3: MHD-Stable QI
    # Constraints: W >= 0, C <= 0.9, QI <= 1e-3.5. Obj: Low A, High L_grad.
    fc_col = "flux_compression_in_regions_of_bad_curvature"
    grad_col = "minimum_normalized_magnetic_gradient_scale_length"
    
    if has_cols(["vacuum_well", "qi", "aspect_ratio", grad_col]):
        # Base physics filter
        p3_mask = (df["vacuum_well"] >= 0) & (df["qi"] <= 10**-3.5)
        
        if fc_col in df.columns:
            p3_mask = p3_mask & (df[fc_col] <= 0.9)
        else:
            print(f"Warning: {fc_col} not found. P3 seeds might violate flux constraint.")

        p3_valid = df[p3_mask]
        
        if len(p3_valid) > 0:
            # Pareto Approximation: Top compact + Top simple
            p3_compact = p3_valid.sort_values("aspect_ratio").head(10)
            p3_simple = p3_valid.sort_values(grad_col, ascending=False).head(10)
            p3_candidates = pd.concat([p3_compact, p3_simple]).drop_duplicates().head(20)
            save_seeds(p3_candidates, "p3_seeds.json")
        else:
            print("No strictly valid P3 seeds found. Using 'Best-of-Failure' (closest to well).")
            # Fallback: best available well (even if slightly negative)
            p3_fallback = df.sort_values("vacuum_well", ascending=False).head(20)
            save_seeds(p3_fallback, "p3_seeds.json")

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Offline Data Pipeline for AI Scientist V2")
    parser.add_argument("--output_dir", type=str, default="checkpoints/v2_1", help="Directory to save artifacts")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--only-seeds", action="store_true", help="Skip training and only generate seeds")
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
    
    if not args.only_seeds:
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

        print("4. Training Diffusion Design Model...")
        # Prepare candidates for diffusion (needs 'params' key)
        diffusion_candidates = []
        for _, row in df_train.iterrows():
            try:
                r_cos = clean_matrix(row["boundary.r_cos"])
                z_sin = clean_matrix(row["boundary.z_sin"])
                params = {
                    "r_cos": r_cos,
                    "z_sin": z_sin,
                    "n_field_periods": row.get("boundary.n_field_periods", 1),
                }
                # Metrics for conditioning
                metrics = {
                    "aspect_ratio": row.get("aspect_ratio", 0.0),
                    "minimum_normalized_magnetic_gradient_scale_length": row.get("minimum_normalized_magnetic_gradient_scale_length", 0.0),
                    "max_elongation": row.get("max_elongation", 0.0),
                    "edge_rotational_transform_over_n_field_periods": row.get("edge_rotational_transform_over_n_field_periods", 0.0),
                }
                diffusion_candidates.append({"params": params, "metrics": metrics})
            except:
                continue
        
        if diffusion_candidates:
            diffusion_model = DiffusionDesignModel(
                epochs=args.epochs,
                batch_size=32,
                learning_rate=1e-4,
                device=device
            )
            diffusion_model.fit(diffusion_candidates)
            
            diff_path = checkpoints_dir / "diffusion_v2.pt"
            # DiffusionDesignModel exposes state_dict() that includes schema, normalizers, and weights
            if diffusion_model._trained:
                checkpoint = diffusion_model.state_dict()
                torch.save(checkpoint, diff_path)
                print(f"Diffusion model saved to {diff_path}")
            else:
                print("Diffusion model training did not complete; skipping checkpoint.")
        else:
            print("No valid candidates for diffusion training.")
    else:
        print("Skipping training (--only-seeds provided).")
            
    # 4. Generate Best-of-Failure Seeds
    select_best_seeds(df)

    print("Pipeline complete.")

if __name__ == "__main__":
    main()
