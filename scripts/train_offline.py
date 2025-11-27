import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
from ai_scientist.optim.data_loader import load_constellaration_dataset, LogRobustScaler, save_scaler

# Define Physics Columns to Scale
# These are the targets the Neural Surrogate will learn.
# Geometry metrics (elongation, aspect_ratio) are calculated analytically.
PHYSICS_COLS = [
    "qi",
    "edge_magnetic_mirror_ratio",
    "minimum_normalized_magnetic_gradient_scale_length",
    "edge_rotational_transform_over_n_field_periods", # iota
    # Add others as discovered in dataset
]

def main():
    parser = argparse.ArgumentParser(description="Offline Data Pipeline for AI Scientist V2")
    parser.add_argument("--output_dir", type=str, default="checkpoints/v2_1", help="Directory to save artifacts")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("1. Loading and Cleaning Data...")
    try:
        df = load_constellaration_dataset(filter_geometry=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Loaded {len(df)} valid samples.")
    
    # Identify columns present
    available_physics_cols = [c for c in PHYSICS_COLS if c in df.columns]
    
    # Check for 'well' related columns (vacuum well)
    well_cols = [c for c in df.columns if 'well' in c.lower() or 'magnetic_well' in c.lower()]
    available_physics_cols.extend(well_cols)
    
    # Remove duplicates
    available_physics_cols = list(set(available_physics_cols))
    
    print(f"Physics columns selected for scaling: {available_physics_cols}")
    
    if not available_physics_cols:
        print("Error: No physics columns found to scale.")
        return

    # Filter out rows with NaNs in these specific physics columns
    initial_count = len(df)
    df = df.dropna(subset=available_physics_cols)
    if len(df) < initial_count:
        print(f"Dropped {initial_count - len(df)} rows with missing physics values.")

    print("2. Fitting LogRobustScaler...")
    scaler = LogRobustScaler()
    # Ensure we pass a 2D array
    X_physics = df[available_physics_cols].values
    scaler.fit(X_physics)
    
    scaler_path = output_dir / "scaler.pkl"
    save_scaler(scaler, str(scaler_path))
    print(f"Scaler saved to {scaler_path}")
    
    # Save the column names so we know what the scaler expects
    cols_path = output_dir / "physics_cols.txt"
    with open(cols_path, "w") as f:
        for col in available_physics_cols:
            f.write(f"{col}\n")
            
    print("Pipeline complete.")

if __name__ == "__main__":
    main()
