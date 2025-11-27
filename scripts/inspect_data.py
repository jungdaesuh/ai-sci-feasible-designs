import sys
import os
import numpy as np
import pandas as pd
from ai_scientist.optim.data_loader import load_constellaration_dataset

sys.path.append(os.getcwd())

def main():
    print("Loading data...")
    try:
        df = load_constellaration_dataset(filter_geometry=True)
    except Exception as e:
        print(f"Error: {e}")
        return

    if len(df) > 0 and 'boundary.r_cos' in df.columns:
        val = df.iloc[0]['boundary.r_cos']
        print(f"Type of boundary.r_cos: {type(val)}")
        print(f"Value sample: {str(val)[:100]}")
        
        try:
            arr = np.asarray(val, dtype=float)
            print(f"Converted to numpy array shape: {arr.shape}")
        except Exception as e:
            print(f"Numpy conversion failed: {e}")
            if isinstance(val, list):
                 print(f"Length of outer list: {len(val)}")
                 if len(val) > 0:
                     print(f"Type of first element: {type(val[0])}")
                     if isinstance(val[0], list):
                         print(f"Lengths of inner lists: {[len(x) for x in val]}")

if __name__ == "__main__":
    main()
