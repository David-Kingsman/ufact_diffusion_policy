import numpy as np
import os

folder = "./demos_collected/run001"
for fname in os.listdir(folder):
    if fname.endswith(".npz"):
        print(f"\nChecking file: {fname}")
        arr = np.load(os.path.join(folder, fname), allow_pickle=True)["data"]
        print("shape:", arr.shape)
        if arr.size == 0:
            print("Warning: This file is empty!")
        elif np.all(arr == 0):
            print("Warning: All data in this file are zeros!")
        elif np.issubdtype(arr.dtype, np.number) and np.isnan(arr).any():
            print("Warning: This file contains NaN values!")
        else:
            # Check if any row is all zeros (only for numeric arrays)
            if np.issubdtype(arr.dtype, np.number) and arr.ndim > 1:
                zero_rows = np.where(~arr.any(axis=-1))[0]
                if len(zero_rows) > 0:
                    print(f"{len(zero_rows)} rows are all zeros (example indices: {zero_rows[:10]})")
                else:
                    print("Data is normal, no all-zero rows found.")
            else:
                print("Data type is object or 1D, skipping all-zero row check.")
        # Optional: print first 2 rows of data
        print("First 2 rows of data:\n", arr[:2])