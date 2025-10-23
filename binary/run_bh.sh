#!/bin/bash

job_id="080725"
# Number of total K-points to compute (should match N_K in your Python script)
N_K=401

# Directory to store results
mkdir -p results

# For OpenBLAS
export OPENBLAS_NUM_THREADS=1

# For MKL (Intel)
export MKL_NUM_THREADS=1

# For Apple's Accelerate (macOS)
export VECLIB_MAXIMUM_THREADS=1

# Also sometimes used by SciPy
export NUMEXPR_NUM_THREADS=1

# General fallback
export OMP_NUM_THREADS=1

export NUMBA_NUM_THREADS=1


# Run in parallel and display a progress bar
parallel --bar -j 6 "python kbin.py ${job_id} {} $N_K" ::: $(seq 0 $((N_K - 1)))

# Combine the results into a single file
python - <<EOF
import numpy as np
import os

N_K = $N_K
job_id = "$job_id"  # Grab job_id from the shell script
result_list = []

for task_id in range(N_K):
    fname = f"results/{job_id}_result_{task_id}.npy"
    if os.path.exists(fname):
        result = np.load(fname)
        result_list.append(result)
    else:
        print(f"Warning: {fname} not found.")

if result_list:
    results = np.vstack(result_list)
    np.save(f"{job_id}_bands_combined.npy", results)
    np.savetxt(f"{job_id}_bands_combined.txt", results, fmt="%.8f")
    print(f"Aggregated results saved as '{job_id}_bands_combined.npy' and '{job_id}_bands_combined.txt'")
else:
    print("No results were aggregated.")
EOF
