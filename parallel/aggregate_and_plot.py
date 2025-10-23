# FILE: aggregate_and_plot.py

import matplotlib.pyplot as plt
import argparse

import os

import numpy as np

def aggregate_and_plot(job_id, N_K, U, phi):
    # --- Parameters that must match the worker script ---
    num_bands_to_find = 4
    
    K_values = np.linspace(-np.pi, np.pi, N_K)
    #K_values = np.linspace(-np.pi/8, np.pi/8, N_K)

    
    # --- Pre-allocate arrays to hold the collected data ---
    continuum_min = np.full(N_K, np.nan)
    continuum_max = np.full(N_K, np.nan)
    # Shape: (num_bands, num_k_points)
    all_bands_raw = np.full((num_bands_to_find, N_K), np.nan)

    # --- Loop through task IDs and load results ---
    print(f"Aggregating results from {N_K} files...")
    results_dir = "results"
    
    for i in range(N_K):
        filepath = os.path.join(results_dir, f"{job_id}_result_{i}.npy")
        try:
            # Load the data: [c_min, c_max, E_band0, E_band1, ...]
            data = np.load(filepath)
            continuum_min[i] = data[0]
            continuum_max[i] = data[1]
            all_bands_raw[:, i] = data[2:]
        except FileNotFoundError:
            print(f"Warning: Result file not found for task {i}. Skipping.")

    # Filter out bands that were not found (all NaNs)
    all_bound_state_bands = [band for band in all_bands_raw if not np.all(np.isnan(band))]

    # --- Plotting ---
    print(f"Plotting results. Found {len(all_bound_state_bands)} bound state band(s).")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.fill_between(K_values / np.pi, continuum_min, continuum_max, 
                    color='gray', alpha=0.3, label='Three-Particle Continuum')

    # Plotting code here is the same as your original script...
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, band_energies in enumerate(all_bound_state_bands):
        ax.plot(K_values / np.pi, band_energies, 'o-', color=colors[i % len(colors)],
                linewidth=2, markersize=4, label=f'Trimer State Band #{i+1}')
    
    fs=25
    ax.set_xlabel(r'Center-of-Mass Momentum ($K/\pi$)',fontsize=fs)
    ax.set_ylabel('Energy $(E)$',fontsize=fs)
    ax.set_title(fr'Three-Particle Dispersion for $U={U}, \phi={phi:.2f}\pi$',fontsize=fs)
    ax.legend(fontsize=fs-10)
    ax.set_xlim(-1, 1)
    
    plt.savefig(f"NDSEG_final_bands_{U}U.pdf")
    print("Plot saved to final_bands.pdf")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate and plot three-particle band structure data.")
    parser.add_argument("job_id", type=str, help="job_id")
    parser.add_argument("N_K", type=int, help="Number of K-points")
    parser.add_argument("U", type=float, help="Interaction strength")
    parser.add_argument("phi", type=float, help="Flux value (in radians)")
    args = parser.parse_args()

    aggregate_and_plot(args.job_id, args.N_K, args.U, args.phi)
