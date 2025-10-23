import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import os
import numpy as np
import matplotlib.font_manager as fm
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'
mpl.rcParams['axes.unicode_minus'] = False

def aggregate_and_plot(job_id, N_K, U, phi):


    # --- Parameters that must match the worker script ---
    num_bands_to_find = 4
    
    K_values = np.linspace(-np.pi, np.pi, N_K)
    
    # --- Pre-allocate arrays to hold the collected data ---
    continuum_min = np.full(N_K, np.nan)
    continuum_max = np.full(N_K, np.nan)
    all_bands_raw = np.full((num_bands_to_find, N_K), np.nan)

    # --- Load results ---
    print(f"Aggregating results from {N_K} files...")
    results_dir = "results"
    
    for i in range(N_K):
        filepath = os.path.join(results_dir, f"{job_id}_result_{i}.npy")
        try:
            data = np.load(filepath)
            continuum_min[i] = data[0]
            continuum_max[i] = data[1]
            all_bands_raw[:, i] = data[2:]
        except FileNotFoundError:
            print(f"Warning: Result file not found for task {i}. Skipping.")

    # Filter out missing bands
    all_bound_state_bands = [band for band in all_bands_raw if not np.all(np.isnan(band))]

    # --- Plotting ---
    print(f"Plotting results. Found {len(all_bound_state_bands)} bound state band(s).")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True, 
                                   gridspec_kw={'height_ratios': [2, 1]})
    
    # Colors for bands
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # --- Top panel: energy bands ---
    ax1.fill_between(K_values / np.pi, continuum_min, continuum_max, 
                     color='gray', alpha=0.3, label='Three-Particle Continuum')

    for i, band_energies in enumerate(all_bound_state_bands):
        ax1.plot(K_values / np.pi, band_energies, 'o-', color=colors[i % len(colors)],
                 linewidth=2, markersize=4, label=f'Trimer State Band #{i+1}')
    
    fs = 30
    ax1.set_ylabel('Energy $(E)$', fontsize=fs, fontname="Times New Roman")
    ax1.set_title(fr'Three-Particle Dispersion for $U={U}, \phi={phi:.2f}\pi$', fontsize=fs, fontname="Times New Roman")
    font_properties = fm.FontProperties(family='serif', weight='bold', size=14, style='italic')
    ax1.legend(fontsize=(fs - 5),prop=font_properties)
    ax1.set_xlim(-1, 1)
    ax1.tick_params(axis='both', which='major', labelsize=14, length=3, width=1.5)

    # --- Bottom panel: dE/dK ---
    for i, band_energies in enumerate(all_bound_state_bands):
        dEdK = np.gradient(band_energies, K_values)  # numerical derivative
        ax2.plot(K_values / np.pi, dEdK, '-', color=colors[i % len(colors)], linewidth=2)

    ax2.axhline(0, color='black', linewidth=2, linestyle='--', alpha=0.6)
    ax2.set_xlabel(r'Center-of-Mass Momentum ($K/\pi$)', fontsize=fs, fontname="Times New Roman")
    ax2.set_ylabel(r'$dE/dK$', fontsize=fs)
    ax2.tick_params(axis='both', which='major', labelsize=12, length=3, width=1.5)
    
    plt.tight_layout()
    plt.savefig(f"NDSEG_final_bands_with_dEdK_{U}U.pdf")
    print(f"Plot saved to NDSEG_final_bands_with_dEdK_{U}U.pdf")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate and plot three-particle band structure data.")
    parser.add_argument("job_id", type=str, help="job_id")
    parser.add_argument("N_K", type=int, help="Number of K-points")
    parser.add_argument("U", type=float, help="Interaction strength")
    parser.add_argument("phi", type=float, help="Flux value (in radians)")
    args = parser.parse_args()

    aggregate_and_plot(args.job_id, args.N_K, args.U, args.phi)
