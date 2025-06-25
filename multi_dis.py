import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from functools import partial
import warnings

# Suppress warnings from root finding when a bound state doesn't exist
warnings.filterwarnings('ignore', 'b_less_than_a', UserWarning)

# --- 1. Single-Particle Properties (Vectorized) ---

def H0(k, phi):
    """Vectorized single-particle momentum-space Hamiltonian H(k)."""
    k = np.atleast_1d(k)
    H = np.zeros((k.shape[0], 3, 3), dtype=complex)
    h12 = np.exp(1j * phi) + np.exp(1j * k)
    h13 = 1 + np.exp(1j * k)
    H[:, 0, 1] = h12
    H[:, 1, 0] = np.conj(h12)
    H[:, 0, 2] = h13
    H[:, 2, 0] = np.conj(h13)
    return H[0] if H.shape[0] == 1 else H

def get_single_particle_bands(k_values, phi):
    """Calculate the 3 single-particle energy bands."""
    energies = np.linalg.eigvalsh(H0(k_values, phi))
    return energies.T

def get_continuum_bounds(K, phi, num_q=101):
    """Calculate the two-particle scattering continuum bounds for a given K."""
    q_values = np.linspace(-np.pi, np.pi, num_q)
    k1 = K/2 + q_values
    k2 = K/2 - q_values
    eigs1 = np.linalg.eigvalsh(H0(k1, phi))
    eigs2 = np.linalg.eigvalsh(H0(k2, phi))
    continuum_energies = (eigs1[:, :, np.newaxis] + eigs2[:, np.newaxis, :]).flatten()
    return np.min(continuum_energies), np.max(continuum_energies)

# --- 2. Two-Particle Formalism (Unchanged) ---

def get_real_space_matrices(phi):
    H_intra = np.array([[0, np.exp(1j*phi), 1], [np.exp(-1j*phi), 0, 0], [1, 0, 0]], dtype=complex)
    H_inter = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=complex)
    H_inter_minus = H_inter.T.conj()
    return H_intra, H_inter, H_inter_minus

def get_T_eff_matrices(K, H_intra, H_inter, H_inter_minus):
    """
    Constructs the effective 9x9 hopping matrices for the relative coordinate.
    This version is based on the definitive derivation using r = pos_1 - pos_2.
    """
    I = np.eye(3, dtype=complex)
    
    # --- T0: Hopping within the same relative coordinate 'r' ---
    # This represents H(r,r) in the Schrodinger equation.
    # It is derived from H_kin = (H_1+H_2) = (H_intra ⊗ I) + (I ⊗ H_intra).
    T0 = np.kron(H_intra, I) + np.kron(I, H_intra)

    # --- T_plus_1: Hopping that increases r by 1 (e.g., r-1 -> r) ---
    # This is the matrix H(r, r-1) that multiplies psi(r-1).
    # It corresponds to a term exp(+ik) in the Fourier domain.
    # Caused by: (p1 hops forward) OR (p2 hops backward).
    T_plus_1 = (np.kron(H_inter, I) * np.exp(1j*K/2)) + \
               (np.kron(I, H_inter_minus) * np.exp(-1j*K/2))

    # --- T_minus_1: Hopping that decreases r by 1 (e.g., r+1 -> r) ---
    # This is the matrix H(r, r+1) that multiplies psi(r+1).
    # It corresponds to a term exp(-ik) in the Fourier domain.
    # Caused by: (p1 hops backward) OR (p2 hops forward).
    T_minus_1 = (np.kron(H_inter_minus, I) * np.exp(-1j*K/2)) + \
                (np.kron(I, H_inter) * np.exp(1j*K/2))

    # The Green's function calculation uses the Fourier transformed kinetic Hamiltonian:
    # H_kin(K,q) = T0 + T_plus_1 * exp(iq) + T_minus_1 * exp(-iq)
    # The returned matrices correspond to these terms.
    return T0, T_plus_1, T_minus_1

# --- 3. Green's Function and Root Finding for Multiple Bands ---

def calculate_G00(E, T0, T1, T_minus_1, num_q=201):
    """Calculates the on-site (r=0) Green's function matrix G(0,0)."""
    q_values = np.linspace(-np.pi, np.pi, num_q)
    G00 = np.zeros((9, 9), dtype=complex)
    I9 = np.eye(9)
    for q in q_values:
        #some sort of plan wave ansatz
        H_kin_K_q = T0 + T1 * np.exp(1j*q) + T_minus_1 * np.exp(-1j*q)
        try:
            #the resolvent
            G00 += np.linalg.inv(E * I9 - H_kin_K_q)
        except np.linalg.LinAlgError:
            return np.full((9, 9), np.inf)
    return G00 / num_q

# ****************** NEW ROOT EQUATION FOR MULTIPLE BANDS ******************
def eigenvalue_root_equation(E, U, T0, T1, T_minus_1, eigenvalue_index):
    """
    Root function based on the eigenvalue condition: lambda_i(E) - 1/U = 0.
    """
    if np.isinf(E) or np.isnan(E) or U == 0:
        return np.inf

    G00 = calculate_G00(E, T0, T1, T_minus_1)
    if np.any(np.isinf(G00)):
        return np.inf
        
    interacting_indices = [0, 4, 8]
    G_reduced = G00[np.ix_(interacting_indices, interacting_indices)]

    try:
        # We use eigvalsh because G_reduced should be Hermitian
        lambdas = np.linalg.eigvalsh(G_reduced)
        # Sort eigenvalues to have a consistent index
        lambdas = np.sort(lambdas)
        
        # The equation to solve
        return lambdas[eigenvalue_index] - (1.0 / U)
        
    except np.linalg.LinAlgError:
        return np.inf
# **************************************************************************


# --- 4. Main Loop and Plotting ---

def main():
    # --- Parameters ---
    U = .2  # Interaction strength. Larger |U| often reveals more bands.
    phi = np.pi
    N_K = 251 # Number of COM momentum points
    
    K_values = np.linspace(-np.pi, np.pi, N_K)
    
    # We now need a place to store multiple bands
    all_bound_state_bands = []
    num_bands_to_find = 3

    # --- Pre-calculate real-space matrices ---
    H_intra, H_inter, H_inter_minus = get_real_space_matrices(phi)
    
    print("Calculating continuum...")
    continuum_min = np.zeros(N_K)
    continuum_max = np.zeros(N_K)
    for i, K in enumerate(K_values):
        continuum_min[i], continuum_max[i] = get_continuum_bounds(K, phi)

    print("Searching for bound state bands...")
    # Loop over each potential band
    for band_idx in range(num_bands_to_find):
        print(f"--- Searching for band #{band_idx + 1} ---")
        current_band_energies = np.full(N_K, np.nan)
        
        # Loop over each K point
        for i, K in enumerate(K_values):
            # Get effective matrices for this K
            T0, T1, T_minus_1 = get_T_eff_matrices(K, H_intra, H_inter, H_inter_minus)
            
            # Define search range. Widen it slightly to catch all bands.
            if U < 0:
                search_min = continuum_min[i] - 4 * abs(U)
                search_max = continuum_min[i] - 1e-6
            else:
                search_min = continuum_max[i] + 1e-6
                search_max = continuum_max[i] + 4 * abs(U)

            # ******************** FIX IS HERE ********************
            # Use the correct keyword argument 'T_minus_1'
            f_to_solve = partial(eigenvalue_root_equation, U=U, T0=T0, T1=T1, T_minus_1=T_minus_1,
                                 eigenvalue_index=band_idx)
            # *****************************************************
            
            try:
                # The bracket might fail if the function doesn't cross zero.
                # We check the signs at the boundaries.
                val_min = f_to_solve(search_min)
                val_max = f_to_solve(search_max)
                if np.isfinite(val_min) and np.isfinite(val_max) and np.sign(val_min) != np.sign(val_max):
                    sol = root_scalar(f_to_solve, bracket=[search_min, search_max], method='brentq')
                    if sol.converged:
                        current_band_energies[i] = sol.root
            except (ValueError, RuntimeError):
                 # This can happen if the root is not in the bracket or other issues.
                 pass
        
        # Only add the band if we actually found any energies for it
        if not np.all(np.isnan(current_band_energies)):
            all_bound_state_bands.append(current_band_energies)

    # --- Plotting ---
    print(f"Plotting results. Found {len(all_bound_state_bands)} bound state band(s).")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot the continuum
    ax.fill_between(K_values / np.pi, continuum_min, continuum_max, 
                    color='gray', alpha=0.3, label='Two-Particle Continuum')
    
    # Plot the single-particle bands for reference
    k_plot = np.linspace(-np.pi, np.pi, 201)
    sp_bands = get_single_particle_bands(k_plot, phi)
    for band in sp_bands:
        ax.plot(k_plot / np.pi, band, 'k--', linewidth=0.5, alpha=0.5, label='_nolegend_')
        
    # Plot all the found bound state bands
    colors = ['blue', 'red', 'green']
    for i, band_energies in enumerate(all_bound_state_bands):
        ax.plot(K_values / np.pi, band_energies, '-', color=colors[i % len(colors)],
                linewidth=2.5, label=f'Dimer Band #{i+1}')
    
    ax.set_xlabel(r'Center-of-Mass Momentum $K/\pi$')
    ax.set_ylabel('Energy $E$')
    ax.set_title(fr'Two-Particle Dispersion for $U={U}, \phi={phi/np.pi:.2f}\pi$')
    ax.legend()
    ax.set_xlim(-1, 1)
    
    plt.savefig(f"{U}U0phibands.pdf")
    plt.show()

if __name__ == '__main__':
    main()