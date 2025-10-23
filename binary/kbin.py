import sys
import os
from scipy.optimize import root_scalar
from functools import partial
import warnings

import numpy as np
np.set_printoptions(linewidth=np.inf, precision=2)
# Suppress warnings from root finding when a bound state doesn't exist
warnings.filterwarnings('ignore', 'b_less_than_a', UserWarning)

# --- global sizes -------------------------------------------------
N_SUB = 3                    # sites in a unit cell
DIM   = N_SUB**2              # 9 internal states

I3  = np.eye(N_SUB, dtype=complex)
I9 = np.eye(DIM,    dtype=complex)

def add_conj(mat):
    return mat + mat.conj().T

# --- 1. Single-Particle Properties (Vectorized) ---

def H0(k, phi):
    """
    single-particle momentum-space Hamiltonian H(k).
    Accepts either a scalar k or a 1D array of k's.
    Returns shape (3,3) for scalar input, or (N_k,3,3) for array input.
    """
    k_arr = np.atleast_1d(k).flatten()
    N = k_arr.size

    H = np.zeros((N, 3, 3), dtype=complex)
    for i, ki in enumerate(k_arr):
        H_i = np.zeros((3, 3), dtype=complex)
        # use the i-th momentum value everywhere
        H_i[0, 1] = np.exp(1j * phi) + np.exp(1j * ki)
        H_i[0, 2] = 1 + np.exp(1j * ki)
        H[i] = add_conj(H_i)

    # if the user passed in a single k, collapse the first axis
    return H[0] if N == 1 else H

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

def get_real_space_matrices(phi):
    H_intra = np.array([[0, np.exp(1j*phi), 1], [np.exp(-1j*phi), 0, 0], [1, 0, 0]], dtype=complex)
    H_inter = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=complex)
    H_inter_minus = H_inter.T.conj()
    return H_intra, H_inter, H_inter_minus

def get_T_eff_matrices(K, H_intra, H_inter, H_inter_minus):
    """
    Constructs the effective 9x9hopping matrices for the relative coordinate.
    """
    T_dict = {}
    
    T_dict[0] = np.kron(H_intra, I3) + np.kron(I3, H_intra)

    T_dict[1] = np.kron(H_inter, I3) * np.exp(1j*K/2)
    T_dict[-1] = np.kron(H_inter_minus, I3) * np.exp(-1j*K/2)

    T_dict[2] = np.kron(I3, H_inter) * np.exp(1j*K/2)
    T_dict[-2] = np.kron(I3, H_inter_minus) * np.exp(-1j*K/2)

    return T_dict

# --- 3. Green's Function and Root Finding for Multiple Bands ---


class GreenFunctionSolver:
    """
    A class to efficiently solve for the Green's function G00(E).
    
    The key idea is to pre-calculate the E-independent parts of the calculation.
    We diagonalize the kinetic Hamiltonian H_kin(q1, q2) over the full grid of
    relative momenta (q1, q2) *once* upon initialization.
    
    Then, calculating G00(E) for any energy E becomes a fast summation
    using the pre-calculated eigenvalues and eigenvectors, avoiding costly
    matrix inversions inside the root-finding loop.
    """
    def __init__(self, T, q_grid_size=51):
        """
        Pre-calculates eigenvalues and eigenvectors of H_kin(q1, q2).
        This is the expensive, one-time setup step.
        """
        q_pts = np.linspace(-np.pi, np.pi, q_grid_size)
       
        # Build H_kin(K,q1) for the whole grid
        H_stack = (
            T[0][None, :, :]                                      # intra‑cell term
          + T[1] * np.exp(1j * q_pts)[:, None, None]              # p1 hopping
          + T[-1] * np.exp(-1j * q_pts)[:, None, None]            # p1 backwards
          + T[2] * np.exp(-1j * q_pts)[:, None, None]             # p2 hopping
          + T[-2] * np.exp(1j * q_pts)[:, None, None]             # p2 backwards
        )
        
        # Reshape for vectorized diagonalization
        H_stack = H_stack.reshape(-1, DIM, DIM) #

        # The expensive step: diagonalize the entire stack of Hamiltonians.
        # This is done only ONCE per K-point.
        # eigh is used because the Hamiltonian is Hermitian.
        self.evals, self.evecs = np.linalg.eigh(H_stack)

    def calculate_G00(self, E):
        """
        Calculates G00(E) using the pre-computed eigenvalues and eigenvectors.
        This method is fast and is called repeatedly by the root finder.
        """
        # Calculate 1 / (E - lambda_n) for all eigenvalues at once.
        # Add a small epsilon to E to avoid division by zero if E is exactly
        # on the continuum.
        E_shifted = E + 1e-12j 
        propagators = 1.0 / (E_shifted - self.evals)

        # Reconstruct the Green's function from eigen-decomposition:
        # G = U * diag(propagators) * U_dagger
        # We use np.einsum for a fast, memory-efficient implementation of
        # this batched matrix multiplication.
        # 'n...ij' = sum_k U_ik * d_k * U_jk^*
        G_stack = np.einsum('nij,nj,nkj->nik', 
                            self.evecs, propagators, self.evecs.conj())
        
        # Average over the (q1, q2) grid to get the final G00
        G00 = np.mean(G_stack, axis=0)
        return G00

_INTERACT = [0, 4, 8] 

def eigenvalue_root_equation_full(E, U, solver, eigen_idx):
    """
    Solve λ_n(E) = 1/U for the nᵗʰ two‑particle bound band by
    diagonalizing the *entire* G00 matrix (no sub-block reduction).
    """
    # 1) guard against bad E or U=0
    if U == 0 or np.isinf(E) or np.isnan(E):
        return np.inf

    # 2) compute the full 9×9 Green's function at energy E
    G00_full = solver.calculate_G00(E)                    # (9,9)
    G00_red  = G00_full[np.ix_(_INTERACT,_INTERACT)]      # (3,3)
    lamb     = np.linalg.eigvalsh(G00_red)
    return lamb[eigen_idx] - (1.0/U)



def run_calculation_for_task(job_id, task_id, N_K):
    U = 10             # your interaction strength
    phi = 0        # example flux
    q_grid_size = 801  # momentum‐grid resolution

    # 1) pick the total‐momentum point
    K_values = np.linspace(-np.pi, np.pi, N_K)
    K = K_values[task_id]
    print(f"Task {task_id}: K = {K:.4f}")

    # 2) build solver as before
    H_intra, H_inter, H_inter_minus = get_real_space_matrices(phi)
    T = get_T_eff_matrices(K, H_intra, H_inter, H_inter_minus)
    c_min, c_max = get_continuum_bounds(K, phi, num_q=51)
    solver = GreenFunctionSolver(T, q_grid_size=q_grid_size)

    # 3) we'll look for N_SUB=3 bound bands
    energies = np.full(N_SUB, np.nan)

    for n in range(N_SUB):
        f = partial(eigenvalue_root_equation_full,
                    U=U,
                    solver=solver,
                    eigen_idx=n)

        # bracket selection: for U>0 (repulsive) bound above continuum
        bracket = (c_max + 1e-6, c_max + 10*abs(U))

        try:
            v0, v1 = f(bracket[0]), f(bracket[1])
            if np.sign(v0) != np.sign(v1):
                sol = root_scalar(f, bracket=bracket, method='brentq')
                if sol.converged:
                    energies[n] = sol.root
        except Exception:
            pass

    # 4) save exactly as before
    result = np.concatenate(([c_min, c_max], energies))
    outdir = "results"; os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, f"{job_id}_result_{task_id}.npy")
    np.save(fname, result)
    print(f"Saved {fname}")





if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python kbin.py <job_id> <task_id> <N_K>")
        sys.exit(1)

    try:
        job_id = sys.argv[1]
        task_id = int(sys.argv[2])
        N_K = int(sys.argv[3])
        run_calculation_for_task(job_id, task_id, N_K)
    except ValueError:
        print("Error: Arguments must be integers.")
        sys.exit(1)
