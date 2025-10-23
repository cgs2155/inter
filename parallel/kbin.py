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
N_SUB = 4                     # sites in a unit cell
DIM   = N_SUB**3              # 64 internal states

I4  = np.eye(N_SUB, dtype=complex)
I64 = np.eye(DIM,    dtype=complex)

def add_conj(mat):
    return mat + mat.conj().T

# --- 1. Single-Particle Properties (Vectorized) ---

def H0(k, phi):
    """single-particle momentum-space Hamiltonian H(k)."""
    k = np.atleast_1d(k).flatten()
    N = k.shape[0]

    H = np.zeros((k.shape[0], 4, 4), dtype=complex)
    for i in range(N):
        H_i = np.zeros((4, 4), dtype=complex)
        H_i[0, 1] = np.exp(-1j * phi/2) + np.exp(1j * (-k[i] + phi/2))
        H_i[0, 2] = 1 + np.exp(-1j * k[i])
        H_i[0, 3] = np.exp(1j * phi/2) + np.exp(-1j * (k[i] + phi/2))
        H[i] = add_conj(H_i)
    
    return H[0] if N == 1 else H

def get_single_particle_bands(k_values, phi):
    """Calculate the 4 single-particle energy bands."""
    energies = np.linalg.eigvalsh(H0(k_values, phi))
    return energies.T

def get_continuum_bounds(K, phi, num_q=101):
    """Calculate the two-particle scattering continuum bounds for a given K."""
    q_values = np.linspace(-np.pi, np.pi, num_q)
    p_values = np.linspace(-np.pi, np.pi, num_q)

    Q, P = np.meshgrid(q_values, p_values, indexing='ij')

    k1 = K/3 + Q
    k2 = K/3 - Q + P
    k3 = K/3 - P

    eigs1 = np.linalg.eigvalsh(H0(k1, phi))
    eigs2 = np.linalg.eigvalsh(H0(k2, phi))
    eigs3 = np.linalg.eigvalsh(H0(k3, phi))

    continuum_energies = (
        eigs1[:, :, np.newaxis, np.newaxis]
        + eigs2[:, np.newaxis, :, np.newaxis]
        + eigs3[:, np.newaxis, np.newaxis, :]
    ).flatten()
    return np.min(continuum_energies), np.max(continuum_energies)

def get_real_space_matrices(phi):
    H_intra = add_conj(np.array([[0,np.exp(-1j*phi/2), 1 , np.exp(1j*phi/2)], [0,0,0,0],[0,0,0,0],[0,0,0,0]] ))
    H_inter_minus = np.array([[0,np.exp(1j*phi/2), 1 , np.exp(-1j*phi/2)], [0,0,0,0],[0,0,0,0],[0,0,0,0]] )
    H_inter = H_inter_minus.T.conj()
    return H_intra, H_inter, H_inter_minus

def get_T_eff_matrices(K, H_intra, H_inter, H_inter_minus):
    """
    Constructs the effective 64x64hopping matrices for the relative coordinate.
    #This version is based on the definitive derivation using r and s.
    """
    T_dict = {}
    
    T_dict[0] = np.kron(np.kron(H_intra, I4),I4) + np.kron(np.kron(I4,H_intra),I4) + np.kron(np.kron(I4,I4),H_intra)

    T_dict[1] = np.kron(np.kron(H_inter, I4),I4)*np.exp(1j*K/3)
    T_dict[-1] =np.kron(np.kron(H_inter_minus, I4),I4)*np.exp(-1j*K/3)

    T_dict[2] = np.kron(np.kron(I4,H_inter),I4)*np.exp(1j*K/3)
    T_dict[-2] =np.kron(np.kron(I4,H_inter_minus),I4)*np.exp(-1j*K/3)

    T_dict[3] = np.kron(np.kron(I4,I4),H_inter)*np.exp(1j*K/3)
    T_dict[-3] =np.kron(np.kron(I4,I4),H_inter_minus)*np.exp(-1j*K/3)
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
    def __init__(self, T, K, q_grid_size=51):
        """
        Pre-calculates eigenvalues and eigenvectors of H_kin(q1, q2).
        This is the expensive, one-time setup step.
        """
        q_pts = np.linspace(-np.pi, np.pi, q_grid_size)
        q1, q2 = np.meshgrid(q_pts, q_pts, indexing='ij')

        # Build H_kin(K,q1,q2) for the whole grid
        H_stack = (
            T[0][np.newaxis, np.newaxis, :, :] +
            T[1] * np.exp(1j * (q1))[:, :, np.newaxis, np.newaxis] +
            T[-1] * np.exp(-1j * (q1))[:, :, np.newaxis, np.newaxis] +
            T[2] * np.exp(1j * (-q1 + q2))[:, :, np.newaxis, np.newaxis] +
            T[-2] * np.exp(-1j * (- q1 + q2))[:, :, np.newaxis, np.newaxis] +
            T[3] * np.exp(1j * (- q2))[:, :, np.newaxis, np.newaxis] +
            T[-3] * np.exp(-1j * (- q2))[:, :, np.newaxis, np.newaxis]
        )
        
        # Reshape for vectorized diagonalization
        H_stack = H_stack.reshape(-1, DIM, DIM) # Shape: (Nq*Nq, 64, 64)

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


def get_trimer_and_dimer_indices(N_SUB):
    """
    Returns two lists:
      - trimer_indices[a] = index of |a,a,a>
      - dimer_sets[a] = list of lists, each sub‑list is the 3 raw indices
                        corresponding to the symmetric state |a,a,b>
    """
    trimer_indices = []
    dimer_sets = []
    for a in range(N_SUB):
        # trimer: (a,a,a)
        tri_idx = a*N_SUB**2 + a*N_SUB + a
        trimer_indices.append(tri_idx)

        # dimer+single: for each b != a, get the three permutations
        sets_for_a = []
        for b in range(N_SUB):
            if b == a:
                continue
            perms = [
                (a, a, b),
                (a, b, a),
                (b, a, a),
            ]
            idx_list = [i*N_SUB**2 + j*N_SUB + k for (i,j,k) in perms]
            sets_for_a.append(idx_list)
        dimer_sets.append(sets_for_a)

    return trimer_indices, dimer_sets



def eigenvalue_root_equation(E, U, solver, trimer_site_index, N_SUB):

    if U == 0 or np.isinf(E) or np.isnan(E):
        return np.inf

    G00 = solver.calculate_G00(E)
    if np.any(np.isinf(G00)):
        return np.inf

    # build indices
    trimer_indices, dimer_sets = get_trimer_and_dimer_indices(N_SUB)
    tri_idx = trimer_indices[trimer_site_index]
    neighbor_sets = dimer_sets[trimer_site_index]

    # dimension of reduced block: 1 trimer + (N_SUB-1) dimer states
    M = 1 + len(neighbor_sets)
    G_red = np.zeros((M, M), dtype=complex)

    # trimer-trimer
    G_red[0, 0] = G00[tri_idx, tri_idx]

    # trimer-dimer (and hermitian conjugate)
    for j, idx_list in enumerate(neighbor_sets, start=1):
        # average over the three raw indices, normalized 1/√3
        val = G00[tri_idx, idx_list].sum() / np.sqrt(3)
        G_red[0, j] = val
        G_red[j, 0] = np.conj(val)

    # dimer-dimer
    for i, idx_i in enumerate(neighbor_sets, start=1):
        for j, idx_j in enumerate(neighbor_sets, start=1):
            # sum over all 3×3 combinations, normalized 1/3
            block = G00[np.ix_(idx_i, idx_j)]
            G_red[i, j] = block.sum() / 3

    # diagonalize
    try:
        lambdas = np.linalg.eigvalsh(G_red)
        lambdas = np.sort(lambdas.real)
        # bound‐state condition for trimer‐energy scale 3U
        return lambdas[0] - (1.0 / (3 * U))
    except np.linalg.LinAlgError:
        return np.inf


def eigenvalue_root_equation_16(E, U, solver, eigen_idx, N_SUB):
    """
    Solve λₙ(E) = 1/(3U) for the nth bound band using the full 16×16
    trimer+all‑dimer subspace.
    """
    # 1. Trivial safeguards
    if U == 0 or np.isinf(E) or np.isnan(E):
        return np.inf

    # 2. Compute the full G₀₀ matrix (64×64)
    G00 = solver.calculate_G00(E)
    if np.any(np.isinf(G00)):
        return np.inf

    # 3. Build the 16 basis vectors: 4 trimers + 12 dimer+single states
    #    (each dimer+single is the symmetric combination of 3 raw states).
    trimer_idxs, dimer_sets = get_trimer_and_dimer_indices(N_SUB)

    # flatten the neighbor‑sets into one big list of length 12
    neighbor_sets = [idx_list
                     for sets_for_a in dimer_sets
                     for idx_list   in sets_for_a]

    # full 16‑state basis: first the 4 trimer raw indices, then 12 lists
    basis = trimer_idxs + neighbor_sets

    # 4. Precompute normalizations: 1 for trimers, √3 for each dimer block
    norms = [1]*len(trimer_idxs) + [np.sqrt(3)]*len(neighbor_sets)

    # 5. Build G_red (16×16) via <v_i|G00|v_j> = ∑ₐ₍ᵢ₎ ∑_b₍ⱼ₎ G00[a,b]/(norm_i·norm_j)
    M = len(basis)
    G_red = np.zeros((M, M), dtype=complex)
    for i, raw_i in enumerate(basis):
        for j, raw_j in enumerate(basis):
            # raw_i, raw_j are either ints or lists of 3 ints
            idx_i = raw_i if isinstance(raw_i, list) else [raw_i]
            idx_j = raw_j if isinstance(raw_j, list) else [raw_j]
            block_sum = np.sum(G00[np.ix_(idx_i, idx_j)])
            G_red[i, j] = block_sum / (norms[i] * norms[j])

    # 6. Diagonalize and pick out the nth eigenvalue
    try:
        lambdas = np.linalg.eigvalsh(G_red)
        lambdas = np.sort(lambdas.real)
        return lambdas[eigen_idx] - (1.0 / (3*U))
    except np.linalg.LinAlgError:
        return np.inf



def run_calculation_for_task(job_id, task_id, N_K):
    U = 3
    phi = 2*np.pi/3
    q_grid_size = 201

    # total-momentum grid
    #K_values = np.linspace(-np.pi, np.pi, N_K)
    #partial grid
    K_values = np.linspace(-np.pi/8, np.pi/8, N_K)

    K = K_values[task_id]
    print(f"Task {task_id}: K = {K:.4f}")

    # build T & solver
    H_intra, H_inter, H_inter_minus = get_real_space_matrices(phi)
    T = get_T_eff_matrices(K, H_intra, H_inter, H_inter_minus)
    c_min, c_max = get_continuum_bounds(K, phi, num_q=51)
    solver = GreenFunctionSolver(T, K, q_grid_size=q_grid_size)

    energies = np.full(N_SUB, np.nan)

    for band_idx in range(N_SUB):
        f = partial(
            eigenvalue_root_equation_16,
            U=U,
            solver=solver,
            eigen_idx=band_idx,
            N_SUB=N_SUB
        )

        # bracket just as before
        if U < 0:
            bracket = (c_min - 10*abs(U), c_min - 1e-6)
        else:
            bracket = (c_max + 1e-6, c_max + 10*abs(U))

        try:
            v0, v1 = f(bracket[0]), f(bracket[1])
            if np.isfinite(v0) and np.isfinite(v1) and np.sign(v0) != np.sign(v1):
                sol = root_scalar(f, bracket=bracket, method='brentq')
                if sol.converged:
                    energies[band_idx] = sol.root
        except Exception:
            pass

    # save continuum + trimer bands
    result = np.concatenate(([c_min, c_max], energies))
    outdir = "results"
    os.makedirs(outdir, exist_ok=True)
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
