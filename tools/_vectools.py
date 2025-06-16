import numpy as np
from tools._counttools import canonical
from scipy.linalg import expm

H_gate = 1/np.sqrt(2) * np.array([[ 1, 1],
                                  [ 1,-1]])
def pad(mat: np.ndarray, pad: int) -> np.ndarray:
    """
    Return a new array consisting of `mat` with a border of zeros of width `pad`.

    Parameters
    ----------
    mat : np.ndarray
        2D input array to pad.
    pad : int
        Width of the zero‐padding to add on each side.

    Returns
    -------
    padded : np.ndarray
        Array of shape (mat.shape[0] + 2*pad, mat.shape[1] + 2*pad)
        with `mat` in the center and zeros around.
    """
    if mat.ndim != 2:
        raise ValueError("Only 2D arrays are supported, got shape %s" % (mat.shape,))
    # create new array of zeros
    padded = np.zeros((mat.shape[0] + 2*pad, mat.shape[1] + 2*pad), dtype=mat.dtype)
    # place the original matrix in the center
    padded[pad:pad + mat.shape[0], pad:pad + mat.shape[1]] = mat
    return padded

def delta(i,j):
    """Kronecker delta implementation"""
    if i == j:
        return 1
    return 0

def e_n(n, d):
    # Creates a d-dimensional vector with 1 at the n-th position and 0 elsewhere
    vec = np.zeros(d)
    vec[n%d] = 1
    return vec

def J_n(n):
    return np.fliplr(np.eye(n))    

def direct_sum(A, B):
  """
  Computes the direct sum of two matrices. (Thank you Google)

  Args:
    A: The first matrix (NumPy array).
    B: The second matrix (NumPy array).

  Returns:
    The direct sum of A and B as a NumPy array.
  """
  m, n = A.shape
  p, q = B.shape
  result = np.zeros((m + p, n + q)).astype(A.dtype)
  result[:m, :n] = A
  result[m:, n:] = B
  return result

""" def C_n(X: list[int]):
    C_n_list = [np.array([[0]])]  # maybe H_0 is a 1x1 zero matrix?

    for n in range(1, len(X)+1):
        d = 2*n+1  # dimension of the matrix
        F_num = np.zeros((d, d))

        for m in range(d-1):
            prefactor= np.sqrt(X[m])
            outer_product = np.outer(e_n(m, d), e_n(m+1, d)) + np.outer(e_n(m+1, d), e_n(m, d))
            F_num += prefactor * outer_product

        F_n_list.append(F_num)

    return F_n_list"""

def temp_C_n(n):
    d = 2*n+2  # dimension of the matrix
    C_n = np.zeros((d, d))
    for i in range(0,d-1):
        C_n[i+1,i] = C_n[i,i+1] = np.sqrt(2)
    C_n[n+1,n] = C_n[n,n+1] = 2
    return C_n

def evolve(H, psi0, t):
    U = expm(-1j * H * t)  # time evolution operator U(t)
    psi_t = U @ psi0
    return psi_t

def limit(eigvecs, a, b):
    """The limiting distribution of the transition between states a and b for a system with given eigenvectors"""
    total = 0.0
    n = a.shape[0]  # dimension of the vectors; same as number of rows in eigvecs
    # Loop over each eigenvector, which is assumed to be stored as a column
    for i in range(n):
        dot_a = 0.0 + 0.0j
        dot_b = 0.0 + 0.0j
        # Calculate the dot products for a and b with the i-th eigenvector
        for j in range(n):
            dot_a += np.conjugate(a[j]) * eigvecs[j, i]
            dot_b += np.conjugate(b[j]) * eigvecs[j, i]
        prod = dot_a * dot_b
        total += prod.real * prod.real + prod.imag * prod.imag  # equivalent to np.abs(prod)**2
    return total


############# FUNCTIONS FOR GENERATING BLOCK MATRICES OF RGC ###################

def neck_permute(neck):
    """Generates the permutation matrices that shuffles the even
    and odd vertices based on the necklace"""
    #forgot how list slicing works so "list[start:stop:step]"
    # get odd indices
    odd = np.round((np.array(neck[0::2])-1)/2).astype(int)
    even = np.round(np.array(neck[1::2])/2 - 1).astype(int)
    N = len(odd)
    P_odd = np.zeros((N,N))
    P_even = np.zeros((N,N))
    for idx in range(0,N):        
        P_odd[idx,odd[idx]]=1
        P_even[idx,even[idx]]=1
    return P_odd, P_even

def B_l(l): 
    """
    Generates the upper block off-diagonal adjacency matrix for the bipartite subgraph 
    representing l hanging leaves on a single side
    """
    # Initialize B as an l x l zero matrix
    B = np.zeros((l, l), dtype=int)
    
    for i in range(l):
        B[i, i] = 1               # connect to own index
        B[i, (i - 1) % l] = 1      # connect to next index (wrap around)
    return B

def permute_bipartite_adjacency(B, P_O, P_E):
    B_new = P_O @ B @ P_E.T
    A_new = np.block([
        [np.zeros_like(B), B_new],
        [B_new.T, np.zeros_like(B)]
    ])
    return A_new

def O(neck): 
    """Generate orthongonal permuation matrix on subgraph"""
    a,b = neck_permute(neck)
    direct_sum(a,b)

def neck_from_O(O):
    """return the necklaces specified by the orthogonal permuation matrix"""
    necklace = []
    N = int(np.shape(O)[0]/2)
    ###get odds
    odds = []
    for row in O[:N,:N]:
        odds.append(np.argmax(row))

    evens = []
    for row in O[N:,N:]:
        evens.append(np.argmax(row))


    for i in range(0,len(odds)):
        necklace.append(int((odds[i] * 2)+1)+1)
        necklace.append(int(evens[i]*2)+1)

    return tuple(canonical(necklace))



######## OVERHAULED FROM ANDREW #############
def child_ratios(j: int,p: int)-> float:
    """
        j : integer index ranging from one to p 
        p : a branching factor

        see definition of child ratios in the paper.
    """
    assert 0 < j <= p

    return 1 - 2*(j-1)/(p-1)

def first_ham(p: int):
    """ 
        p    : branching factor.

        returns the (p + 2 by p +2 ) unfluxed adjacency matrix of the pnary 
        glued tree at depth one. The placquette flux is the variable flux this 
        should realistically only be used as a helper function to cascade below.
    """
    prefactor = 2 *child_ratios(1,p) - 2*child_ratios(2,p) 
    fluxrow = [0 if i == 0 or i == p+1 else -child_ratios(i,p)/prefactor + 1j for i in range(p+2)]
    fluxrow = np.array(fluxrow)

    first_elem = np.zeros(p + 2)
    first_elem[0] = 1
    out = np.outer(first_elem, fluxrow) - np.outer(np.conjugate(fluxrow),first_elem) 
    return out + np.flip(out)

def iterate_matrix(input_matrix, branching_factor: int):
    """
        input_matrix     : square hermitian matrix 
        branching_factor : number of copies of input_matrix along the diagonal of the output

        See paper. This function calculates T_d by using T_{d-1} and some trickery. 
        You have to build matrices like this or you will get unexpected behavior
        related to the fact that complex exponentials are periodic. 
        You could avoid this trickery if you solved the flux tiling problem in general. 
    """
    # take a tensor product of input matrix, and then pad the resulting matrix with zeros on all sides
    start = np.pad(np.kron(np.identity(branching_factor), input_matrix),1)

    # extract phases from previous iteration
    ### trickery start
    off_diag = np.diagonal(start,1) 
    phases = np.real(off_diag.copy())
    for x in range(2,len(phases)):
        if phases[x-1] == 0:
            phases[x] = 0
    phase_apply = (1 - 4 * np.sum(phases))/(2 - 2*child_ratios(2,branching_factor)) 
    ### trickery end

    # see App A of paper
    phase_vector = np.array([-child_ratios(i + 1, branching_factor)*phase_apply + 1j for i in range(0,branching_factor)])


    old_first = np.zeros(len(input_matrix))
    old_first[0] = 1
    padded_row = np.pad(np.kron(phase_vector,old_first),1)

    # new_first is first vector of T_d
    new_first = np.zeros(len(start))
    new_first[0] = 1

    start = start + 2*(np.outer(new_first, padded_row) - np.outer(np.conjugate(padded_row),new_first))
    start = (start + np.flip(start))/2
    return start


def cascade(numbers):
    """ 
        numbers : list of branching factors

        Calculates the antisymmetric flux function and 
        unfluxed adjacency matrix of a glued tree with 
        numbers branching factors
    """
    branching_factor = numbers[0]
    assert branching_factor > 1

    result = first_ham(branching_factor)
    for x in range(1,len(numbers)): 
        result = iterate_matrix(result,numbers[x])
    return result 

