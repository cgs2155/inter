import numpy as np
from _vectools import e_n

def F_n(X: list[int]):
    """
    Implementation of a tridiagonal-like operator from a sequence X,
    producing H_n for n in [1, len(X]]. Ultimately a 1D chain with hopping coefficient sqrt(X)
    """
    F_n_list = [np.array([[0]])]  # maybe H_0 is a 1x1 zero matrix?

    for n in range(1, len(X)+1):
        d = n+1  # dimension of the matrix
        F_num = np.zeros((d, d))

        for m in range(d-1):
            prefactor= np.sqrt(X[m])
            outer_product = np.outer(e_n(m, d), e_n(m+1, d)) + np.outer(e_n(m+1, d), e_n(m, d))
            F_num += prefactor * outer_product

        F_n_list.append(F_num)

    return F_n_list

def tbw(bands):
    """total bandwidth of a walk """
    total = 0
    for band in bands:
        total += np.max(band) - np.min(band)
    return total
