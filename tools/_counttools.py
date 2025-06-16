import numpy as np
import random
from scipy.special import comb
from math import factorial
from itertools import permutations
from tqdm import tqdm 

def num_rand_cycle(n):
    return factorial(n)*factorial(n-1)/(2)

def canonical(necklace):
    """
    Given a necklace (tuple of beads), this function returns a canonical
    representation by considering all rotations and reflections.
    """
    n = len(necklace)
    # Generate all rotations of the necklace
    rotations = [necklace[i:] + necklace[:i] for i in range(n)]
    
    # Generate rotations of the reversed (reflected) necklace
    reversed_necklace = necklace[::-1]
    rotations_reflected = [reversed_necklace[i:] + reversed_necklace[:i] for i in range(n)]
    
    # Return the lexicographically smallest rotation/reflection as the canonical form
    return min(rotations + rotations_reflected)

def enumerate_necklaces(n):
    """
    Enumerate all distinct necklaces with 2*n beads, where odd numbers (1,3,...,2n-1)
    and even numbers (2,4,...,2n) alternate and rotations/reflections are considered equivalent.
    """
    # Create lists for odd and even beads
    odd = list(range(1, 2*n + 1, 2))
    even = list(range(2, 2*n + 1, 2))
    
    necklaces = set()
    
    # Optional: Fix the first bead (here the first odd bead) to reduce rotational redundancy.
    # Then we only permute the remaining odd beads.
    for odd_perm in permutations(odd[1:]):
        odd_seq = (odd[0],) + odd_perm
        for even_perm in permutations(even):
            # Interlace odd and even beads
            necklace = []
            for o, e in zip(odd_seq, even_perm):
                necklace.append(o)
                necklace.append(e)
            # Convert to tuple and compute its canonical form
            canonical_necklace = canonical(tuple(necklace))
            necklaces.add(canonical_necklace)
    
    return necklaces

def enumerate_necklaces_tqdm(n):
    """
    Enumerate all distinct necklaces with 2*n beads, where odd numbers (1,3,...,2n-1)
    and even numbers (2,4,...,2n) alternate and rotations/reflections are considered equivalent.
    """
    # Create lists for odd and even beads
    odd = list(range(1, 2*n + 1, 2))
    even = list(range(2, 2*n + 1, 2))
    
    necklaces = set()
    
    # The total number of iterations is factorial(n-1) * factorial(n)
    # since odd[1:] has n-1 elements and even has n elements.
    total_iterations = factorial(n-1) * factorial(n)
    
    with tqdm(total=total_iterations) as pbar:
        # Fix the first odd bead to reduce some rotational redundancy
        # and only permute the remaining odd beads.
        for odd_perm in permutations(odd[1:]):
            odd_seq = (odd[0],) + odd_perm
            for even_perm in permutations(even):
                # Interlace odd and even beads
                necklace = []
                for o, e in zip(odd_seq, even_perm):
                    necklace.append(o)
                    necklace.append(e)

                # Convert to tuple and compute its canonical form
                canonical_necklace = canonical(tuple(necklace))
                necklaces.add(canonical_necklace)
                
                # Update the progress bar after each inner loop iteration
                pbar.update(1)

    return necklaces


def gen_necklace(n, rng = None):
    if rng is None:
        rng = random

    odd_seq = list(range(3, 2*n+1,2))
    even_seq = list(range(2, 2*n+1,2))
    rng.shuffle(odd_seq)
    odd_seq = [1]+odd_seq
    rng.shuffle(even_seq)

    necklace = []
    for o, e in zip(odd_seq, even_seq):
        necklace.append(o)
        necklace.append(e)
    return canonical(tuple(necklace))


def isSubset(bigger_set, smaller_set):
    for element in smaller_set:
        if element not in bigger_set:
            return False
    return True