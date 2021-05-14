import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from math import factorial
import database_preparation as datprep

'''
standard size for BoB:
    {'C': 7, 'H': 16, 'N': 3, 'O': 3, 'S': 1}
    #number of bags has no effect on dimensionality?
size of OM for max and min number of orbitals.
# of orbitals:
    H, He: 1
    Li, Be, B, C, N, O, F, Ne: 5
    Na, Mg, Al, Si, P, S, Cl, Ar: 9
    K, Ca: 13
'''

def nchoose2(vec):
    V = []
    for i in vec:
        nfac = factorial(i)
        try:
            kfac = 2* factorial(i-2)
        except ValueError:
            nfac = 1
            kfac = 1
        V.append(nfac/kfac)
    return(np.asarray(V))

def BOB_dimension(N, l, k=[]):
    self_interactions = 0
    cross_interactions = 0
    print("k = ", k)
    for i in range(1, l+1):
        self = float(k[i-1])*(float(k[i-1])-1.)/2.
        self_interactions += self
        for j in range(i+1, l+1):
            cross_interactions += k[i-1] * k[j-1]
    
    dim = N + self_interactions + cross_interactions
    return(dim)



