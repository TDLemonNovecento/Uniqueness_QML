#contains representation mapping functions
import qml
import numpy as np
import sympy as sp


#r1, r2, r3, Z, N = sp.symbols('r1 r2 r3 Z N')
'''i, j are molecules with xyz, Z, and N info '''

def Coulomb_Matrix_FVec(r1, r2, r3, Z, N, n):
    '''Compute Vector containing coulomb matrix information

    Parameters
    ----------
    r1,r2,r3 : variable
        Armstroeng, coordinates in xyz space
    Z : vector
        variable of nuclear charges
    N : variable
        number of electrons
    n : number
        size of Z = number of atoms in system

    Returns
    -------
    fM : vector
        function
        '''
    def fM(r1, r2, r3, Z, N):
        i, j = 0
        n = len(Z)
        print(n)
        size = n*(n + 1) / 2
        fM = zeros(size)
        print(fM)

        fM = 2*Z**2/(r1**2 + r2**2 + r3**2)
    return(fM)

def Coulomb_Matrix_VtoM(v):
    CM = representations.vector_to_matrix(v) 
    return(CM)
    

def coulomb_element(i, j, atomic_pointer):
    #coulomb matrix
    size = 2 
    mol = qml.Compound(xyz)
    mol.generate_coulomb_matrix(size = size, sorting="row-norm")
    return(mol.representation)

def cm_ev(xyz):
    mol = qml.Compound(xyz)
    mol.generate_eigenvalue_coulomb_matrix(mol)
    return(mol.representation)

class Atomic_Pointer:
    def __init__(self, position, atomic_charge, no_electrons):
        self.position = position
        self.atomic_charge = atomic_charge
        self.no_electrons = no_electrons
