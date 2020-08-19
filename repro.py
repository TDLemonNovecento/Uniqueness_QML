#contains representation mapping functions
import qml
import numpy as np
import sympy as sp
import derivative
import sys

N = sp.symbols('N')

Z1, Z2, Z3 = sp.symbols('Z1 Z2 Z3')

Z = sp.Matrix([[Z1, Z2, Z3]])

#x1, x2, x3, y1, y2, y3, z1, z2, z3 = sp.symbols('x1 x2 x3 y1 y2 y3 z1 z2 z3')
r1, r2, r3 = sp.symbols('r1 r2 r3')

R =  sp.Matrix([[r1, r2, r3]])


'''Sample program for max. 3 atoms '''

def Coulomb_Matrix_FVec(R, Z, N, n, which_derivative = 0, which_dx = [0,0], which_ddx = [0,0]):

    '''Compute Vector containing coulomb matrix information

    Parameters
    ----------
    R : vector
        position vectors r1, r2, ... , rn of n Atoms
        Armstroeng, coordinates in xyz space
    Z : vector
        nuclear charges Z1, Z2, ..., Zn of n Atoms
    N : variable
        number of electrons in system
        if neutral, N = sum_i(Zi)
    which_derivative :
        0 = original function is returned
        1 = first derivative defined by which_dx is returned
        2 = second derivative defined by which_dx and which_ddx is returned
    which_dx, which_ddx :
        [0, i] returns derivative by ri
        [1, i] returns derivative by Zi
        [2, i] returns derivative by N

    Returns
    -------
    fM : Matrix
        '''
    
    assert (which_derivative in [0, 1, 2]), "which_derivative value out of bounds. Choose 0, 1, 2 to get no derivative, the first or the second."
            
    def f(i, j):
        if(i == j):
            print(Z[i])    
            return(Z[i]**(2.4)/2)
        else:
            difference = sp.sqrt((R[i]-R[j])**2)
            return (Z[i]*Z[j]/difference)
    fM = sp.Matrix(n, n, f)
                      

    if (which_derivative == 0):
        function = fM       
    elif (which_derivative == 1):
        if which_dx[0] == 0:
            print("derive by %s" % R.row(0).col(which_dx[1]))
            print(R.row(0).col(which_dx[1]))
            function = first_derivative_CM(fM, R.row(0).col(which_dx[1]))
        elif which_dx[0] == 1:
            print("derive by %s" % Z.row(0).col(which_dx[1]))
            function = first_derivative_CM(fM, Z.row(0).col(which_dx[1]))
        elif which_dx[0] == 2:
            print("derive by N")
            function = first_derivative_CM(fM, N)
        else:
            print("your which_dx pointer is messed up")
    
    elif (which_derivative == 2):
        function = "second derivative"

    return(function)

def first_derivative_CM(fM, dx = Z1):

    '''Compute the first derivative of a mapping function fM.
    Tries analytical derivation first.
    If it fails, numerical derivation is performed.

    Parameters
    ----------
    fM : function
        maps from xyz to space of representation M
    n : number of atoms

    Internal Variables
    ------------------
    r1,r2,r3 : variable
        Armstroeng, coordinates in xyz space
    Z : vector
        variable of Nuclear charge
    N : variable
        number of electrons

    Returns
    -------
    vector of 5 derivatives
    matching format to mapping function fM
    
    '''
    n = fM.shape[0] #determine size of CM

    def f(i,j): #use sp.FunctionMatrix, its clearer
        return(derivative.firstd(fM.row(i).col(j), dx))

    d_fM = (n, n, f)
    print(d_fM)

    return("first_der")



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
