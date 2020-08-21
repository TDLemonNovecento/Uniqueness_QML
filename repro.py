#contains representation mapping functions
import qml
import numpy as np
import sympy as sp
import derivative
import sys
from der_symbols import * 

N = sp.symbols('N')


'''Sample program for max. 3 atoms '''

def Coulomb_Matrix_FM(R, Z, N, n, which_derivative = 0, which_dx = [0,0, 0], which_ddx = [0,0,0]):

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
    print("first checkpoint")       
    def f_CM(i, j):
        if(i == j):
            return(Z[i]**(2.4)/2)
        else:
            difference = R.row(i) - R.row(j)
            distance = sp.sqrt(difference.dot(difference))
            return (Z[i]*Z[j]/distance)
    
    
    fM = sp.Matrix(n, n, f_CM)
    print("second checkpoint")                  

    if (which_derivative == 0):
        function = fM       
    elif (which_derivative == 1):
        function = derivative_organizer(derivative_CM, fM, which_dx)

        '''        if which_dx[0] == 0:
            print("derive by %s" % R[0,which_dx[1]])
            function = first_derivative_CM(fM, R[0,which_dx[1]])
        elif which_dx[0] == 1:
            print("derive by %s" % Z[0,which_dx[1]])
            function = first_derivative_CM(fM, Z[0,which_dx[1]])
        elif which_dx[0] == 2:
            print("derive by N")
            function = first_derivative_CM(fM, N)
        else:
            print("your which_dx pointer is messed up")
        '''    
    elif (which_derivative == 2):
        d_function = derivative_organizer(derivative_CM, fM, which_dx)
        function = derivative_organizer(derivative_CM, d_function, which_ddx)

    return(function)

def derivative_organizer(der_f, fM, which_dx):
    '''handles derivative arguments [i,j] of which_dx or which_ddx
    Assigns them to dri, dZi or dN and sends function to representation
    specific derivative function der_f

    Parameters
    ----------
    der_f : derivative function
            representation specific
    fM: variable
        matrix, vector, or else of representation mapping
    which_dx : array [i,j, k]
            i index chooses from [R, Z, N]
            j index chooses from [r1, r2, ...] or [Z1, Z2, ...]
            k index chooses from [xi, yi, zi] of the ri vector

    Internal Variables
    ------------------
    R = [r1, r2, ...]
    ri = [xi, yi, zi]
    Z = [Z1, Z2, ...]

    Returns
    -------
    function of same shape as fM
    (or depending on der_f different shape)
    '''

    if which_dx[0] == 0:
        print("derive by %s" % R[which_dx[1],which_dx[2]])
        function = der_f(fM, R[which_dx[1], which_dx[2]])
    elif which_dx[0] == 1:
        print("derive by %s" % Z[0,which_dx[1]])
        function = der_f(fM, Z[0,which_dx[1]])
    elif which_dx[0] == 2:
        print("derive by N")
        function = der_f(fM, N)
    else:
        print("your which_dx pointer is messed up")
        sys.exit(1)
    return(function)


def derivative_CM(fM, dx = Z1):
    #may be replaced by analytic_func(A, f, x) in sympy.matrices 
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
        return(derivative.firstd(fM[i,j], dx))
    d_fM = sp.Matrix(n, n, f)
       

    return(d_fM)


def subs_CM(f, cZ, cR, cN):
    '''Substitutes sympy symbols by real values
    Parameters
    ----------
    f : function
    cZ : vector
        contains nuclear charges [Z1, Z2, ...]
    cR : matrix
        contains position vectors [r1, r2, ...]
        with ri = [xi, yi, zi]
    cN : value
        total number of electrons in system
        in uncharged system == no. Zi
    Returns
    -------
    Matrix of shape of f
    '''
    n = f.shape[0]
    nZ = cZ.shape[0]
    
    try:
        for i in range(n):
            for j in range(n):
                for k in range(1, nZ+1):
                    f[i,j] = f[i,j].subs({sp.Symbol('x%i' %k): cR[k-1, 0], sp.Symbol('y%i' %k) : cR[k-1,1], sp.Symbol('z%i' %k) : cR[k-1,2], sp.Symbol('Z%i' %k) : cZ[k-1]})
                f[i,j] = f[i,j].subs({sp.Symbol('N') : cN})
    except IndexError:
        pass
    return(f)


def subs_CM2(f, cZ, cR, cN):
    '''Substitutes sympy symbols by real values
    Parameters
    ----------
    f : function
    cZ : vector
    	contains nuclear charges [Z1, Z2, ...]
    cR : matrix
    	contains position vectors [r1, r2, ...]
    	with ri = [xi, yi, zi]
    cN : value
    	total number of electrons in system
    	in uncharged system == no. Zi
    Returns
    -------
    Matrix of shape of f
    '''
    n = f.shape[0]
    
    try:
        for i in range(n):
            for j in range(n):
                f[i,j] = f[i,j].subs({sp.Symbol('x1'):cR[0, 0], sp.Symbol('y1'):cR[0,1], sp.Symbol('z1'):cR[0,2], sp.Symbol('x2'):cR[1,0], sp.Symbol('y2'): cR[1,1], sp.Symbol('z2'):cR[1,2], sp.Symbol('Z1'):cZ[0], sp.Symbol('Z2') : cZ[1], sp.Symbol('N') : cN})
    except IndexError:
        pass
    return(f)
