'''In this package representation functions are stored and their derivatives returned'''
import jax.numpy as jnp
import numpy as np
from jax import grad, ops
import qml
import basis
from basis import empty_BoB_dictionary
from scipy import misc, special, linalg
import jax_math as jmath
import list_math as lmath

def CM_full_unsorted_matrix(Z, R):
    ''' Calculates unsorted coulomb matrix
    Parameters
    ----------
    Z : 1 x n dimensional array
    contains nuclear charges
    R : 3 x n dimensional array
    contains nuclear positions
    
    Return
    ------
    D : 2D array (matrix)
    Full Coulomb Matrix, dim(Z)xdim(Z)
    '''
    n = Z.shape[0]
    D = jnp.zeros((n, n))
    
    #indexes need to be adapted to whatever form comes from xyz files
    for i in range(n):
        Zi = Z[i]
        D = ops.index_update(D, (i,i), Zi**(2.4)/2)
        for j in range(n):
            if j != i:
                Zj = Z[j]
                Ri = R[i, :]
                Rj = R[j, :]
                distance = jnp.linalg.norm(Ri-Rj)
                D = ops.index_update(D, (i,j) , Zi*Zj/(distance))
    return(D)

def CM_full_sorted(Z, R, N = 0):
    ''' Calculates sorted coulomb matrix
    Parameters
    ----------
    Z : 1 x n dimensional array
    contains nuclear charges
    R : 3 x n dimensional array
    contains nuclear positions
    
    Return
    ------
    D : 2D array (matrix)
    Full Coulomb Matrix, dim(Z)xdim(Z)
    '''
    
    unsorted_M = CM_full_unsorted_matrix(Z,R)
    val_row = np.asarray([jnp.linalg.norm(row) for row in unsorted_M])
    order = val_row.argsort()[::-1]
    D = jnp.asarray([[unsorted_M[i,j] for j in order] for i in order])
    return(D, order)
    
def CM_ev(Z, R, N):
    '''
    Parameters
    ----------
    Z : 1 x n dimensional array
        contains nuclear charges
    R : 3 x n dimensional array
        contains nuclear positions
    N : float
        number of electrons in system
        here: meaningless, can remain empty
    
    Return
    ------
    ev : vector (1 x n dim.)
        contains eigenvalues of sorted CM
    (vectors: tuple
        contains Eigenvectors of matrix (n dim.)
        If i out of bounds, return none and print error)
    '''

    M, order = CM_full_sorted(Z,R)
    ev, vectors = jnp.linalg.eigh(M)
    return(ev, order)

def CM_single_ev(Z, R, N =  0., i = 0):
    '''
    Parameters
    ----------
    Z : 1 x n dimensional array
        contains nuclear charges
    R : 3 x n dimensional array
        contains nuclear positions
    N : float
        number of electrons in system
        here: meaningless, can remain empty
    i : integer
        identifies EV to be returned

    Return
    ------
    ev : scalar
        Eigenvalue EV(i)
        If i out of bounds, return none and print error
    '''

    M = CM_trial(Z,R)
    ev, vectors = jnp.linalg.eigh(M)
    
    if i in range(len(ev)):
        return(ev[i])
    else:
        print("EV integer out of bounds, maximal i possible: %i" % len(ev))
        return()

def CM_index(Z, R, N, i = 0, j = 0):
    '''
    Parameters
    ----------
    Z : 1 x n dimensional array
        contains nuclear charges
    R : 3 x n dimensional array
        contains nuclear positions
    N : float
        number of electrons in system
        here: meaningless, can remain empty
    i : integer
        identifies row to be returned
    j : integer
        identifies column to be returned

    Return
    ------
    D[i,j] : scalar
        Entry of Coulomb Matrix at position [i,j]
    '''
    n = Z.shape[0]
    Zi = Z[i]
    if i == j:
        return(Zi**(2.4)/2)
    else:
        Zj = Z[j]
        Ri = R[i, :]
        Rj = R[j, :]
        distance = jnp.linalg.norm(Ri-Rj)
        return( Zi*Zj/(distance))


def OM_full_unsorted_matrix(Z, R, N):
    '''
    The overlap matrix is constructed as described in the
    'student-friendly guide to molecular integrals' by Murphy et al, 2018
    STO-3G basis set (3 gaussian curves used to approximate the STO solution)

    Parameters
    ----------
    Z : 1 x n dimensional array
        contains nuclear charges
    R : 3 x n dimensional array
        contains nuclear positions

    Return
    ------
    D : 2D array (matrix)
        Full Coulomb Matrix, dim(Z)xdim(Z)
    '''


    thisbasis, K = basis.build_sto3Gbasis(Z, R)
    S = np.zeros((K,K))
    
    for a, bA in enumerate(thisbasis):      #access row a of S matrix; unpack list from tuple
        for b, bB in enumerate(thisbasis):  #same for B
            
            rA, rB = bA['r'], bB['r'] #get atom centered coordinates of A and B
            lA,mA,nA = bA['l'],bA['m'],bA['n'] #get angular momentumnumbers of A
            lB,mB,nB = bB['l'],bB['m'],bB['n']
            
            aA, aB = bA['a'], bB['a'] #alpha vectors

            for alphaA, dA in zip(bA['a'], bA['d']): #alpha is exp. coeff. and dA contraction coeff.
                for alphaB, dB in zip(bB['a'], bB['d']):
                    #Implement overlap element
                      
                    normA = jmath.OM_compute_norm(alphaA, lA, mA, nA) #compute norm for A
                    normB = jmath.OM_compute_norm(alphaB, lB, mB, nB)
                    S_xyz = jmath.OM_compute_Sxyz(rA, rB, alphaA, alphaB, lA, lB, mA, mB, nA, nB)
                    exponent = np.exp(-alphaA*alphaB *jmath.IJsq(rA, rB)/(alphaA + alphaB))

                    S[a,b] += dA * dB * normA * normB *exponent* S_xyz

    return(S, K)

def OM_full_sorted(Z, R, N = 0):
    ''' Calculates sorted coulomb matrix
    Parameters
    ----------
    Z : 1 x n dimensional array
    contains nuclear charges
    R : 3 x n dimensional array
    contains nuclear positions
    
    Return
    ------
    D : 2D array (matrix)
    Full Coulomb Matrix, dim(Z)xdim(Z)
    '''

    M_unsorted, dim = OM_full_unsorted_matrix(Z, R, N)
    val_row = np.asarray([jnp.linalg.norm(row) for row in M_unsorted])
    order = val_row.argsort()[::-1]

    M_sorted = jnp.asarray([[M_unsorted[i,j] for j in order] for i in order])
    return(M_sorted, order)


def OM_dimension(Z):
    '''
    Returns dimensions of OM matrix without calculating the basis explicitely

    Variables
    ---------
    Z : 1 x n dimensional array
        contains nuclear charges

    Returns
    -------
    d : integer
        dimentsion of OM matrix
    '''
    d = 0
    for nuc in Z:
        d += len(basis.orbital_configuration[nuc])
    return d

def OM_ev(Z, R, N):
    '''
    Parameters
    ----------
    Z : 1 x n dimensional array
        contains nuclear charges
    R : 3 x n dimensional array
        contains nuclear positions
    N : float
        number of electrons in system
        here: meaningless, can remain empty

    Return
    ------
    ev : vector (1 x n dim.)
        contains eigenvalues of sorted CM
    (vectors: tuple
        contains Eigenvectors of matrix (n dim.)
        If i out of bounds, return none and print error)
    '''

    M, order = OM_full_sorted(Z,R)
    ev, vectors = jnp.linalg.eigh(M)
    return(ev, order)


def BoB_full_sorted(Z, R, N = 0, k_dictionary = empty_BoB_dictionary ):
    ''' Calculates sorted BoB
    Parameters
    ----------
    Z : 1 x n dimensional array
    contains nuclear charges
    R : 3 x n dimensional array
    contains nuclear positions
    N : value
    total charge of system
    k_dictionary : 20-dim dictionary
    contains nuclear charge and corresponding maximal number of incidences
    used for dimensionality tracking across database
    For values that are to be taken according to no. of occurances in passed molecules,
    index is -1

    Return
    ------
    D : 2D array (matrix)
    Full Coulomb Matrix, dim(Z)xdim(Z)
    '''
    #replace -1 indices in k_dictionary by actual values
    #get keys to be replaced
    key_list = lmath.BoB_emptyZ(k_dictionary)
    
    unique, counts = np.unique(Z, return_counts = True)
    this_k_dictionary = dict(zip(unique, counts))

    for i in key_list:
        try:
            k_dictionary[i] = this_k_dictionary[i]
        except KeyError:
            k_dictionary.pop(i)

    clean_k_dictionary = {x:y for x, y in k_dictionary.items() if y !=0}
    
    
    #first bag corresponds to CM_diagonal
    unsorted_first_bag =  [Zi**(2.4)/2 for Zi in Z]
    sorted_first_bag = np.sort(unsorted_first_bag)[::-1]
    bag0 = jmath.BoB_fill(sorted_first_bag, sum(clean_k_dictionary.values()))
    
    l = len(clean_k_dictionary)

    for i in range(1, l+1):
        for j in range(i, l+1)
        bag_i = 

            Zj = Z[j]
        Ri = R[i, :]
        Rj = R[j, :]
        distance = jnp.linalg.norm(Ri-Rj)
        return( Zi*Zj/(distance))


        sorted_bag_i = np.sort(bag_i)[::-1]
        padded_bag_i = jmath.BoB_fill(sorted_bag_i, k_dictionary[i]

    M_unsorted, dim = OM_full_unsorted_matrix(Z, R, N)
    val_row = np.asarray([jnp.linalg.norm(row) for row in M_unsorted])
    order = val_row.argsort()[::-1]

    M_sorted = jnp.asarray([[M_unsorted[i,j] for j in order] for i in order])
    return(M_sorted, order)


