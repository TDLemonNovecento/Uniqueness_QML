'''In this package representation functions are stored and their derivatives returned'''
import jax.numpy as jnp
import numpy as np
from jax import grad, ops
import qml
import basis
from scipy import misc, special, linalg
import jax_math as jmath

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
    print('K', K)
    S = np.zeros((K,K))
    for a, bA in enumerate(thisbasis):
        print('a:', a, 'bA', bA)
    
    for a, bA in enumerate(thisbasis):      #access row a of S matrix; unpack list from tuple
        for b, bB in enumerate(thisbasis):  #same for B
            if (a == b):
                S[a,b] = 1
            else:
                rA = np.array(bA['r']) #get atom centered coordinates of atom A
                rB = np.array(bB['r'])
                lA,mA,nA = bA['l'],bA['m'],bA['n'] #get angular momentumnumbers of A
                lB,mB,nB = bB['l'],bB['m'],bB['n']
                disAB = np.linalg.norm(rA - rB)
                aA, aB = np.array(bA['a']), np.array(bB['a']) #alpha vectors
                rP = np.add(np.dot(aA, rA),np.dot(aB, rB))/ np.add(aA, aB) #calculate weighted center
                rPA = np.subtract(rP,rA) # distance between weighted center and A
                rPB = np.subtract(rP,rB)
                

                for alphaA, dA in zip(bA['a'], bA['d']): #alpha is exp. coeff. and dA contraction coeff.
                    for alphaB, dB in zip(bB['a'], bB['d']):
                        
                        #Implement overlap element
                        gamma = alphaA + alphaB
                        normA = jmath.OM_compute_norm(alphaA, lA, mA, nA) #compute norm for A
                        normB = jmath.OM_compute_norm(alphaB, lB, mB, nB)
                        S[a,b] += dA * dB * normA * normB *\
                            np.exp(-(alphaA*alphaB)/(alphaA + alphaB) * disAB**2) *\
                            jmath.OM_compute_Si(lA, lB, rPA[0], rPB[0], gamma) *\
                            jmath.OM_compute_Si(mA, mB, rPA[1], rPB[1], gamma) *\
                            jmath.OM_compute_Si(nA, nB, rPA[2], rPB[2], gamma)

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

    M_unsorted = OM_full_unsorted_matrix(Z, R, N)[0]
    val_row = np.asarray([jnp.linalg.norm(row) for row in M_unsorted])
    order = val_row.argsort()[::-1]

    print('order', order, 'unsorted_M', M_unsorted)
    M_sorted = jnp.asarray([[M_unsorted[i,j] for j in order] for i in order])
    print('sorted:', M_sorted)
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


def OM_index(Z, R, N, i, j):
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
        Entry of Overlap Matrix at position [i,j]

    '''

    print('do nothing')
    return


def derivative(fun, dx = [0,0]):
    if dx[0] == 0:
        d_fM = grad(fun, dx[1])(Z[0], Z[1], Z[2])
    elif dx[0] == 1:
        d_fM = grad(fun, dx[1])(R[1], R[2], R[3])
    else:
        d_fM = grad(fun)(N)
    return(d_fM)


def trial(i,j):
    if (i==j):
        k = i**2.4/2
    else:
        k = i*j
    return(k)




