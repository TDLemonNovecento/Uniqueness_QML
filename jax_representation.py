'''In this package representation functions are stored and their derivatives returned'''
import jax.numpy as jnp
import numpy as np
from jax.config import config
config.update("jax_enable_x64", True) #increase precision from float32 to float64
from time import perf_counter as tic
from jax import grad, ops
import jax_basis as basis
from jax_basis import empty_BoB_dictionary, BoB_emptyZ
from scipy import misc, special, linalg
import jax_math as jmath
import collections



def CM_full_unsorted_matrix(Z, R, N=0, size = 23):
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
    nick_version = False
    
    n = Z.shape[0]
    D = jnp.zeros((size, size))
    
    if nick_version:
        
        # calculate distances between atoms
        dr = R[:, None] - R
        
        distances = jnp.linalg.norm(dr, axis = 2)
        
        # compute Zi*Zj matrix
        charge_matrix = jnp.outer(Z, Z)

        # returns i,i indexes (of diagonal elements)
        diagonal_idx = jnp.diag_indices(n, ndim=2)        
        charge_matrix = ops.index_update(charge_matrix, diagonal_idx,  0.5 * Z ** 2.4)

        # fix diagonal elements to 1 in distance matrix
        distances = ops.index_update(distances, diagonal_idx, 1.0)

        #compute cm by dividing charge matrix by distance matrix
        cm_matrix = jnp.asarray(charge_matrix / distances)
        
        return(cm_matrix)

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

def CM_full_sorted(Z, R, N = 0, size=3, unsorted = False):
    ''' Calculates sorted coulomb matrix
    Parameters
    ----------
    Z : 1 x n dimensional array
    contains nuclear charges
    R : 3 x n dimensional array
    contains nuclear positions
    N : int, total charge, irrelevant for CM but needed for derivatives
    
    Return
    ------
    D : 2D array (matrix)
    Full Coulomb Matrix, dim(Z)xdim(Z)
    '''
    unsorted_M = CM_full_unsorted_matrix(Z,R,N, size)
    if unsorted:
        return(unsorted_M, 0)
    val_row = jnp.asarray([jnp.linalg.norm(row) for row in unsorted_M])
    order = val_row.argsort()[::-1]
    D = jnp.asarray([[unsorted_M[i,j] for j in order] for i in order])
    return(D, order)
    
def CM_ev(Z, R, N=0, maxsize = 23, unsorted = False):
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
    dim = Z.shape[0]
    if unsorted:
        M = CM_full_unsorted_matrix(Z,R,N, 23)
        order = jnp.asarray(range(dim))
    else:
        M, order = CM_full_sorted(Z,R, N, dim)
    ev, vectors = jnp.linalg.eigh(M)
    
    #pad ev by max size (23 for QM9, QM7)
    ev = jnp.pad(ev, (0,maxsize-dim))

    return(ev, order)

def CM_ev_unsrt(Z, R, N=0, size = 23):
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
    
    M = CM_full_unsorted_matrix(Z,R,N, size = size)
    
    ev, vectors = jnp.linalg.eigh(M)

    return(ev)


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

def CM_eigenvectors_EVsorted(Z, R, N= 0, cutoff = 10):
    ''' Matrix containing eigenvalues of unsorted Coulomb matrix,
    sorted by their eigenvalues. Cutoff possible at dedicated len.
    or for certain sizes of eigenvalues


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
    M : Matrix 
        contains eigenvectors of sorted CM
    (vectors: tuple
        contains Eigenvectors of matrix (n dim.)
        If i out of bounds, return none and print error)
    '''
    N = CM_full_unsorted_matrix(Z, R)
    ev, evec = jnp.linalg.eigh(N)
    order = jnp.argsort(ev)[:min(ev.size, cutoff)]
        
    sorted_evec = evec[order]
    
    return(sorted_evec)



def OM_full_unsorted_matrix(Z, R, N= 0):
    #print("started fast OM calculation")
    '''
    The overlap matrix is constructed as described in the
    'student-friendly guide to molecular integrals' by Murphy et al, 2018
    STO-3G basis set (3 gaussian curves used to approximate the STO solution)
    and was optimized by myself and Nick Browning

    Parameters
    ----------
    Z : 1 x n dimensional array
        contains nuclear charges
    R : 3 x n dimensional array
        contains nuclear positions
    size : size for hashed matrix
 
    Return
    ------
    D : 2D array (matrix)
        Full Coulomb Matrix, dim(Z)xdim(Z)
    '''
    tstart = tic()

    thisbasis, K = basis.build_sto3Gbasis(Z, R)
    S = np.zeros((K,K))
    

    #tbasis = tic()
    #taold = tbasis
    #tbold = tbasis

    for a, bA in enumerate(thisbasis):      #access row a of S matrix; unpack list from tuple
        #ta = tic() 
        for b, bB in enumerate(thisbasis):  #same for B
            #tb = tic()
            rA, rB = bA['r'], bB['r'] #get atom centered coordinates of A and B
            lA,mA,nA = bA['l'],bA['m'],bA['n'] #get angular momentumnumbers of A
            lB,mB,nB = bB['l'],bB['m'],bB['n']
 
            aA, aB = np.asarray(bA['a']), np.asarray(bB['a']) #alpha vectors
            
            for alphaB, dB in zip(bB['a'], bB['d']):
              
                #Implement overlap element
                normA = jmath.OM_compute_norm(aA, lA, mA, nA) #compute norm for A
                normB = jmath.OM_compute_norm(alphaB, lB, mB, nB)
 
                S_xyz = jmath.OM_compute_Sxyz(rA, rB, aA, alphaB, lA, lB, mA, mB, nA, nB)
                exponent = np.exp(-aA*alphaB *jmath.IJsq(rA, rB)/(aA + alphaB))
                
                #t1 = tic()
                #factor is array over alphaA, dA elements
                factor = np.array(bA['d']) * dB * normA * normB*exponent* S_xyz
                
                S[a][b] += sum(factor) 
                #t2 = tic()

                #print("adding factor to blist:", t2-t1)

            #print("b,bB loop:", tbold - tb)
            #tbold = tb
        #print("a,bA loop:", taold - ta)
        #taold = ta

    tend = tic()

    #print("total time:", tend - tstart)

    return(S)




def OM_full_sorted(Z, R, N = 0, size = 51):
    ''' Calculates sorted coulomb matrix
    Parameters
    ----------
    Z : 1 x n dimensional array
    contains nuclear charges
    R : 3 x n dimensional array
    contains nuclear positions
    size: int, size of hashed matrix
    
    Return
    ------
    D : 2D array (matrix)
    Full Coulomb Matrix, dim(Z)xdim(Z)
    '''

    M_unsorted = OM_full_unsorted_matrix(Z, R, N)
    dim = M_unsorted.shape[0]


    val_row = np.asarray([jnp.linalg.norm(row) for row in M_unsorted])
    order = val_row.argsort()[::-1]

    M_sorted = jnp.asarray([[M_unsorted[i,j] for j in order] for i in order])

    #hash the matrix
    M = np.pad(M_sorted, [(0, size-dim), (0, size-dim)], mode='constant')


    return(M, order)


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

def OM_ev(Z, R, N=0, maxsize = 51):
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
    maxsize : int, size of hashed fingerprint

    Return
    ------
    ev : vector (1 x n dim.)
        contains eigenvalues of sorted CM
    (vectors: tuple
        contains Eigenvectors of matrix (n dim.)
        If i out of bounds, return none and print error)
    '''

    M, order = OM_full_sorted(Z,R, size = maxsize)
    ev, vectors = jnp.linalg.eigh(M)
    return(ev, order)

def OM_ev_unsrt(Z, R, N=0):
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
    M = OM_full_unsorted_matrix(Z,R)
    ev, vectors = jnp.linalg.eigh(M)
    return(ev)

def BoB_unsorted(nuclear_charges, coordinates, N= 0, asize={1:16, 6:7, 7:6, 8:4, 9:4}):
    """ Generates a bag-of-bonds representation of the molecule. ``size=`` denotes the max number of atoms in the molecule (thus the size of the resulting square matrix.
    ``asize=`` is the maximum number of atoms of each type (necessary to generate bags of minimal sizes), with Z:no_of_atoms.
    The resulting matrix is the upper triangle put into the form of a 1D-vector.
    The returned type will be a list of 1D coulomb matrices.
    :param arg1: Input representation.
    :type arg1: (N, 3) shape numpy array.
    :param arg1: Nuclear charges.
    :type arg1: list of floats
    :return: List of 1D Coulomb matrix
    """
    Z = nuclear_charges
    R = coordinates

    natoms = len(Z)

    coulomb_matrix = CM_full_unsorted_matrix(Z, R, N=0, size = natoms)

    descriptor = []
    #atomtypes = collections.Counter(Z)
    
    for atom1, size1 in asize.items():
        pos1 = np.where(Z == atom1)[0]
        feature_vector = np.zeros(size1)
        feature_vector[:pos1.size] = np.diag(coulomb_matrix)[pos1]
        feature_vector.sort()
        descriptor.append(feature_vector[:])
        for atom2, size2 in asize.items():
            if atom1 > atom2:
                continue
            if atom1 == atom2:
                size = int(size1*(size1-1)/2)
                feature_vector = np.zeros(size)
                sub_matrix = coulomb_matrix[np.ix_(pos1,pos1)]
                feature_vector[:int(pos1.size*(pos1.size-1)/2)] = sub_matrix[np.triu_indices(pos1.size, 1)]
                feature_vector.sort()
                descriptor.append(feature_vector[:])
            else:
                pos2 = np.where(Z == atom2)[0]
                feature_vector = np.zeros(size1*size2)
                feature_vector[:pos1.size*pos2.size] = coulomb_matrix[np.ix_(pos1,pos2)].ravel()
                feature_vector.sort()
                descriptor.append(feature_vector[:])

    return np.concatenate(descriptor)

def BoB_dimension(Z):
    ''' Calculates dimension of BoB
    Parameters
    ----------
    Z : 1 x n dimensional array
    contains nuclear charges
    R : 3 x n dimensional array
    contains nuclear positions
    
    Returns
    -------
    n : integer
    '''
    #replace -1 indices in k_dictionary by actual values
    #get keys to be replaced
    
    unique, counts = np.unique(Z, return_counts = True)
    #calculate total number of instances in Zi,Zj bags
    ijbag = 0

    for i in range(len(unique)):
        for j in range(i+1, len(unique)):
            bag = counts[i]*counts[j]
            ijbag += bag

    #calculate total number of instances in Zi,Zi bags
    iibag = 0

    for i in counts:
        bag = jmath.binomial(i, 2)
        iibag += bag
    
    return(len(Z) + iibag + ijbag)


