'''In this package representation functions are stored and their derivatives returned'''
import jax.numpy as jnp
from jax import grad, ops
#import qml
import jax_basis
from jax_basis import empty_BoB_dictionary, BoB_emptyZ
from scipy import misc, special, linalg
import jax_math as jmath




def CM_full_unsorted_matrix(Z, R, size = 23):
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
    D = jnp.zeros((size, size))
    
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
    unsorted_M = CM_full_unsorted_matrix(Z,R, size)
    if unsorted:
        return(unsorted_M, 0)
    val_row = jnp.asarray([jnp.linalg.norm(row) for row in unsorted_M])
    order = val_row.argsort()[::-1]
    D = jnp.asarray([[unsorted_M[i,j] for j in order] for i in order])
    return(D, order)
    
def CM_ev(Z, R, N, maxsize = 23, unsorted = True):
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
    dim = len(Z)
    #print("len = ", dim)
    if unsorted:
        M = CM_full_unsorted_matrix(Z,R,N)
        order = jnp.asarray(range(dim))
    else:
        M, order = CM_full_sorted(Z,R, N, dim)
    ev, vectors = jnp.linalg.eigh(M)
    
    #pad ev by max size (23 for QM9, QM7)
    ev = jnp.pad(ev, (0,maxsize-dim))

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

def CM_eigenvectors_EVsorted(Z, R, N, cutoff = 10):
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
    S = jnp.zeros((K,K))
    
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

                    S = S.at[a,b].add(dA * dB * normA * normB *exponent* S_xyz)

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
    key_list = BoB_emptyZ(k_dictionary)
    
    unique, counts = jnp.unique(Z, return_counts = True)
    this_k_dictionary = dict(zip(unique, counts))

    for i in key_list:
        try:
            k_dictionary[i] = this_k_dictionary[i]
        except KeyError:
            k_dictionary.pop(i)

    clean_k_dictionary = {x:y for x, y in k_dictionary.items() if y !=0}
    clean_Z = list(clean_k_dictionary.keys())
    atomic_zipper = list(zip(Z, R))
    


    #first bag corresponds to CM_diagonal
    unsorted_first_bag =  [Zi**(2.4)/2 for Zi in Z]
    sorted_first_bag = np.sort(unsorted_first_bag)[::-1]
    bag0 = jmath.BoB_fill(sorted_first_bag, sum(clean_k_dictionary.values()))
    
    l = len(clean_Z)
    
    #loop over all i, j in clean_Z to get all possible combinations of Zi and Zj for the bags
    all_nonself_bags = []
    #Zi,Zj values needed to sort bags according to sum(Zi, Zj) and to lower(Zi, Zj)
    Zi_Zj_of_bags = []

    for i in range(l):
        Zi = clean_Z[i]
        i_zipper = [atom for atom in atomic_zipper if atom[0] == Zi]
        for j in range(i, l):
            bag_ij = []
            #same element bags (HH, OO, NN, ...)
            if (i == j):
                Zj = Zi
                Zi_Zj_of_bags.append([Zi, Zj])
                for index_i in range(len(i_zipper)):
                    for index_j in range(index_i + 1, len(i_zipper)):
                        distance = jnp.linalg.norm(i_zipper[index_i][1] - i_zipper[index_j][1])
                        entry = float(i_zipper[index_i][0]*i_zipper[index_j][0]/distance)
                        bag_ij.append(entry)
                

            #different element bags (HO, HN, ON, ...)
            else:
                Zj = clean_Z[j]
                Zi_Zj_of_bags.append([Zi, Zj])
                j_zipper = [atom for atom in atomic_zipper if atom[0] == Zj]
                #create ij bag
                for atom_i in i_zipper:
                    for atom_j in j_zipper:
                        distance = jnp.linalg.norm(atom_i[1] - atom_j[1])
                        entry = float(atom_i[0]*atom_j[0]/distance)
                        bag_ij.append(entry)
            #reverse sorting of bag_ij to get largest values at the beginning
            sorted_bag_ij = np.sort(bag_ij)[::-1]
            #pad bag with 0 to fit standard length of respective bag
            padded_bag_ij = jmath.BoB_fill(sorted_bag_ij, clean_k_dictionary[Zi]*clean_k_dictionary[Zj])
            all_nonself_bags.append( padded_bag_ij)
    
    sorted_nonself_vector = BoB_shuffle_bags(all_nonself_bags, Zi_Zj_of_bags)
    BoB = jnp.append(bag0, sorted_nonself_vector)
    
    return(BoB)

def BoB_shuffle_bags(unsorted_bags, Zi_Zj_array):
    '''sort BoB bags according to Zi, Zj values and return vector (without self interactions of atoms)'''
    index_list = []
    lower_list = []

    #calculate both Zi+Zj and get lower of the two values into list
    for Zij in Zi_Zj_array:
        index = Zij[0]+Zij[1]
        lower = min(Zij[0], Zij[1])
        index_list.append(index)
        lower_list.append(lower)

    #presort by lower Z value in case Zi+Zj values are equal (NN bag comes after CO bag)
    order_lower = jnp.argsort(lower_list)
    presorted_index = jnp.array(index_list)[order_lower]
    presorted_bags = jnp.array(unsorted_bags)[order_lower]

    #order by index Zi+Zj now
    order_index = jnp.argsort(presorted_index)
    sorted_bags = presorted_bags[order_index]
    
    #concatenate list of arrays into single np array
    bags = jnp.concatenate(sorted_bags)

    return(bags)




