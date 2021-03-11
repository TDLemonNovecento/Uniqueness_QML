'''
all representations that work well with the kernel learning files
!!!     ONLY HASHED AND SORTED REPRESENTATIONS WORK  !!!

Variables:
    Z: np.array of nuclear charges
    R: np.array of nuclear positions in cartesian coordinates [x,y,z]
    N: total number of electrons in system, irrelevant for most representations
'''
import numpy as np
import jax_representation as jrep
import qml


def Overlap_Matrix(Z, R, N = 0, size = 51):
    '''
    Returns OM hashed to size
    '''
    return(jrep.OM_full_sorted(Z, R, N, size)[0])

def Eigenvalue_Overlap_Matrix(Z, R, N=0, size = 51):
    '''
    returns hashed EVOM representation with hash = size
    '''
    return(jrep.OM_ev(Z, R, N, size)[0])

def Coulomb_Matrix(Z, R, N = 0, size = 23):
    '''
    returns hashed CM to size
    '''
    #return(qml.representations.generate_coulomb_matrix(Z, R, size = dim, sorting = 'row-norm'))
    return(jrep.CM_full_sorted(Z,R,N, size = size)[0])

def Eigenvalue_Coulomb_Matrix(Z, R, N = 0, size = 23):
    '''requires: : preordering of Z and R according to CM_full_sorted(Z, R, N)[1]
    (otherwise two identical molecules with atoms 1 and 2 switched in the Z and R array
    will be mapped onto different representations)
    '''
    #return(qml.representations.generate_eigenvalue_coulomb_matrix(Z, R, size = dim))
    return(np.asarray(jrep.CM_ev(Z, R, N, maxsize = size)[0]))

'''
def SLATM(Z, R, N = 0):

    #SLATM works across a dataset, but not for single results, therefore I am excluding it as
    #my solution of calculating the mbtypes does work but is not how this representations should
    #be used. A derivative of this function can either be taken w.r.t. the mbtypes, too, which
    #requires changes to my derivative functions which only consider one compound at a time,
    #or would require ignoring the dependence of the mbtypes on Z. derivatives w.r.t R and N
    #however can be taken easily by generating the mbtypes across the dataset in question
    #and modifying the function to include mbtypes as a fourth argument.

    #generates Spectrum of London and Axillrod-Teller-Muto potentials
    mbtypes = qml.representations.get_slatm_mbtypes([Z])
    return(qml.representations.generate_slatm(R, Z, mbtypes))

'''

def Bag_of_Bonds(Z, R, N = 0):
    '''
    qml gnerate_bob function requires superfluous variable 'atomtypes'
    as third argument
    '''
    return(qml.representations.generate_bob(Z, R, 'N', asize = {'C':7, 'H': 16, 'N':6, 'O': 4, 'F': 4}))


