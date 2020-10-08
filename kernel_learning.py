'''do automatic differentiation all the way'''
import jax.numpy as jnp 
import jax_representation as jrep
import numpy as np
import qml
import os
from xtb.interface import Calculator
from xtb.utils import get_method


datapath = "/home/stuke/Databases/XYZ_ethin/"

#representation to be used: CM_eigenvectors_EVsorted(Z, R, N, cutoff = 8)
def calculate_energies(information_list):
    '''
    Variables
    ---------
    information_list : list of lists, with
                       information_list[i] = ['name of file', [Z], [R], [N], [atomtypes]]
                       [Z].type and [R].type is np.array
    Returns
    -------
    List of len(information_list) with energies of the moelcules
    '''
    energies = []
    
    for mol in information_list:
        Z = mol[1]
        R = mol[2]
        #create xtb calculator
        calc = Calculator(get_method("GFN2-xTB"), Z, R)
        results = calc.singlepoint()
        energy = results.get_energy()
        energies.append(energy)

    return(np.array(energies))


def my_kernel_ridge(test_folder, training_folder, sigma = 1000):
    ''' Kernel ridge regression model
    y(X') = sum_i alpha_i K(X', X_i)

    Input
    -----
    M :  Representation Matrix

    Return
    ------
    ?
    '''
    '''jnp.linalg.invert(kernel_matrix) #(do this everytime?)

    K_{ij} = exp(- (||X_i-X_j||^2_2)/(2\sigma^2))
    This is the Gaussian Kernel matrix

    Regression koefficients are given by 
    \alpha = (K + \lambda I)-1 \cdot y

    
    '''
    set_size = 0

    for xyzfile in os.listdir(test_folder):
        if xyzfile.endswith(".xyz"):
            set_size += 1
    Identity = jnp.zeros((set_size , set_size))
    
    represented_training_set, information_training_set = read_files(test_folder)
    #get energies if necessary:
    energies = calculate_energies(information_training_set)
    
    #get kernel matrix
    K = build_kernel_matrix(represented_training_set, set_size, sigma)
    
    #get alpha coefficients
    alpha = get_alphas(K, energies, set_size)




def read_files(folder, representation = jrep.CM_eigenvectors_EVsorted):
    compound_list = []
    represented_list = []
    for xyzfile in os.listdir(folder):
        if xyzfile.endswith(".xyz"):
            xyz_fullpath = folder + xyzfile 
            compound = qml.Compound(xyz_fullpath)

            Z = compound.nuclear_charges.astype(float)
            R = compound.coordinates
            N = float(len(Z))
            atomtypes = compound.atomtypes
       
            compound_list.append([xyz_fullpath,Z, R, N, atomtypes])
            represented = representation(Z,R,N)
            represented_list.append(represented)

    return(represented_list, compound_list)

def gaussian_kernel(x,y, sigma):
    distance = jnp.subtract(x,y)
    absolute = jnp.linalg.norm(distance)
    k = jnp.exp(-(absolute**2)/(2*sigma**2))
    return(k)


def build_kernel_matrix(dataset_represented, dim, sigma, kernel_used = gaussian_kernel):
    empty = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            kij = kernel_used(dataset_represented[i], dataset_represented[j], sigma)
            empty[i][j] = kij
    K = jnp.asarray(empty)
    return(K)

def get_alphas(K, properties, dim, lamda = 1.0e-3):
    lamdaI = lamda*np.identity(dim)
    invertable = jnp.add(K, lamdaI)
    inverted = array(jnp.linalg.inv(invertable))
    alpha = jnp.multiply(inverted, properties)
    return(alpha)


my_kernel_ridge(datapath, datapath) 
