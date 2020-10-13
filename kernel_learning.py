'''do automatic differentiation all the way'''
import jax.numpy as jnp 
import jax_representation as jrep
import numpy as np
import qml
import os
import random
from xtb.interface import Calculator
from xtb.utils import get_method
from basis import atomic_energy_dictionary, atomic_signs

final_file = "/home/stuke/Databases/QM9_XYZ_below10/learning_data.dat"

datapath = "/home/stuke/Databases/XYZ_ethin/"

#representation to be used: CM_eigenvectors_EVsorted(Z, R, N, cutoff = 8)
def calculate_energies(information_list, has_energies = False ):
    '''
    Variables
    ---------
    information_list : list of lists, with
                       information_list[i] = ['name of file', [Z], [R], [N], [atomtypes], [energies] ]
                       [Z].type and [R].type is np.array
    has_energies : True if energies are at end of information_list
    Returns
    -------
    energies : np.array with atomization energies of the moelcules
    '''

    if has_energies == False:
        mol[5] = []
    #calculate total energy of molecules based on molecular information 
        for mol in information_list:
            Z = mol[1]
            R = mol[2]
            #create xtb calculator
            calc = Calculator(get_method("GFN2-xTB"), Z, R)
            results = calc.singlepoint()
            energy = results.get_energy()
            mol[5].append(energy)

    #make new list to store all atomization energies        
    atomization_energies = []

    #calculate atomization energy for every molecule
    for mol in information_list:
        Z = mol[1]
        energy = mol[5]
        #calculate atomic energy of molecule
        total_atomic_energy = 0
        for atom in Z:
            total_atomic_energy += atomic_energy_dictionary[atom]
        #atomization energy = total energy - atomic energy, add to list    
        atomization_energy.append(np.array(energy - total_atomic_energy))

    return(np.array(atomization_energies))


def my_kernel_ridge(folder, training_size, sigma = 1000):
    ''' Kernel ridge regression model
    y(X') = sum_i alpha_i K(X', X_i)
    
    K_{ij} = exp(- (||X_i-X_j||^2_2)/(2\sigma^2))
    This is the Gaussian Kernel matrix

    Regression koefficients are given by 
    \alpha = (K + \lambda I)-1 \cdot y


    Input
    -----
    folder :  path to folder containing .xyz files
    training_size : desired size of training set
    sigma : fitting coefficient

    Return
    ------
    training_energies: np array of energies for training set for learning
    test_energies: np array of energies for test set as reference
    predicted_energies: np array of energies
    prediction_errors
    '''

    Identity = jnp.zeros((training_size , training_size))
    
    #make training and test list:
    training_list, test_list, training_size = make_training_test(folder, training_size)

    #get representations and other data
    represented_training_set, information_training_set = read_files(training_list)
    represented_test_set, information_test_set = read_files(test_list)
    
    #get energies if necessary:
    training_energies = calculate_energies(information_training_set)
    test_energies = calculate_energies(information_test_set)

    #get kernel matrix
    K = build_kernel_matrix(represented_training_set, training_size, sigma)
    
    #get alpha coefficients
    alphas = get_alphas(K, training_energies, training_size)

    results, errors = predict_new(sigma, alphas, represented_training_set, represented_test_set, test_energies)
    type(errors)
    return(training_energies, test_energies, results, errors)

def gaussian_kernel(x, y, sigma):
    distance = jnp.subtract(x,y)
    absolute = jnp.linalg.norm(distance)
    k = jnp.exp(-(absolute**2)/(2*sigma**2))
    return(k)

def predict_new(sigma, alphas, list_represented_training, list_represented_test, test_properties, kernel_function = gaussian_kernel):
    '''Calculates property of new compound via formula
    y(X_new) = \sum_i alpha_i kernelfunction(X_new, X_i)
    Variables
    ---------
    sigma: float
    alphas: numpy array of alpha coefficients
    list_represented_training: list of training data in a certain reperesentation
    list_represented_test: same for test data
    test_properties: np array, calculated test properties
    kernel_function: function, e.g. Gaussian Kernel or Laplacian or whatever
    
    Returns
    -------
    predicted_properties: numpy array of predicted properties
    prediction_errors: numpy array of difference between predicted and expected property
    '''
    print("calculating new properties")
    predicted_properties = []
    
    for test_X in list_represented_test:
        test_prop = 0.0
        for alpha_i, training_i in zip(alphas, list_represented_training):
            i_element = alpha_i * gaussian_kernel(test_X, training_i, sigma)
            test_prop += i_element
        
        predicted_properties.append(test_prop)
    
    prediction_errors = np.subtract(test_properties, np.array(predicted_properties))
    
    return(np.array(predicted_properties), np.array(prediction_errors))


def read_xyz(file_list, representation = jrep.CM_eigenvectors_EVsorted):
    ''' Opens files and stores Z, R and N data + makes representation
    Variables
    ---------
    file_list: List of strings that are paths to .xyz files
    representation: some kind of representation that can work with Z, R, N data

    Returns
    -------
    represented_list: List of representations of the files in file_list
    compound_list: List of information on files in file_list
                    [name of file, Z, R, N]
                    with Z, R and N being numpy arrays and the
                    name of file being a string
    '''

    compound_list = []
    represented_list = []
    
    for xyzfile in file_list:
        atoms = []
        R = []
        #open file and read lines
        with open(xyzfile, 'r') as f:
            content = f.readlines()
        
        N = int(content[0])

        #in the QM9 database, the comment line contains information on the molecule. See https://www.nature.com/articles/sdata201422/tables/4 for more info.
        comment = content[1].split()
        #extract internal energy at 0K in Hartree
        zero_point_energy = comment[12]

        #read in atomic information from xyz file
        for line in range(2, N+2):
            atom, x, y, z, mulliken_charge = content[line].split()
            atoms.append(atom)
            R.append(np.array([float(x), float(y), float(z)]))
        
        #transform to np arrays for further use
        Z = np.array([atomic_signs[atom] for atom in atoms])
        R = np.array(R)

        #create list of compound information and represented information
        compound_list.append([xyzfile,Z, R, N, atoms, zero_point_energy])
        represented = representation(Z,R,N)
        represented_list.append(represented)

    return(represented_list, compound_list)

def make_training_test(folder, training_size):
    ''' reads file names and splits them randomly into test and training set
    Variables
    ---------
    folder: string
            path to folder containing molecular data (xyz)
    training_size: int
                    desired size of training set
    Return
    ------
    training_list: List of paths to training files
    test_list: List of paths to test files
    training_size: int, in case len of training_list is not equal to training_size
    '''

    file_list = []
    for xyzfile in os.listdir(folder):
        if xyzfile.endswith(".xyz"):
            xyz_fullpath = folder + xyzfile
            file_list.append(xyz_fullpath)
    #make a random shuffle:
    random.shuffle(file_list)
    
    if training_size >= len(file_list):
        print("your training size exceeds or is equal to the number of files you provided\n Picking 1 file as test set")
        training_size = len(file_list)-1
    
    #split randomly shuffled files into test and training set
    training_list = file_list[:training_size]
    test_list = file_list[training_size:]
    print('test_list has size', len(test_list), 'training list length', len(training_list), 'original list', len(file_list))
    
    return(training_list, test_list, training_size)


def build_kernel_matrix(dataset_represented, dim, sigma, kernel_used = gaussian_kernel):
    empty = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            kij = kernel_used(dataset_represented[i], dataset_represented[j], sigma)
            empty[i][j] = kij
    K = jnp.asarray(empty)
    return(K)

def get_alphas(K, properties, dim, lamda = 1.0e-3):
    '''calculates alpha coefficients
    Variables
    ---------
    K : Kernel Matrix based on training set
    properties: jax numpy array of properties related to training set
    dim: int, dimension of training set
    '''
    lamdaI = lamda*np.identity(dim)
    invertable = jnp.add(K, lamdaI)
    inverted = jnp.linalg.inv(invertable)
    alpha = jnp.dot(inverted, properties)
    return(alpha)



#my_kernel_ridge(datapath, 10) 
