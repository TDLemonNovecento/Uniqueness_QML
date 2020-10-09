'''do automatic differentiation all the way'''
import jax.numpy as jnp 
import jax_representation as jrep
import numpy as np
import qml
import os
import random
from xtb.interface import Calculator
from xtb.utils import get_method

final_file = "/home/stuke/Databases/QM9_XYZ_below10/learning_data.dat"

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
    energies : np.array with energies of the moelcules
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


def my_kernel_ridge(folder, training_size, sigma = 1000):
    ''' Kernel ridge regression model
    y(X') = sum_i alpha_i K(X', X_i)

    Input
    -----
    test_folder :  List of paths to .xyz files for testing
    training_folder : List of paths to .xyz files for training
    sigma : fitting coefficient
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
    print('results', results)
    print('errors', errors)
    print('test energies', test_energies)

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
    print('there are ', len(list_represented_test), ' items for which properties are being calculated')
    for test_X in list_represented_test:
        test_prop = 0.0
        for alpha_i, training_i in zip(alphas, list_represented_training):
            i_element = alpha_i * gaussian_kernel(test_X, training_i, sigma)
            print('i element', i_element)
            test_prop += i_element
            print('test_prop after this i', test_prop)
        
        predicted_properties.append(test_prop)
    
    print('number of predicted properties is ', len(predicted_properties), '\n', predicted_properties)
    prediction_errors = np.subtract(test_properties, np.array(predicted_properties))
    
    return(np.array(predicted_properties), np.array(prediction_errors))


def read_files(file_list, representation = jrep.CM_eigenvectors_EVsorted):
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
        compound = qml.Compound(xyzfile)

        Z = compound.nuclear_charges.astype(float)
        R = compound.coordinates
        N = float(len(Z))
        atomtypes = compound.atomtypes
       
        compound_list.append([xyzfile,Z, R, N, atomtypes])
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
    lamdaI = lamda*np.identity(dim)
    invertable = jnp.add(K, lamdaI)
    inverted = jnp.linalg.inv(invertable)
    alpha = jnp.multiply(inverted, properties)
    return(alpha)


my_kernel_ridge(datapath, 10) 
