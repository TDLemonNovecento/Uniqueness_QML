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
import pickle

final_file = "/home/stuke/Databases/Learning_Results/QM9_CM_eigenvec.obj"

class LearningResults:
    def __init__(self, lamda, sigma, set_sizes, maes):
        self.lamda = lamda
        self.sigma = sigma
        self.set_sizes = set_sizes
        self.maes = maes



def gaussian_kernel(x, y, sigma):
    distance = jnp.subtract(x,y)
    absolute = jnp.linalg.norm(distance)
    k = jnp.exp(-(absolute**2)/(2*sigma**2))
    return(k)


def build_kernel_matrix(dataset_represented, dim, sigma, kernel_used = gaussian_kernel):
    '''calculates kernel matrix K with 
    e.g. K_{ij} = exp(- (||X_i-X_j||^2_2)/(2\sigma^2))
    This is the Gaussian Kernel matrix
    Variables
    ---------
    dataset_represented: list of training data in a representation
    dim: int, number of training instances
    sigma : float, variable that is adapted during test runs
    kernel_used : function for kernel
    
    Return
    ------
    K : Kernel matrix
    '''
    	
    empty = np.zeros((dim,dim))
    
    #fill empty matrix
    for i in range(dim):
        for j in range(dim):
            kij = kernel_used(dataset_represented[i], dataset_represented[j], sigma)
            empty[i][j] = kij
    
    #convert numpy to jax numpy
    K = jnp.asarray(empty)
    
    return(K)

def get_alphas(K, properties, dim, lamda = 1.0e-3):
    '''calculates alpha coefficients
    Regression koefficients are given by 
    alpha = (K + \lambda I)-1 \cdot y
    
    Variables
    ---------
    K : Kernel Matrix based on training set
    properties: jax numpy array of properties related to training set
    dim: int, dimension of training set
    
    Return
    ------
    alphas : list of alpha coefficients
    '''
    lamdaI = lamda*np.identity(dim)
    invertable = jnp.add(K, lamdaI)
    inverted = jnp.linalg.inv(invertable)
    alphas = jnp.dot(inverted, properties)
    return(alphas)


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
        for mol in information_list:
            mol[5] = []
    #calculate total energy of molecules based on molecular information 
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
        energy = mol[5][0]
        #calculate atomic energy of molecule
        total_atomic_energy = 0
        for atom in Z:
            total_atomic_energy += atomic_energy_dictionary[atom]
        #atomization energy = total energy - atomic energy, add to list    
        atomization_energies.append(np.array(energy - total_atomic_energy))

    return(np.array(atomization_energies))


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


def read_xyz(folder, representation = jrep.CM_eigenvectors_EVsorted, get_energy = False):
    ''' Opens files and stores Z, R and N data + makes representation
    Variables
    ---------
    folder: Path to folder containing xyz files
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
    
    for xyzfile in os.listdir(folder):
        if xyzfile.endswith(".xyz"):
            filename = folder + xyzfile
            atoms = []
            R = []
            #open file and read lines
            with open(filename, 'r') as f:
                content = f.readlines()
        
            N = int(content[0])

            if get_energy:
                #in the QM9 database, the comment line contains information on the molecule. See https://www.nature.com/articles/sdata201422/tables/4 for more info.
                comment = content[1].split()
                #extract internal energy at 0K in Hartree
                zero_point_energy = comment[12]
            else:
                zero_point_energy = 0

            #read in atomic information from xyz file
            for line in range(2, N+2):
                atominfo = content[line].split()
                atoms.append(atominfo[0])
                R.append(np.array([float(atominfo[1]), float(atominfo[2]), float(atominfo[3])]))
        
            #transform to np arrays for further use
            Z = np.array([atomic_signs[atom] for atom in atoms])
            R = np.array(R)
            
            #create list of compound information and represented information
            compound_list.append([xyzfile,Z, R, N, atoms, zero_point_energy])
            represented = representation(Z,R,N)
            represented_list.append(represented)

    return(represented_list, compound_list)

def make_training_test(represented_list, compound_list, training_size, print_indices = False, upperlim = 1000):
    ''' splits data randomly into test and training set
    Variables
    ---------
    represented_list: list of training data in representation
    compund_list : list of other data ([filename, Z, R, N, atomnames, energy])
    training_size: int
                    desired size of training set
    print_indices: if True, print indices lists
    upperlim : int
                gives total of training+test set
    Return
    ------
    training_list: represented training set
    test_list: represented test set
    training_info : compound training info
    test_info : compound test info
    training_size: int, in case len of training_list is not equal to training_size
    '''
    dim = len(represented_list)
    #we are going to shuffle indices and then get the corresponding data from the
    #represented list and the compound list for our training and test set
    indices = [i for i in range(dim)]
    
    #make a random shuffle:
    random.shuffle(indices)
    
    if training_size >= dim:
        print("your training size exceeds or is equal to the number of files you provided\n Picking 1 file as test set")
        training_size = dim-1
    if upperlim > 0:
        if training_size >= upperlim:
            print("your training size exceeds or is equal to the upperlimit you set\n Picking 1 file as test set")
            training_size = upperlim-1

    #split randomly shuffled files into test and training set
    training_indices = indices[:training_size]
    if upperlim > 0:
        test_indices = indices[training_size:upperlim]
    else: 
        test_indices = indices[training_size:]
    
    if print_indices:
        print('training set:' , training_indices)
        print('test set:', test_indices)

    training_list = [represented_list[i] for i in training_indices]
    training_info = [compound_list[i] for i in training_indices]

    test_list = [represented_list[i] for i in test_indices]
    test_info = [compound_list[i] for i in test_indices]

    return([training_list, training_info, training_indices],[test_list, test_info, test_indices], training_size)

'''----------------------------------------------------
-							-
-	Kernel ridge regression function		-
-							-
----------------------------------------------------'''
def full_kernel_ridge(folder, result_file, set_sizes , sigmas = [], lambdas = [], rep_no = 1, representation = jrep.CM_eigenvectors_EVsorted):
    ''' Kernel ridge regression model
    y(X') = sum_i alpha_i K(X', X_i)
    
    Input
    -----
    folder :  path to folder containing .xyz files
    result_file : file where data is stored with pickle. 
    training_size : desired size of training set
    sigma : fitting coefficient

    Return
    ------
    learning_list : list of LearningResults Objects
    raw_data_files : list of names of files where raw data was stored to
    

    Stored
    ------
    raw data is stored to raw_data_file entries, learning_list is sotred to result_file
    '''

    learning_list = []
    raw_data_files = []
    #get representations and other data once and for all
    represented, compound = read_xyz(folder, representation)

    
    for i in range(rep_no):
        for s in sigmas:
            for l in lambdas:
                #for every i, s, l combination, a new Learning Object is created and stored to the learning list
                maes = []

                for sets in set_sizes:
                    #make training and test list:
                    training, test, tr_size = make_training_test(represented, compound, sets)
                    
                    tr_list = training[0]
                    tr_info = training[1]
                    tr_fileindex = training[2]

                    tst_list = test[0]
                    tst_info = test[1]
                    tst_fileindex = test[2]
                    
                    #get energies if necessary:
                    tr_energies = calculate_energies(tr_info)
                    tst_energies = calculate_energies(tst_info)

                    K = build_kernel_matrix(tr_list, tr_size, s)
                    
                    #get alpha coefficients
                    alphas = get_alphas(K, tr_energies, tr_size, l)

                    #predict properties of test set
                    results, errors = predict_new(s, alphas, tr_list, tst_list, tst_energies)
                    mae = sum(abs(errors))/(len(errors))
                    maes.append(mae)

                    #save raw data
                    filename = folder + '/rawdata_rep%i_sigma%s_lamda%f_set%i.dat' %(i, str(s), l, sets)
                    raw_data_files.append(filename)
                    save_raw_data(filename, tr_energies, tr_fileindex, tst_energies, results, tst_fileindex)
                    
                #add learning result to list
                learning_list.append(LearningResults(l, s, np.array(set_sizes), np.array(maes)))
        print("round %i successfully finished" % (i+ 1))
    
    #save maes with data so it can be plotted
    save_all_Results(learning_list, result_file) 
    

    return(learning_list, raw_data_files)
    
    
'''-------------------------------------------'''
'''						'''
'''functions for saving Results and raw data	'''
'''						'''
'''						'''
'''-------------------------------------------'''
def save_all_Results(results_list, filename = final_file):
    f = open(filename, 'wb')
    pickle.dump(results_list, f)
    f.close()
    return(print('results were saved'))
    
def get_all_Results(filename = final_file):
    '''
    Variables
    ---------
    filename: path to .obj file that contains pickled list of Results objects

    Returns: list of Results objects
    '''

    results_list = []
    f = open(filename, 'rb')
    
    while 1:
        try:
            results_list.append(pickle.load(f))
        except EOFError:
            break
    f.close()
    
    return(results_list[0])

def save_single_Result(result, filename = final_file):
    results_list = get_all_Results(filename)
    results_list.append(result)
    save_all_Results(results_list, filename)
    return(print('result was saved'))

def save_raw_data(filename, training_energy, training_fileindex, test_energy, predicted_energies, test_fileindex):
    f = open(filename, 'w+')
    f.write('no_training:\t%i\tno_test:\t%i' %(len(training_fileindex), len(test_fileindex)))
    f.write('file identifier\t training data')
    for i in range(len(training_fileindex)):
        f.write('%s\t%f' %(str(training_fileindex[i]), training_energy[i])) 
    f.write('\nfile identifier \t test data \t predicted')
    for i in range(len(test_fileindex)):
        f.write('%s\t%f\t%f' %(str(test_fileindex[i]), test_energy[i], predicted_energies[i]))
    f.close()
    return(print('raw data was saved to ', filename))
