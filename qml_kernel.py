import qml
from qml.kernels import gaussian_kernel
from qml.math import cho_solve

import numpy as np
import database_preparation as datprep
import kernel_learning as kler
import plot_kernel as pltker
import representation_ZRN as ZRN_rep

#define datapath to a pickled list of compound instances
datapath = "./Pickled/qm7.pickle"

#how many compounds should be screened?
training_no = [100, 500, 1000, 2000, 3500]

#list of representations to be considered, 0 = CM, 1 = EVCM, 2 = BOB, 3 = OM, 4 = EVOM
representation_list = [0, 1]# 2]
rep_names = ["CM", "EVCM", "BOB", "OM", "EVOM"]


#how many are to be predicted?
test_no = 400

#set hyperparameters, sigma is kernel width and lamda variation
sigma = [1700.0, 1000.0, 1500.0, 1000.0, 1000.0] #sigmas for every representation
lamda = 1e-8

#get maximum number of compounds for which representations need to be calculated
total = training_no[-1] + test_no


#unpickle list of compounds
compound_list = datprep.read_compounds(datapath)

#make list with all the representations
X_list = [[],[],[],[],[]]
Y_energy_list = []


for c in range(total):
    compound = compound_list[c]
    
    # Create the compound object mol from the file qm7/0001.xyz which happens to be methane
    mol = qml.Compound()
    mol.natoms = len(compound.Z)
    mol.nuclear_charges = compound.Z
    mol.coordinates = compound.R

    #calculate atomization energy from internal energy and add to qml.Compound class object
    atomization_energy = datprep.atomization_energy(potential_energy = float(compound.energy), nuclear_charges = compound.Z)
    #mol.energy = atomization_energy
    Y_energy_list.append(atomization_energy)
    

    # Generate and representations and append to X_list
    for rep in representation_list:
        if rep == 0:
            mol.generate_coulomb_matrix()

        if rep == 1:
            M = ZRN_rep.Eigenvalue_Coulomb_Matrix_h(compound.Z, compound.R)
            mol.representation = M.flatten()

        if rep == 2:
            mol.generate_bob(asize={'C':7, 'H': 16, 'N':6, 'O': 4, 'F': 4})

        if rep == 3:
            M = ZRN_rep.Overlap_Matrix_h(compound.Z, compound.R)
            mol.representation = M.flatten()

        if rep == 4:
            M = ZRN_rep.Eigenvalue_Overlap_Matrix_h(compound.Z, compound.R)
            mol.representation = M.flatten()


        #add representation array to X_list
        X_list[rep].append(mol.representation)



'''prepare training and test sets '''

training_list = []
test_list = []


for no in training_no:
    #randomly divide data into training and test set
    training, test = kler.make_training_test(total, training_size = no, upperlim = no + test_no)
    
    training_list.append(training)
    test_list.append(test)


X_training = [[],[],[],[],[]]
X_test = [[],[],[],[],[]]

Y_training = []
Y_test = []


for no in range(len(training_no)):
    
    Y_training.append(np.array([Y_energy_list[t] for t in training_list[no]]))
    Y_test.append(np.array([Y_energy_list[t] for t in test_list[no]]))

    for rep in representation_list:
        #make 2D arrays
        X_training[rep].append(np.array([X_list[rep][t] for t in training_list[no]]))
        X_test[rep].append(np.array([X_list[rep][t] for t in test_list[no]]))

'''create Kernel, run learning and store results '''
maes = [[],[],[],[],[]] #maes for each representation (CM, EVCM, BOB, OM, EVOM)

for rep in representation_list:
    for no in range(len(training_no)):
    
        #K is also a np array, create kernel matrix
        K = gaussian_kernel(X_training[rep][no], X_training[rep][no], sigma[rep])
        
        #add small lambda to the diagonal of the kernel matrix
        K[np.diag_indices_from(K)] += lamda

        #use the built in Cholesky-decomposition to solve
        alpha = cho_solve(K, Y_training[no])
    

        #predict new, calculate kernel matrix between test and training
        Ks = gaussian_kernel(X_test[rep][no], X_training[rep][no], sigma[rep])

        #make prediction
        Y_predicted = np.dot(Ks, alpha)
    
        # Calculate mean-absolute-error (MAE):
        mae = np.mean(np.abs(Y_predicted - Y_test[no]))
        maes[rep].append(mae)


plottable_maes = [maes[i] for i in representation_list]
labels = [rep_names[i] for i in representation_list]

pltker.plot_learning(set_sizes = training_no, maes = plottable_maes, labels = labels) 
print("sigmas:")
print(sigma)
print("maes:")
for i in representation_list:
    print("representation: ", rep_names[i])
    print(maes[i])
