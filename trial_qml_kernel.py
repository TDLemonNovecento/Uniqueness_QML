import qml
import numpy as np
import database_preparation as datprep
import kernel_learning as kler
import plot_kernel as pltker
import representation_ZRN as ZRN_rep

#define datapath to a pickled list of compound instances
datapath = "./Pickled/qm7.pickle"

#how many compounds should be screened?
training_no = [2]#, 500, 1000, 2000, 3500]


#list of representations to be considered, 0 = CM, 1 = EVCM, 2 = BOB, 3 = OM, 4 = EVOM
representation_list = [0]#3, 4]# 2]


rep_names = ["CM", "EVCM", "BOB", "OM", "EVOM"]
kernel_class_paths = ["./tmp/CM_raw_Kernel_Results", "./tmp/EVCM_raw_Kernel_Results"]
#how many are to be predicted?
test_no =5 

#set hyperparameters, sigma is kernel width and lamda variation
sigma = [50.0, 800.0, 1500.0, 500.0, 1000.0] #sigmas for every representation
lamda = 1e-8

#hyperparameters for screening
sigma_list = [1, 3, 7, 17, 30, 67, 97, 160, 270, 450, 760, 1000, 1300]
lamda_list = [1e-8, 1e-10, 1e-12, 1e-14, 1e-16]

#get maximum number of compounds for which representations need to be calculated
total = training_no[-1] + test_no
this_job_name = "%i%.2e" %(test_no, lamda)
for i in representation_list:
    this_job_name += rep_names[i] + "_%.3f_" %(sigma[i])



#unpickle list of compounds
compound_list = datprep.read_compounds(datapath)
for i in representation_list:
    learning_list[i] = datprep.read_compounds[kernel_class_paths[i]]
CM_list = []




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
print("all data red and transformed to representation")

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
        
        m = datprep.Kernel_Result()
        m.sigma = sigma[rep]
        m.lamda = lamda
        m.representation_name = rep_names[rep]
        m.x_training = X_training[rep][no]
        m.x_test = X_test[rep][no]
        m.y_training = Y_training[no]
        m.y_test = Y_test[no]

        m.do_qml_gaussian_kernel()
        
        #save results
        learning_list.append(m)
        maes[rep].append(m.mae)


datprep.store_compounds(CM_list, "./tmp/CM_raw_Kernel_Results")

datprep.store_compounds(learning_list, "./tmp/Kernel_Results_%s" %this_job_name)


#plot results
plottable_maes = [maes[i] for i in representation_list]
labels = [rep_names[i] for i in representation_list]

pltker.plot_learning(set_sizes = training_no, maes = plottable_maes, labels = labels) 
print("sigmas:")
print(sigma)
print("maes:")
for i in representation_list:
    print("representation: ", rep_names[i])
    print(maes[i])
