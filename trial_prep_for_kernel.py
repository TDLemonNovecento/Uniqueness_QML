import qml
import numpy as np
import database_preparation as datprep
import kernel_learning as kler
import plot_kernel as pltker
import representation_ZRN as ZRN_rep
""""
this file creates pickled lists of Kernel_Result class objects with the represented information
"""


calculated = datprep.read_compounds("./tmp/BOB_raw_Kernel_Results")

print("len of list: ", len(calculated))
print("first element:", calculated[0])
print("CM of first element:", calculated[0].representation_name)
print(calculated[0].x[0])
print("energy:", calculated[0].y[0])


#define datapath to a pickled list of compound instances
datapath = "./Pickled/qm7.pickle"

#list of representations to be considered, 0 = CM, 1 = EVCM, 2 = BOB, 3 = OM, 4 = EVOM
rep = 4

rep_names = ["CM", "EVCM", "BOB", "OM", "EVOM"]

#get maximum number of compounds for which representations need to be calculated
total = 3993

#unpickle list of compounds
compound_list = datprep.read_compounds(datapath)
learning_list = []
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



#prepare Kernel_Result raw instances (no training/test split, no sigma, no lamda)


m = datprep.Kernel_Result()
m.representation_name = rep_names[rep]
m.x = X_list[rep]
m.y = Y_energy_list
        
CM_list.append(m)


datprep.store_compounds(CM_list, "./tmp/%s_raw_Kernel_Results" %rep_names[rep])

