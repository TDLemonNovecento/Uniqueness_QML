import os, time
import jax_derivative as jder
import qml
import numpy as np
import plot_derivative as pltder
import jax_representation as jrep
import numerical_derivative as numder

#path to xyz files
database_opt = "/home/linux-miriam/Databases/Ethyne_invariant/Optimized/"
ethin_image = "/home/linux-miriam/Databases/Ethyne_invariant/ehtyne.png"
database_raw = "/home/linux-miriam/Databases/Ethyne_invariant/Raw"

'''define folder of .xyz files'''
datapath = database_opt
reference = database_raw + "/ethin_00.xyz"

indexes = range(20)

print(indexes)

#the ethin files are sorted by index. the numbering shows how big angle theta is:
'''we start with a straight molecule stretching out on the x axis:

      H--C===C--H

to then move both H simultaneously anti-clockwise by an angle phi:
    
                H
               / phi
         C===C.......
        /
        H

'''

#calculate reference values
compound = qml.Compound(reference)
Z = compound.nuclear_charges.astype(float)
R = compound.coordinates

ref_M_EVCM = jrep.CM_ev_unsrt(Z, R, size = 4)


dZ_eigenvalues = [] #list of eigenvalue vectors. length is same as len of name_vector
dimZ_list = [] #dimension of files may vary. store all dimensions here


dZ_slot1_list = [[],[],[],[]]
dZ_slot2_list = [[],[],[],[]]
dZ_slot3_list = [[],[],[],[]]
dZ_slot4_list = [[],[],[],[]]

EVCM_aberration = []

M_list = [[],[],[],[]]

for i in indexes:
    num = str(i).zfill(2)
    filename = datapath + "ethin_" + num + ".xyz"
    compound = qml.Compound(filename)
    Z = compound.nuclear_charges.astype(float)
    R = compound.coordinates
    
    #calculate difference to reference constitution
    M_EVCM = jrep.CM_ev_unsrt(Z, R, N = 0, size = 4)

    diff = (ref_M_EVCM - M_EVCM)
    error = 0
    for d in diff:
        error += d*d

    EVCM_aberration.append(error)


    #calculate CM
    M_CM = jrep.CM_full_unsorted_matrix(Z, R, N=0, size = 4)
    
    #collect all dZ for plotting
    for j in range(4):
        dZj = numder.derivative(jrep.CM_ev_unsrt,[Z, R, 0], order = 1, d1 = [0,j])
        dZ_slot1_list[j].append(dZj[0])
        dZ_slot2_list[j].append(dZj[1])
        dZ_slot3_list[j].append(dZj[2])
        dZ_slot4_list[j].append(dZj[3])
        M_list[j].append(M_EVCM[j])
            

#dZ1 is dZ[0] and so forth, it contains a list of arrays in 4 dimensions
#for every atom, extract it's data

#eigenvalues per atom
atom1 = np.array(M_list[0])
atom2 = np.array(M_list[1])
atom3 = np.array(M_list[2])
atom4 = np.array(M_list[3])

print("atom1 array:", atom1)


#prepare list of lists for plotting
yvalues = []

for i in range(4):
    yvalues.append([np.asarray(dZ_slot1_list[i]), "dZ%i D1" %(i + 1)])
    yvalues.append([np.asarray(dZ_slot2_list[i]), "dZ%i D2" %(i + 1)])
    yvalues.append([np.asarray(dZ_slot3_list[i]), "dZ%i D3" %(i + 1)])
    yvalues.append([np.asarray(dZ_slot4_list[i]), "dZ%i D4" %(i + 1)])

#add error of EVCM list compared to reference to lineplots
linevalues = [[np.asarray(EVCM_aberration), "MSE to Reference"]]


pltder.plot_ethyne(indexes, yvalues, \
        savetofile = "./Images/Final/dZ_Ethyne.png",\
        #lineplots = linevalues,\
        plot_title = False,\
        plot_dZ = True)

pltder.plot_ethyne(indexes, [[atom1, "Eigenvalue 1"], [atom2, "Eigenvalue 2"], [atom3, "Eigenvalue 3"], [atom4, "Eigenvalue 4"]],\
        savetofile = "trial_EVCM.png", plot_title = False)
