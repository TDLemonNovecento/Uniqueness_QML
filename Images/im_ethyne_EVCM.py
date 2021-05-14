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
database_raw = "/home/linux-miriam/Databases/Ethyne_invariant/Raw/"

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

EVCM_aberration = []
EVCM_aberration2 = []

M_list = [[],[],[],[]]
M_list2 = [[],[],[],[]]


for i in indexes:
    num = str(i).zfill(2)
    filename = datapath + "ethin_" + num + ".xyz"
    filename2 = database_raw + "ethin_" + num + ".xyz"
    
    print(filename, filename2)
    compound = qml.Compound(filename)
    compound2 = qml.Compound(filename2)

    Z = compound.nuclear_charges.astype(float)
    R = compound.coordinates
    
    Z2 = compound2.nuclear_charges.astype(float)
    R2 = compound2.coordinates

    #calculate difference to reference constitution
    M_EVCM = jrep.CM_ev_unsrt(Z, R, N = 0, size = 4)
    M_EVCM2 = jrep.CM_ev_unsrt(Z2, R2, N = 0, size = 4)

    diff = (ref_M_EVCM - M_EVCM)
    diff2 = (ref_M_EVCM - M_EVCM2)

    error = 0
    error2 = 0

    for i in range(len(diff)):
        d = diff[i]
        d2 = diff2[i]
        error += d*d
        error2 = d2*d2

    EVCM_aberration.append(error)
    EVCM_aberration2.append(error2)


    
    #collect all dZ for plotting
    for j in range(4):
        M_list[j].append(M_EVCM[j])
        M_list2[j].append(M_EVCM2[j])
            

#dZ1 is dZ[0] and so forth, it contains a list of arrays in 4 dimensions
#for every atom, extract it's data

#eigenvalues per atom
atom1 = np.array(M_list[0])
atom2 = np.array(M_list[1])
atom3 = np.array(M_list[2])
atom4 = np.array(M_list[3])

#eigenvalues per atom
atom1_2 = np.array(M_list2[0])
atom2_2 = np.array(M_list2[1])
atom3_2 = np.array(M_list2[2])
atom4_2 = np.array(M_list2[3])


#prepare list of lists for plotting

#add error of EVCM list compared to reference to lineplots
linevalues = [[np.asarray(EVCM_aberration), "MSE to Reference"]]



pltder.plot_ethyne2(indexes,\
        [[atom1, "Eigenvalue 1"], [atom2, "Eigenvalue 2"],\
        [atom3, "Eigenvalue 3"], [atom4, "Eigenvalue 4"]],\
        [[atom1_2, "Unoptimized"], [atom2_2, ""], [atom3_2 , ""], [atom4_2, ""]],\
         savetofile = "./Images/Final/ethyne_EVCM.png", plot_title = False, cutout_15_75 = True)
