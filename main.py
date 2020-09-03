#In this program routines and subroutines are called depending on what shoud be done. Unused parts should be hashed and parts that conflict with each other marked with exclamation marks and special characters respectively twice that character. Written by Miriam Stuke.
### signifies sympy use, other approach than jax
import os
import qml
import jax_representation
import jax_derivative
###import sympy as sp
import numpy as np
import sys
from jax import grad

print("I see you")

#define path to folder containing xyz files. All files are considered.
database = "/home/stuke/Databases/XYZ_random/"
###f = repro.Eigenvalue_CM_FM(R, Z, N, 3, 1, [0, 0, 0], [0, 0, 0])


print("iterate over all molecules")
for xyzfile in os.listdir(database):
    xyz_fullpath = database + xyzfile #probably path can be gotten more directly
    compound = qml.Compound(xyz_fullpath)

    print("compound %s" % xyzfile)
    iZ = compound.nuclear_charges
    cZ = iZ.astype(float)
    print(type(cZ))
    cR = compound.coordinates
    cN = float(len(cZ))
    
###    print("calculate Coulomb Matrix Representations")
###    f = repro.Coulomb_Matrix_FM(R, Z, N, cN , 0, [0, 0, 0], [0, 0, 0])
###    print("Representation we're currently working with:")
###    print(f)

###    print("Filling in values from %s gives the following result:" % xyzfile)
###    eval_f = repro.subs_CM(f, cZ, cR, cN)    
###    print(eval_f)
           

    '''in order to evaluate the derivative of the CM, the following code can be used'''
    CM_sorted, order = jax_representation.CM_full_sorted(cZ, cR)
    dim = len(order) #dimension of CM
    print("the CM Matrix of your compound is:")
    print(np.asarray(CM_sorted)) 
    derive_by = 0 

    if derive_by == 0:
        CM_d_sorted = np.zeros((dim, dim, dim))
        print("derivation in order  by ", cZ)
    elif derive_by == 1:
        CM_d_sorted = np.zeros((dim,dim,dim,3))
    else:
        CM_d_sorted = np.zeros((dim,dim))
        print("you'll only ever get 0 cause the CM is not dependent on the total number of electrons N")

    for i in range(dim):
        for j in range(dim):
            der_ij = grad(jax_representation.CM_index, derive_by)
            new_value = der_ij(cZ, cR, i, j)
            dd_ij = new_value
            CM_d_sorted[i][j] = dd_ij

    print(CM_d_sorted)
