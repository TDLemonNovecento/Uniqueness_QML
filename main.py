#In this program routines and subroutines are called depending on what shoud be done. Unused parts should be hashed and parts that conflict with each other marked with exclamation marks and special characters respectively twice that character. Written by Miriam Stuke.
### signifies sympy use, other approach than jax
import os
import qml
import jax_representation as jrep
import jax_derivative as jder
import jax_norms as jnorm
###import sympy as sp
import numpy as np
import sys
from jax import grad, jacfwd, jacrev

def hessian(f):
    return( jacfwd(jacfwd(f)))

#define path to folder containing xyz files. All files are considered.
database = "/home/stuke/Databases/XYZ_random/"
###f = repro.Eigenvalue_CM_FM(R, Z, N, 3, 1, [0, 0, 0], [0, 0, 0])


print("iterate over all molecules")
for xyzfile in os.listdir(database):
    xyz_fullpath = database + xyzfile #probably path can be gotten more directly
    compound = qml.Compound(xyz_fullpath)

    print("compound %s" % xyzfile)
    iZ = compound.nuclear_charges
    Z = iZ.astype(float)
    R = compound.coordinates
    N = float(len(Z))
    
###    print("calculate Coulomb Matrix Representations")
###    f = repro.Coulomb_Matrix_FM(R, Z, N, cN , 0, [0, 0, 0], [0, 0, 0])
###    print("Representation we're currently working with:")
###    print(f)

###    print("Filling in values from %s gives the following result:" % xyzfile)
###    eval_f = repro.subs_CM(f, cZ, cR, cN)    
###    print(eval_f)
           
    HM = jder.sort_derivative('CM', Z, R)
    print('derivative', HM)
        

'''in order to evaluate the derivative of the CM, the following code can be used'''
    
'''dim = len(order) #dimension of CM
    print("the CM Matrix of your compound is:")
    print(np.asarray(CM_sorted))
    
    print("direct derivative using jacobian:")
    dCM_dZ = jacfwd(jax_representation.CM_full_sorted)
    reference_dCM_dZ = dCM_dZ(cZ, cR)[0]
    print(np.asarray(reference_dCM_dZ))
    #assign derivative to correct field in sorted matrix by reordering derZ_ij to derZ_kl
    dCMkl_dZkl = [[[column[m] for m in order] for column in row]for row in reference_dCM_dZ]
    print("sorted derivatives\n", np.asarray(dCMkl_dZkl))
     
    

    print("second derivative")
    #print(H[0][0][0])
    #print(H[0][0][1])
    print("order is:", order)
    for i in range(dim):
        for j in range(dim):
            H_ZijdZij = Hraw[0][i][j]
            print("unsorted is:", H_ZijdZij)
            H_ZijdZkl = [[row[m] for m in order] for row in H_ZijdZij]
            H_ZkldZkl = print("I fucking don't know")
            print("sorted:", H_ZkldZkl)
            #H_ZkldZkl = [[[column[m] for m in order] for column in row]for row in H_ZijdZij]
            #Hformatted[i][j] = H_ZkldZkl
    print(Hformatted)
    sys.exit()

    print("shape of hessian tupleoutput is:", type(H[0][0]), type(H[0][1]), type(H[1][0]))
    print("elements in hessian tuple output:", len(H[0]))



'''



def trialfunction():
    print("and this is the manually done one:")

    
    '''from here on i,j is used for unsorted basis and k,l for sorted basis'''
    derZ_ij = grad(jax_representation.CM_index, 0)
    derR_ij = grad(jax_representation.CM_index, 1)
    derN_ij = grad(jax_representation.CM_index, 2)
    
    CM_derZ_sorted = np.zeros((dim, dim, dim))
    CM_derR_sorted = np.zeros((dim,dim,dim,3))
    CM_derN_sorted = np.zeros((dim,dim))

    for k in range(dim):
        for l in range(dim):
            #i = order[k], j = order[l]. we're mapping to k,l here
            dZunsort_kl = derZ_ij(cZ, cR, cN, order[k], order[l])
            #assign derivative to correct field in sorted matrix by reordering derZ_ij to derZ_kl
            dZsort_kl = np.asarray([dZunsort_kl[m] for m in order])
            
            
        

            #print("order is:", order)
            #print("unsorted:", dZunsort_kl)
            #print("sorted  :", dZsort_kl)
            CM_derZ_sorted[k][l] = dZsort_kl
            
            der_ij = grad(jax_representation.CM_index, derive_by)
            new_value = der_ij(cZ, cR, i, j)
            dd_ij = new_value
            CM_d_sorted[i][j] = dd_ij

    print(CM_derZ_sorted)
