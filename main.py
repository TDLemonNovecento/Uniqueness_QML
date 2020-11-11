'''In this program routines and subroutines are called depending on what shoud be done. Unused parts should be hashed and parts that conflict with each other marked with exclamation marks and special characters respectively twice that character. Written by Miriam Stuke.
'''
import database_preparation as datprep
import jax_derivative as jder
import jax_representation as jrep
import jax.numpy as jnp

'''
PROBLEM:
    Sorting in derivative file takes a lot of time
    Example from running on a <5ha atom, CM dim: 9

DDZ TIMES:
calculation time:  -1.3426744937896729 reordering time:  -13.307524681091309
I am calculating the second derivative
you want to calculate dZdR or dRdZ
DRDZ TIMES:
 calculation time:  -1.2504057884216309 reordering time:  -46.016727685928345
I am calculating the second derivative
do dRdR sorting
DDR TIMES:
calculation time:  -1.537632703781128 reordering time:  -169.21404123306274

'''

#define path to folder containing xyz files. All files are considered.
database = "/home/linux-miriam/Databases/QM9_XYZ/"
database_file = "/home/linux-miriam/Uniqueness_QML/Pickled/qm9.pickle"
dat_ha_file = "/home/linux-miriam/Uniqueness_QML/Pickled/qm7.pickle"
trial_file = "/home/linux-miriam/Uniqueness_QML/Pickled/XYZ_random_ha5.txt"
result_file = "/home/linux-miriam/Databases/results.pickle"


#qm7_compounds = datprep.read_compounds(dat_ha_file)
compound_ls = datprep.read_compounds(trial_file)
def: 
eigenvalues = []
for c in compound_ls:
    Z = jnp.asarray([float(i)for i in c.Z])
    R = c.R
    N = float(c.N)

    CM, order = jrep.CM_full_sorted(Z, R, N)
    dim = CM.shape[0]
    
    dZ = jder.sort_derivative('CM', Z, R, N, 1, 'Z')
    dR = jder.sort_derivative('CM', Z, R, N, 1, 'R')
    
    ddZ = jder.sort_derivative('CM', Z, R, N, 2, 'Z', 'Z')
    dZdR = jder.sort_derivative('CM', Z, R, N, 2, 'R', 'Z')
    ddR = jder.sort_derivative('CM', Z, R, N, 2, 'R', 'R')

    #calculate dZ eigenvalues
    dZ_ev = []
    dR_ev = []

    ddZ_ev = []
    dZdR_ev = []
    ddR_ev = []
    
    for i in range(dim):
        print(dZ[i].shape)
        eigenvals, eigenvec = jnp.linalg.eig(dZ[i])
        dZ_ev.append(eigenvals)

        for x in range(3):
            eigenvals, eigenvec = jnp.linalg.eig(dR[i,x])
            dR_ev.append(eigenvals)

            for j in range(dim):
                eigenvals, eigenvec = jnp.linalg.eig(dZdR[i, j, x])
                dZdR_ev.append(eigenvals)

                for y in range(3):
                    eigenvals, eigenvec = jnp.linalg.eig(ddR[i, x, j, y])
                    ddR_ev.append(eigenvals)
        
        for j in range(dim):
            eigenvals, eigenvec = jnp.linalg.eig(ddZ[i,j])
            ddZ_ev.append(eigenvals)

    #now save results somewhere
    der_result = datprep.derivative_results(c.filename)
    der_result.add_all_RZev(dZ_ev, dR_ev, ddZ_ev, ddR_ev, dZdR_ev)

    eigenvalues.append(der_result)


datprep.store_compounds(eigenvalues, result_file)
            

    

