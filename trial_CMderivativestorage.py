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

problem resolved (example: H2O): 
DDZ Times:
slow ordering:  -13.042343378067017 fast ordering:  -0.0732736587524414

DDR TIMES:
slow calculation:  -165.17349076271057 fast_calculation:  -0.31604862213134766

dZDR Times:
slow ordering:  -46.52089214324951 fast ordering:  -0.12790203094482422


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
result_file = "/home/linux-miriam/Uniqueness_QML/results.pickle"


#qm7_compounds = datprep.read_compounds(dat_ha_file)
compound_ls = datprep.read_compounds(trial_file)
 
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

    #now save results somewhere
    der_result = datprep.derivative_results(c.filename, Z, CM)
    der_result.add_all_RZev(dZ, dR, ddZ, ddR, dZdR)
    percentile_results = der_result.calculate_percentage()

    eigenvalues.append(der_result)


datprep.store_compounds(eigenvalues, result_file)
            

    

