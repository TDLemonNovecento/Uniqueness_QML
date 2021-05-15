'''try calculating derivatives of rep'''
from time import perf_counter as tic

import database_preparation as datprep
import jax_derivative as jder
import jax_additional_derivative as jader
import jax_representation as jrep
import plot_derivative as pltder
import jax.numpy as jnp
import sys
import representation_ZRN as ZRNrep

path_QM7 = "../Databases/Pickled/qm7.pickle"

data_file = path_QM7
results_file = "./tmp/res_trial"

#for numerical calculations:
numerical_representations = [ZRNrep.Coulomb_Matrix, ZRNrep.Eigenvalue_Coulomb_Matrix,\
         ZRNrep.Bag_of_Bonds, ZRNrep.Overlap_Matrix, \
        ZRNrep.Eigenvalue_Overlap_Matrix]

#which representation should be computed? 0 = CM, 1 = EVCM, 2 = BOB, 3 = OM, 4 = EVOM
which_rep = 0

try:
        init, end = int(sys.argv[1]), int(sys.argv[2])
except IndexError:
        init = int(input("starting point"))
        end = int(input("end point"))

name = str(init) + "-" + str(end)

###read list of compounds from data file
full_compound_ls = datprep.read_compounds(data_file)
print(len(full_compound_ls), " compounds in full data file")

###B)

#If you want to plot only part of all compounds, use this code:
try:
        compound_ls = full_compound_ls[init : end]
except IndexError:
        print("Your indices were out of bound, restart. min: 0, max: ", len(full_compound_ls))
        exit()

#print("you are going to calculate the repro on a list of compounds of length:")
#print(len(compound_ls))

t1 = tic()
results, resultaddition = jader.calculate_num_der(numerical_representations[which_rep], compound_ls)
t2 = tic()

print("time for calculating results:", t2 - t1)


res_file = results_file + name

datprep.store_compounds(results, res_file)

print("results were successfully stored to ", res_file)
