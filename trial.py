import database_preparation as datprep
import jax_derivative as jder
import jax_additional_derivative as jader
import jax_representation as jrep
import plot_derivative as pltder
import jax.numpy as jnp
import sys
import representation_ZRN as ZRNrep

path_CM_results = "/home/linux-miriam/Databases/Pickled/qm7_CM_results.pickle"
four_compounds = "./Pickled/fourcompounds.pickle"
water = "./TEST/H2O.xyz"
results_file = "./Pickled/trial_numder.pickle"

data_file = four_compounds

#for numerical calculations:
numerical_representations = [ZRNrep.Coulomb_Matrix, ZRNrep.Eigenvalue_Coulomb_Matrix, ZRNrep.Overlap_Matrix, \
        ZRNrep.Eigenvalue_Overlap_Matrix, ZRNrep.Bag_of_Bonds]



try:
        init, end = int(sys.argv[1]), int(sys.argv[2])
except IndexError:
        init = int(input("starting point"))
        end = int(input("end point"))


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


print("checkpoint1: database was successfully read, trial.py line 44")

results, resultaddition = jader.calculate_num_der(numerical_representations[2], compound_ls)
datprep.store_compounds(results, results_file)

