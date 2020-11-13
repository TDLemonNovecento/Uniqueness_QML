'''In this program routines and subroutines are called depending on what shoud be done. Unused parts should be hashed and parts that conflict with each other marked with exclamation marks and special characters respectively twice that character. Written by Miriam Stuke.
'''
import time
import database_preparation as datprep
import jax_derivative as jder
import jax_representation as jrep
import jax.numpy as jnp
import plot_derivative as pltder

#define path to folder containing xyz files. All files are considered.
database = "/home/linux-miriam/Databases/QM9_XYZ/"
database_file = "/home/linux-miriam/Uniqueness_QML/Pickled/qm9.pickle"
dat_ha_file = "/home/linux-miriam/Uniqueness_QML/Pickled/qm7.pickle"
trial_file = "/home/stuke/Uniqueness_QML/Pickled/XYZ_random_ha5.txt"
result_file = "/home/stuke/Uniqueness_QML/results.pickle"
new_result_file = "/home/stuke/Uniqueness_QML/new_results.pickle"


#qm7_compounds = datprep.read_compounds(dat_ha_file)
results = datprep.read_compounds(result_file)
data = datprep.read_compounds(trial_file)

dZ_percentages = []
dR_percentages = []
dZdZ_percentages = []
dRdR_percentages = []
dZdR_percentages = []

norms = []

for i in range(len(data)):
    Z = data[i].Z
    R = data[i].R
    M, order = jrep.CM_full_sorted(Z,R)
    results[i].add_Z_norm(Z, M)
    norm = results[i].norm
    dZ_perc, dR_perc, dZdZ_perc, dRdR_perc, dZdR_perc = results[i].calculate_percentage()
    
    norms.append(norm)
    dZ_percentages.append(dZ_perc)
    dR_percentages.append(dR_perc)
    dZdZ_percentages.append(dZdZ_perc)
    dRdR_percentages.append(dRdR_perc)
    dZdR_percentages.append(dZdR_perc)

datprep.store_compounds(results, new_result_file)

pltder.plot_percentage_zeroEV(np.asarray(norms), np.asarray(dZ_percentages), title = "dZ Derivatives")
pltder.plot_percentage_zeroEV(np.asarray(norms), np.asarray(dR_percentages), title = "dR Derivatives")
pltder.plot_percentage_zeroEV(np.asarray(norms), np.asarray(dZdZ_percentages), title = "dZdZ Derivatives")
pltder.plot_percentage_zeroEV(np.asarray(norms), np.asarray(dZdR_percentages), title = "dZdR Derivatives")
pltder.plot_percentage_zeroEV(np.asarray(norms), np.asarray(dRdR_percentages), title = "dRdR Derivatives")

