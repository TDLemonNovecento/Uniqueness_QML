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
trial_file = "/home/linux-miriam/Uniqueness_QML/Pickled/XYZ_random_ha5.txt"
result_file = "/home/linux-miriam/Uniqueness_QML/results.pickle"
new_result_file = "/home/linux-miriam/Uniqueness_QML/new_results.pickle"


#qm7_compounds = datprep.read_compounds(dat_ha_file)
results = datprep.read_compounds(result_file)

dZ_percentages = []
dR_percentages = []
dZdZ_percentages = []
dRdR_percentages = []
dZdR_percentages = []

norms = []

for i in range(len(results)):
    norms.append(results[i].norm)
    results_perc = results[i].calculate_percentage()
    dZ_percentages.append(results[i].dZ_perc)
    dR_percentages.append(results[i].dR_perc)
    dZdZ_percentages.append(results[i].dZdZ_perc)
    dRdR_percentages.append(results[i].dRdR_perc)
    dZdR_percentages.append(results[i].dZdR_perc)

ylist_toplot = [[jnp.asarray(dZ_percentages), "dZ"],[jnp.asarray(dR_percentages), "dR"],[jnp.asarray(dZdZ_percentages), "dZdZ"] ,[jnp.asarray(dZdR_percentages), "dZdR"], [jnp.asarray(dZdZ_percentages), "dZdZ"]]

pltder.plot_percentage_zeroEV(jnp.asarray(norms), ylist_toplot, "Nonzero Eigenvalues of CM Derivatives", "perc_nonzeroEV_CM_one", True)
pltder.plot_percentage_zeroEV(jnp.asarray(norms), ylist_toplot, "Nonzero Eigenvalues of CM Derivatives", "perc_nonzeroEV_CM_panel", False)

