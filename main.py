'''In this program routines and subroutines are called depending on what shoud be done. Unused parts should be hashed and parts that conflict with each other marked with exclamation marks and special characters respectively twice that character. Written by Miriam Stuke.
'''
import time
import database_preparation as datprep
import jax_derivative as jder
import jax_additional_derivative as jader
import jax_representation as jrep
import jax.numpy as jnp
import plot_derivative as pltder

#define path to folder containing xyz files. All files are considered.
datapath = "../Database/QM9/"

compounds = 

#where do you want these compounds to be saved to?
small_data_file = "../Database/Pickled/compounds.pickle"
CM_ev_result_file = "/home/linux-miriam/Uniqueness_QML/Pickled/fourcompounds_res.pickle"



compounds = datprep.read_compounds(small_data_file)

single = False
if single:
    for c in compounds:
        ev, vectors = jrep.CM_ev(c.Z, c.R, c.N)
        print("name of compound:", c.filename)
        #print("eigenvalue repro:\n", ev)
    
        derivative = jader.sort_derivative('CM_EV', c.Z, c.R, c.N, 2, "R", "R")
        print(derivative)

results, fractions = jader.calculate_eigenvalues('CM_EV', compounds)
datprep.store_compounds(results, CM_ev_result_file)

print(type(results[0]))

#y-axis information
dZ_percentages = []
dR_percentages = []
dZdZ_percentages = []
dRdR_percentages = []
dZdR_percentages = []

#x-axis information
norms = []

#C)
#get all the data from our results list
for i in range(len(results)):
    norms.append(results[i].norm)
    results_perc = results[i].calculate_percentage()
    dZ_percentages.append(results[i].dZ_perc)
    dR_percentages.append(results[i].dR_perc)
    dZdZ_percentages.append(results[i].dZdZ_perc)
    dRdR_percentages.append(results[i].dRdR_perc)
    dZdR_percentages.append(results[i].dZdR_perc)

    #some results were off, check for filenames
    if results[i].dZdR_perc > 0.8 or results[i].dZ_perc > 0.8:
        print(results[i].filename)

CM_ev_ylist_toplot = [[jnp.asarray(dZ_percentages), "dZ CM_ev"],[jnp.asarray(dR_percentages), "dR CM_ev"],[jnp.asarray(dRdR_percentages), "dRdR CM_ev"] ,[jnp.asarray(dZdR_percentages), "dZdR CM_ev"], [jnp.asarray(dZdZ_percentages), "dZdZ CM_ev"]]

#C)
#plot and save all datapoints in one and in multiple panels
pltder.plot_percentage_zeroEV(jnp.asarray(norms), CM_ev_ylist_toplot, "Nonzero Eigenvalues of CM Derivatives", "perc_nonzeroEV_CM_ev_one", True)
pltder.plot_percentage_zeroEV(jnp.asarray(norms), CM_ev_ylist_toplot, "Nonzero Eigenvalues of CM Derivatives", "perc_nonzeroEV_CM_ev_panel", False)

