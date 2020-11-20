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
result_file = "/home/linux-miriam/Uniqueness_QML/Pickled/results"

numbers = ["0-166", "166-332", "332-498", "664-830", "830-996", "996-1162", "1162-1328", "1328-1494", "1494-1660", "1660-1826", "1494-1660", "1660-1826", "1826-1992", "1992-2158", "2158-2324", "2324-2490", "2490-2656", "2656-2822", "2822-2988", "2988-3154", "3154-3320", "3320-3486", "3486-3652", "3652-3818", "3818-3993"] 

#"498-550", "550-600", "600-664",
compounds = []

for n in numbers:
    filename = result_file + "_" + n + ".pickle"
    compoundlist = datprep.read_compounds(filename)
    print("number of compounds in sublist: ", len(compoundlist))
    
    compounds.extend(compoundlist)

print("number of compounds: ", len(compounds))

resultfile = result_file + ".pickle"
datprep.store_compounds(compounds, resultfile)

dZ_percentages = []
dR_percentages = []
dZdZ_percentages = []
dRdR_percentages = []
dZdR_percentages = []

norms = []

for i in range(len(compounds)):
    norms.append(compounds[i].norm)
    dZ_percentages.append(compounds[i].dZ_perc)
    dR_percentages.append(compounds[i].dR_perc)
    dZdZ_percentages.append(compounds[i].dZdZ_perc)
    dRdR_percentages.append(compounds[i].dRdR_perc)
    dZdR_percentages.append(compounds[i].dZdR_perc)

ylist_toplot = [[jnp.asarray(dZ_percentages), "dZ"],[jnp.asarray(dR_percentages), "dR"],[jnp.asarray(dZdZ_percentages), "dZdZ"] ,[jnp.asarray(dZdR_percentages), "dZdR"], [jnp.asarray(dZdZ_percentages), "dZdZ"]]

pltder.plot_percentage_zeroEV(jnp.asarray(norms), ylist_toplot, "Nonzero Eigenvalues of CM Derivatives", "QM7_perc_nonzeroEV_CM_one", True)
pltder.plot_percentage_zeroEV(jnp.asarray(norms), ylist_toplot, "Nonzero Eigenvalues of CM Derivatives", "QM7_perc_nonzeroEV_CM_panel", False)

