import database_preparation as datprep
import jax_derivative as jder
import jax_representation as jrep
import jax.numpy as jnp
import plot_derivative as pltder

#define path to folder containing xyz files. All files are considered.
database = "/home/linux-miriam/Databases/QM9_XYZ/"
database_file = "/home/linux-miriam/Uniqueness_QML/Pickled/qm9.pickle"
dat_ha_file = "/home/stuke/Uniqueness_QML/Pickled/qm7.pickle"
trial_file = "/home/stuke/Uniqueness_QML/Pickled/XYZ_random_ha5.txt"
result_file = "/home/linux-miriam/Uniqueness_QML/Pickled/results_4.pickle"


all_compounds = datprep.read_compounds(dat_ha_file)
print("all compounds: ", len(all_compounds))


results = datprep.read_compounds(result_file)


old_dZ =  []
corr_dZ = []
new_dZ = []

old_dR = []
corr_dR = []
new_dR = []

old_dZdZ = []
corr_dZdZ =[]
new_dZdZ = []

old_dRdR = []
corr_dRdR =[]
new_dRdR = []

old_dZdR = []
corr_dZdR =[]
new_dZdR = []

norms = []

for res in results:
    old_val, corr_val, new_val = res.calculate_percentage()
    norms.append(res.norm)

    old_dZ.append(old_val[0])
    corr_dZ.append(corr_val[0])
    new_dZ.append(new_val[0])

    old_dR.append(old_val[1])
    corr_dR.append(corr_val[1])
    new_dR.append(new_val[1])

    old_dRdR.append(old_val[3])
    corr_dRdR.append(corr_val[3])
    new_dRdR.append(new_val[3])

    old_dZdZ.append(old_val[2])
    corr_dZdZ.append(corr_val[2])
    new_dZdZ.append(new_val[2])

    old_dZdR.append(old_val[4])
    corr_dZdR.append(corr_val[4])
    new_dZdR.append(new_val[4])


ylist_toplot = [[jnp.asarray(old_dZ), "old dZ"],[jnp.asarray(corr_dZ), "corrected dZ"],[jnp.asarray(new_dZ), "new dZ"] ,[jnp.asarray(old_dR), "old dR"], [jnp.asarray(corr_dR), "corrected dR"], [jnp.asarray(new_dR), "new dR"], [jnp.asarray(old_dRdR), "old dRdR"], [jnp.asarray(corr_dRdR), "corrected dRdR"], [jnp.asarray(new_dRdR), "new dRdR"]]

pltder.plot_percentage_zeroEV(jnp.asarray(norms), ylist_toplot)

