import database_preparation as datprep
import jax_derivative as jder
import jax_representation as jrep
import jax.numpy as jnp


#define path to folder containing xyz files. All files are considered.
database = "/home/linux-miriam/Databases/QM9_XYZ/"
database_file = "/home/linux-miriam/Uniqueness_QML/Pickled/qm9.pickle"
dat_ha_file = "/home/stuke/Uniqueness_QML/Pickled/qm7.pickle"
trial_file = "/home/stuke/Uniqueness_QML/Pickled/XYZ_random_ha5.txt"
result_file = "/home/stuke/Uniqueness_QML/Pickled/results_4.pickle"


all_compounds = datprep.read_compounds(dat_ha_file)
print("all compounds: ", len(all_compounds))

compounds = []

for i in all_compounds:
        if i.filename == "dsgdb9nsd_000004.xyz":
                compounds.append(i)
        elif i.filename == "dsgdb9nsd_000024.xyz":
                compounds.append(i)
        elif i.filename == "dsgdb9nsd_000486.xyz":
                compounds.append(i)
        elif i.filename == "dsgdb9nsd_000023.xyz":
                compounds.append(i)
namelist =  [i.filename for i in compounds]
print("compounds: ", len(compounds), namelist)


results, old_per, corr_per, new_per = jder.calculate_eigenvalues('CM', compounds)

datprep.store_compounds(results, result_file)

old_dZ =  []
corr_dZ = []
new_dZ = []

old_dR = []
corr_dR = []
new_dR = []

old_dRdR = []
corr_dRdR =[]
new_dRdR = []

norms = []

for i in range(len(results)):
    norms.append(results[i].norm)

    old_dZ.append(old_per[i][0])
    corr_dZ.append(corr_per[i][0])
    new_dZ.append(new_per[i][0])

    old_dR.append(old_per[i][1])
    corr_dR.append(corr_per[i][1])
    new_dR.append(new_per[i][1])

    old_dRdR.append(old_per[i][3])
    corr_dRdR.append(corr_per[i][3])
    new_dRdR.append(new_per[i][3])

ylist_toplot = [[jnp.asarray(old_dZ), "old dZ"],[jnp.asarray(corr_dZ), "corrected dZ"],[jnp.asarray(new_dZ), "new dZ"] ,[jnp.asarray(old_dR), "old dR"], [jnp.asarray(corr_dR), "corrected dR"], [jnp.asarray(new_dR), "new dR"], [jnp.asarray(old_dRdR), "old dRdR"], [jnp.asarray(corr_dRdR), "corrected dRdR"], [jnp.asarray(new_dRdR), "new dRdR"]]

pltder.plot_percentage_zeroEV(jnp.asarray(norms), ylist_toplot)

