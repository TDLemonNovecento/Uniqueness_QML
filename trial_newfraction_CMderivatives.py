import database_preparation as datprep
import jax_derivative as jder
import jax_representation as jrep
import jax.numpy as jnp

#define path to folder containing xyz files. All files are considered.
database = "/home/linux-miriam/Databases/QM9_XYZ/"
database_file = "/home/linux-miriam/Uniqueness_QML/Pickled/qm9.pickle"
dat_ha_file = "/home/linux-miriam/Uniqueness_QML/Pickled/qm7.pickle"
trial_file = "/home/linux-miriam/Uniqueness_QML/Pickled/XYZ_random_ha5.txt"
result_file = "/home/linux-miriam/Uniqueness_QML/Pickled/results"


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


results = jder.calculate_eigenvalues('CM', compounds)

datprep.store_compounds(results, result_file)

ylist_toplot = [[jnp.asarray(dZ_percentages), "dZ"],[jnp.asarray(dR_percentages), "dR"],[jnp.asarray(dZdZ_percentages), "dZdZ"] ,[jnp.asarray(dZdR_percentages), "dZdR"], [jnp.asarray(dZdZ_percentages), "dZdZ"]]

print(ylist_toplot)
exit()

