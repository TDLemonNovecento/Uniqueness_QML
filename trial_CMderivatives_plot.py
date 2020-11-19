'''In this program routines and subroutines are called depending on what shoud be done. Unused parts should be hashed and parts that conflict with each other marked with exclamation marks and special characters respectively twice that character. Written by Miriam Stuke.
'''
import database_preparation as datprep
import jax_derivative as jder
import jax_representation as jrep
import jax.numpy as jnp
import sys

try:
	init, end = int(sys.argv[1]), int(sys.argv[2])
except IndexError:
	init = int(input("starting point"))
	end = int(input("end point"))

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
database = "/home/miriam/Databases/QM9_XYZ/"
database_file = "/home/miriam/Uniqueness_QML/Pickled/qm9.pickle"
dat_ha_file = "/home/miriam/Uniqueness_QML/Pickled/qm7.pickle"
result_folder = "/home/miriam/Uniqueness_QML/Pickled/"

'''
#read xyzfiles and store into a molecular list as well as a compound list
mol_ls, compound_ls = datprep.read_xyz_energies(database)

#store compounds to database_file
datprep.store_compounds(compound_ls, database_file)

#take info from database_file and extract all molecules with less than 7 heavy atoms to dat_ha_file
max_atoms = datprep.sortby_heavyatoms(database_file, dat_ha_file, 7)

#max_atoms is maximal number of atoms in file. needed to set size of CM
print("all CM should have size " , max_atoms)
input("Press enter once you have made sure the size of the unsorted CM matrix has been adapted accordingly")
'''
#read list of compounds from data file
full_compound_ls = datprep.read_compounds(dat_ha_file)
print(len(full_compound_ls), " compounds in full data file")
try:
	compound_ls = full_compound_ls[init : end]
except IndexError:
	print("Your indices were out of bound, restart. min: 0, max: ", len(full_compound_ls))
	exit()

print(len(compound_ls), " of which are being processed")

#create new list of results from list of compounds
results = jder.calculate_eigenvalues('CM', compound_ls)


#store list of results in result_file
result_file = result_folder + "results_%i-%i.pickle" %(init, end)
datprep.store_compounds(results, result_file)

#prepare plotting
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

            

    

