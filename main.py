'''In this program routines and subroutines are called depending on what shoud be done. Unused parts should be hashed and parts that conflict with each other marked with exclamation marks and special characters respectively twice that character. Written by Miriam Stuke.
'''
import database_preparation as datprep
import jax_derivative as jder


#define path to folder containing xyz files. All files are considered.
database = "/home/linux-miriam/Databases/QM9_XYZ/"
database_file = "/home/linux-miriam/Databases/Pickled/qm9.pickle"
dat_ha_file = "/home/linux-miriam/Databases/Pickled/qm7.pickle"

mol_ls, compound_ls = datprep.read_xyz_energies(database)

datprep.store_compounds(compound_ls, database_file)

datprep.sortby_heavyatoms(database_file, dat_ha_file, 7)

ha_compounds = datprep.read_compounds(dat_ha_file)

for c in compound_ls:
    print(c.heavy_atoms())

print("now all heavy atoms in sorted list")
for i in ha_compounds:
    print(i.heavy_atoms())
