import database_preparation as datprep
'''
with this file XYZ files can be converted to database_preparation.compounds class objects
'''

#define path to folder containing xyz files
database = "../Databases/QM9_XYZ/"

#define path to where you want to store your data
database_file = "../Databases/Pickled/qm9.pickle"

#define path to where you want to store data of molecules with
#less heavy atoms than in the database_file
dat_ha_file = "../Databases/Pickled/qm7.pickle"

#read all compounds in database file and convert to datprep class objects
mol_ls, compound_ls = datprep.read_xyz_energies(database)

#store compounds to database_file
datprep.store_compounds(compound_ls, database_file)

#store all compounds with less than 7 atoms with code below:
datprep.sortby_heavyatoms(database_file, dat_ha_file, 7)

ha_compounds = datprep.read_compounds(dat_ha_file)

for c in compound_ls:
    print(c.heavy_atoms())

print("now all heavy atoms in sorted list")
for i in ha_compounds:
    print(i.heavy_atoms())
