import database_preparation as datprep


pickled_compounds = "/home/linux-miriam/Databases/Pickled/qm7.pickle"
filenames = ["dsgdb9nsd_000004.xyz", "dsgdb9nsd_000024.xyz", "dsgdb9nsd_000486.xyz", "dsgdb9nsd_000023.xyz
"]

mols, compounds = datprep.read_compounds(pickled_compounds)

