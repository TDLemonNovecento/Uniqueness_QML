import representation_ZRN as ZRNrep
import jax_representation as jrep
import database_preparation as datprep
from time import time as tic
import statistics
import numpy as np

data_file = "../Databases/Pickled/qm7.pickle"

repros = [ZRNrep.Coulomb_Matrix, ZRNrep.Eigenvalue_Coulomb_Matrix, ZRNrep.Overlap_Matrix, \
        ZRNrep.Eigenvalue_Overlap_Matrix, ZRNrep.Bag_of_Bonds]

repronames = ["CM", "EVCM", "OM", "EVOM", "BOB"]
###read list of compounds from data file

compounds = datprep.read_compounds(data_file)
compounds = compounds[:1]
print("number of compounds:", len(compounds))

#store times for every single and total calculation
one_times = [[],[],[],[],[]]
total_times = []


for i in range(5):
    start = tic()
    for c in compounds:
        thisstart = tic()
        M = repros[i](c.Z, c.R)
        thisend = tic()
        one_times[i].append(thisend-thisstart)
        
    end = tic()
    total_times.append(end-start)


#print out results in a latex table form
#np.set_printoptions(precision=3, suppress=True)


print("Representation \t &  Total & Median & Min & Max")
for i in range(5):
    print(repronames[i], "&", \
            "{0:.3e}".format(total_times[i]), "&",\
            "{0:.3e}".format(statistics.median(one_times[i])), "&",\
            "{0:.3e}".format(min(one_times[i])), "&",\
            "{0:.3e}".format(max(one_times[i])))

