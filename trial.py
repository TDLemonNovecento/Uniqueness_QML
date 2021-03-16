import database_preparation as datprep
import representation_ZRN as ZRNrep
import numpy as np
import jax_representation as jrep
import numerical_derivative as numder


filepath = "/home/linux-miriam/Databases/Pickled/OM_numder_res"
finalfile = "/home/linux-miriam/Databases/Pickled/qm7_OM_results.pickle" 
compounds = "./Pickled/qm7.pickle"

filecompounds = "./Pickled/fourcompounds.pickle"
seveneight = "/home/linux-miriam/Databases/Pickled/BoB_numder_res700-800"


results = datprep.read_compounds(compounds)
res2 = datprep.read_compounds(seveneight)


dZarray = []
for c in results:
    if c.filename == "dsgdb9nsd_000177.xyz":
        print("Z =",  np.asarray(c.Z))
        #print("BOB")
        #Bob = ZRNrep.Bag_of_Bonds(c.Z, c.R)
        #print(Bob)
        for i in range(len(c.Z)):
            dZ = numder.derivative(ZRNrep.Bag_of_Bonds, [c.Z, c.R, 0], d1 = [0, 0])
            dZarray.append(dZ)
            print("zeros in dZ:", np.count_nonzero(dZ))
            print(dZ)
            rounded = np.round(dZ, 5)
            print(rounded[rounded != 0.])

for c in res2:
    if c.filename == "dsgdb9nsd_000177.xyz":
        print("dZ percentage:", c.dZ_perc)
        print("zero valued entries total:", np.count_nonzero(np.asarray(c.dZ_ev)))
        print("zeri valued entries by dZ:")
        for i in c.dZ_ev:
            #print("dZ ", i, ":   ", np.count_nonzero(i))
            dZ = np.round(i, 5)
            print(dZ[dZ != 0.])
        print("dimensions of Bob:", c.calculate_dim(2))


