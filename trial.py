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
firsthundred = "/home/linux-miriam/Databases/Pickled/BoB_numder_res0-100"

results = datprep.read_compounds(compounds)[:10]
#res2 = datprep.read_compounds(seveneight)
res3 = datprep.read_compounds(firsthundred)[:10]


dZarray = []
for c in results:
    #if c.filename == "dsgdb9nsd_000177.xyz":
    print("filename:", c.filename)
    print("Z =",  np.asarray(c.Z))
        #print("BOB")
        #Bob = ZRNrep.Bag_of_Bonds(c.Z, c.R)
        #print(Bob)
    dZ = []
    for i in range(len(c.Z)):
        dZ.append(numder.derivative(ZRNrep.Bag_of_Bonds, [c.Z, c.R, 0], d1 = [0, i]))
    
    nonzero = np.count_nonzero(dZ)
    dimBOB = jrep.BoB_dimension(c.Z)

    print("dim BOB:", dimBOB)
    print("nonzero:", nonzero)
    print("nonzero dZ fraction:", nonzero/(dimBOB*len(c.Z)))
        #rounded = np.round(dZ, 5)
        #print(rounded[rounded != 0.])


print("\nResults as calculated before:\n")

for c in res3:
    #if c.filename == "dsgdb9nsd_000177.xyz":
    
    print("filename:", c.filename)
    print("Z:", c.Z)
    
    print("nonzero:", np.count_nonzero(c.dZ_ev))

    #calculate nonzero:
    frac, no = c.calculate_smallerthan(repro = 2)
    
    print("dZ fractions:", frac[0])
    
    frac2 = c.calculate_percentage(repro = 2)
    print("dZ fraction by absolute:", frac2[0])
    #print("zero valued entries total:", np.count_nonzero(np.asarray(c.dZ_ev)))
    
    #print("dimensions of Bob:", c.calculate_dim(2))


