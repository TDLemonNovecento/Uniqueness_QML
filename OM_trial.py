import qml
import jax_representation as jrep
import oei
import basis
import numpy as np

path = "/home/stuke/Databases/XYZ_random/H2O.xyz"
compound = qml.Compound(path)

'''get information from xyz file'''
Z = compound.nuclear_charges.astype(float)
R = compound.coordinates
N = float(len(Z))

'''using my own basis'''
myOM, order  = jrep.OM_full_sorted(Z, R, N)


'''using basis and S from literature'''
#thisbasis, k = basis.build_sto3Gbasis(Z,R)
#M = np.zeros((k,k))
#myOM = oei.buildS(thisbasis, M)

'''print results nicely'''
np.set_printoptions(precision=3, suppress=True)
print('my OM is:')
print(myOM)

