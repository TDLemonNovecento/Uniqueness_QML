import qml
import jax_representation as jrep
import jax_derivative as jder
import jax_basis
from jax_math import BoB_fill
import jax.numpy as jnp

path = "/home/linux-miriam/Uniqueness_QML/TEST/H2O.xyz"
compound = qml.Compound(path)


'''get information from xyz file'''
Z = compound.nuclear_charges.astype(float)
R = compound.coordinates
N = float(len(Z))

'''using my own basis'''
M, order  = jrep.CM_full_sorted(Z, R, N)

print("row-sorting: ", order)

'''using basis and S from literature'''
#thisbasis, k = basis.build_sto3Gbasis(Z,R)
#M = np.zeros((k,k))
#myOM = oei.buildS(thisbasis, M)

'''print results nicely'''
jnp.set_printoptions(precision=3, suppress=True)
print('Representation:\n------------------')
print(M)


#jder.cal_print_1stder('CM', Z, R, N)
jder.cal_print_2ndder('CM', Z, R, N)
