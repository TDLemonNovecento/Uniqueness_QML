import qml
import jax_representation as jrep
import jax_derivative as jder
import jax_additional_derivative as jader
import jax_basis
from jax_math import BoB_fill
import jax.numpy as jnp

path = "./Examples/H2O.xyz"
compound = qml.Compound(path)


'''get information from xyz file'''
Z = compound.nuclear_charges.astype(float)
R = compound.coordinates
N = float(len(Z))

'''using my own basis'''
M, order  = jrep.CM_full_sorted(Z, R, N)

print("row-sorting: ", order)


'''print results nicely'''
jnp.set_printoptions(precision=3, suppress=True)
print('Representation:\n------------------')
print(M)

jader.cal_print_1stder('CM_EV', Z, R, N)
jader.cal_print_2ndder('CM_EV', Z, R, N)
