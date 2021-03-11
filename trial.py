import qml
import jax_representation as jrep
import jax_derivative as jder
import oei
import basis
from jax_math import BoB_fill
import jax.numpy as jnp

path = "/home/stuke/Databases/TEST_ALL/H2O.xyz"
compound = qml.Compound(path)


'''get information from xyz file'''
Z = compound.nuclear_charges.astype(float)
R = compound.coordinates
N = float(len(Z))

'''using my own basis'''
M, order  = jrep.CM_full_sorted(Z, R, N)

