'''
Test out derivatives easily with this little function

'''
import numerical_derivative as numder
import qml
import representation_ZRN as ZRNrep
import jax_representation as jrep
import jax_derivative as jder
import jax_additional_derivative as jader
import jax_basis as basis
import numpy as np


#define path to xyz file in question
path = "./TEST/H2O.xyz"
compound = qml.Compound(path)

'''get information from xyz file'''
Z = compound.nuclear_charges.astype(float)
R = compound.coordinates
N = float(len(Z))

'''using my own basis'''
myOM, order  = jrep.OM_full_sorted(Z, R, N)

derOM = jder.sort_derivative('OM', Z, R, N, grad = 1, dx = "R")

'''numerical derivative'''
fun = ZRNrep.Overlap_Matrix
numericalder = numder.derivative(fun, [Z, R, N], d1 = [1,0, 0])
print("numerical derivative")
print(numericalder)

'''print results nicely'''
np.set_printoptions(precision=3, suppress=True)
print('my OM is:')
print(myOM)

