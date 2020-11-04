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

print("row-sorting: ", order)

'''using basis and S from literature'''
#thisbasis, k = basis.build_sto3Gbasis(Z,R)
#M = np.zeros((k,k))
#myOM = oei.buildS(thisbasis, M)

'''print results nicely'''
jnp.set_printoptions(precision=3, suppress=True)
print('Representation:\n------------------')
print(M)


def cal_print_1stder(repro, Z, R, N):
    dim = len(Z)
    '''calculates all derivatives and prints them nicely'''    
    dZ = jder.sort_derivative(repro, Z, R, N, 1, 'Z')
    dN = jder.sort_derivative(repro, Z, R, N, 1, 'N')
    dR = jder.sort_derivative(repro, Z, R, N, 1, 'R')

    print('first derivatives:\n------------------')
    for i in range(dim): #3 atoms in H2O
        print('dZ%i' % (i+1))
        print(dZ[i])

    xyz_labels = ['x', 'y', 'z']
    for xyz in range(3): #x, y and z
        for i in range(dim): #3 atoms in H2O
            print('d%s%i' % (xyz_labels[xyz], (i+1)))
            print(dR[i][xyz]) #derivatives are unsorted

def cal_print_2ndder(repro, Z, R, N):
