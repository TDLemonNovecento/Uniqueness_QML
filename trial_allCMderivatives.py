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
    dim = len(Z)
    which_return = [False, True, False]
    '''calculates all second derivatives'''
    if which_return[0]:
        dZdZ = jder.sort_derivative(repro, Z, R, N, 2, 'Z', 'Z')
        for i in range(dim): #3 atoms in H2O
            for j in range(dim): #second dZ over 3 atoms in H2O
                print('dZ%idZ%i' %(i+1, j+1))
                print(dZdZ[i,j])
 
    if which_return[1]:
        dZdR = jder.sort_derivative(repro, Z, R, N, 2, 'Z', 'R')

        print("dZdR derivatives:")

        xyz = [[0,'x'],[1,'y'],[2,'z']]
        for i in range(dim): #3 atoms in H2O
            for j in range(dim):
                for x in xyz:
                    print('dZ%id%s%i' %(i+1, x[1], j+1))
                    print(dZdR[i, j, x[0]])

    if which_return[2]:
        dRdR = jder.sort_derivative(repro, Z, R, N, 2, 'R', 'R')
        print("dRdR derivatives:")

        xyz = [[0,'x'],[1,'y'],[2,'z']]
        for i in range(dim): #3 atoms in H2O
            for x in xyz:
                for j in range(dim):
                    for y in xyz:
                        print('d%s%id%s%i' %(x[1], i+1, y[1], j+1))
                        print(dRdR[i,x[0], j, y[0]])
    
    
    

cal_print_2ndder('CM', Z, R, N)
