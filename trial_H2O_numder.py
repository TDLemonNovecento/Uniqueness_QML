'''
This program prints all numerical derivatives of a function
'''
import numpy as np
import jax.numpy as jnp
import jax.ops
import jax_representation as jrep
import representation_ZRN as ZRNrep
import database_preparation as datprep
import numerical_derivative as numder

#give xyz coordinates of H2O
path = "./TEST/H2O.xyz"

#read xyz file and create compound instances

Z_orig, R_orig, N, e_orig = datprep.read_xyzfile(path)
M, order = jrep.CM_full_sorted(Z_orig, R_orig, N)

#reorder your initial data. otherwise numerical differentiation will fail
#needs to be performed with preordered data, otherwise sorting can change

Z = Z_orig[order]
R = R_orig[order]
print("Z: ", Z, "R: ", R)
dim = len(Z)
xyz = [[0,'x'],[1,'y'],[2,'z']]
h1 = 0.01
h2 = 0.01
Z = jnp.asarray(Z, dtype = jnp.float32)
fun = ZRNrep.Coulomb_Matrix_Sorted
#fun = jrep.CM_ev


#store all results in arrays to print them later
Zder = []
Rder = []
Z2der = []
R2der = []
ZRder = []


#do all Z derivatives
for i in range(dim):
    name = ("dZ%i:" % (i+1))
    
    der = numder.derivative(fun, [Z, R, N], d1 = [0, i])
    Zder.append([name,der])

    #do all dZdZ derivatives:
    for j in range(dim):
        name = ("dZ%i dZ%i:" %( i+1, j+1))
        
        der = numder.derivative(fun, [Z, R, N], d1 = [0,i], d2 = [0,j]) 
        
        Z2der.append([name, der])
        
        #do all dZdR derivatives:
        for x in xyz:
            name = ("dZ%i d%s%i:" %(i+1, x[1], j+1))

            der = numder.derivative(fun, [Z, R, N], d1 = [0, i], d2 = [1,j, x[0]])
            ZRder.append([name,der])
    
    #do all dR derivatives:
    for y in xyz:
        name = ("d%s%i :" %(y[1], i+1))

        der = numder.derivative(fun, [Z, R, N], d1 = [1, i, y[0]])
        Rder.append([name,der])

        #do all dRdR derivatives:
        for k in range(i, dim):
            for z in xyz:
                name = ("d%s%i d%s%i :" %(y[1], i+1, z[1], k+1))

                der = numder.derivative(fun, [Z, R, N], d1 = [1, i, y[0]],d2 = [1, k, z[0]])
                R2der.append([name,der])
    
#now print results properly:
print("all dR derivatives")
for i in Rder:
    print(i[0], "\n", np.reshape(i[1], (3,3)))

print("all dZ derivatives")
for i in Zder:
    print(i[0], "\n", np.reshape(i[1],(3,3)))

print("all dZdZ derivatives")
for i in Z2der:
    print(i[0], "\n", np.reshape(i[1],(3,3)))

print("all dRdR derivatives")    
for i in R2der:
    print(i[0], "\n", np.reshape(i[1], (3,3)))

print("all dZdR derivatives")    
for i in ZRder:
    print(i[0], "\n", np.reshape(i[1], (3,3)))

