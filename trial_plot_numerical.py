'''
This program prints numerical derivative convergance
'''
import jax_representation as jrep
import numpy as np
from numpy import linalg
import jax_derivative as jder
import database_preparation as datprep
import matplotlib.pyplot as plt
import numerical_derivative as numder

#give xyz coordinates of H2O
path = "./TEST/H2O.xyz"

#read xyz file and create compound instances
Z_orig, R_orig, N, e_orig = datprep.read_xyzfile(path)
Z_orig = np.array(Z_orig, dtype = np.float64)

M, order = jrep.CM_full_sorted(Z_orig, R_orig, N)
print("coulomb Matrix")
print(M)
sys.exit()
#reorder your initial data. otherwise numerical differentiation will fail
#needs to be performed with preordered data, otherwise sorting can change
Z = Z_orig[order.tolist()]
R = R_orig[order]


Z = np.asarray(Z, dtype = np.float32)
fun = jrep.CM_full_unsorted_matrix
function = 'CM'
#fun = jrep.CM_ev

#start plotting
fig, ax = plt.subplots(3,4)


#derivatives
der = [[0,0], [0,1]] #, [0,2], [1,0,0], [1,0,1],[1,0,2], [1,1,0], [1,1,1], [1,1,2], [1,2,0], [1,2,1], [1,2,2]]

d2 = [0,0]
for d in der:
    name = "d%s" % (str(d))
    
    #calculate exact result
    if d[0] == 0:
        name = "dZ%i" % d[1]
        dx = "Z"
        exact = jder.sort_derivative(function, Z, R, N, 1, dx)[d[1]]

    if d[0] == 1:
        xyz = ['x', 'y', 'z']
        name = "d%s%i" %(xyz[d[1]], d[2])
        dx = 'R'
        exact = jder.sort_derivative(function, Z, R, N, 1, dx)[d[1]][d[2]]
    print("derivative", name)
    print("exact analytical result")
    print(exact)


    hlist = np.logspace(-1, -5, 10)
    print("hlist : ", hlist)
    ylist = []
    
    print("dZ derivative")
    for h in hlist:
        derf = numder.derivative(fun, [Z,R,N], 'numerical', 1, d, d2, 3, h)
        exf = exact.flatten()
        error = linalg.norm(derf - exf)
        print("h", h, "error", error)
        print("numerical derivative")
        print(derf.reshape(3,3))
        ylist.append(error)


    ax[d[1]][d[0]].loglog(hlist, ylist, label = name)
plt.title('Norm of Numerical Derivative on CM')
plt.show()
