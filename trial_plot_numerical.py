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

#reorder your initial data. otherwise numerical differentiation will fail
#needs to be performed with preordered data, otherwise sorting can change
Z = Z_orig[order.tolist()]
R = R_orig[order]

R = np.asarray(R, dtype = np.float64)
Z = np.asarray(Z, dtype = np.float64)
fun = jrep.CM_full_unsorted_matrix
function = 'CM'
#fun = jrep.CM_ev

#start plotting
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)


#derivatives
der =  [[0,1] ,  [1,0,0], [1,0,1],[1,0,2], [1,1,0], [1,1,1], [1,1,2], [1,2,0], [1,2,1], [1,2,2]]

d2 = [0,0]
for d in der:
    name = "d%s" % (str(d))
    
    #calculate exact result
    if d[0] == 0:
        name = "dZ%i" % order[d[1]]
        dx = "Z"
        exact = jder.sort_derivative('CM_unsrt', Z, R, N, 1, dx)[d[1]]

    if d[0] == 1:
        xyz = ['x', 'y', 'z']
        name = "d%s%i" %(xyz[d[1]], d[2])
        dx = 'R'
        exact = jder.sort_derivative('CM_unsrt', Z, R, N, 1, dx)[d[1]][d[2]]

    print("\n ------------------- \n Derivative %s \n ------------------ \nAnalytical derivative" %name)
    print(exact)

    hlist = np.logspace(-1, -5, 10)
    print("hlist : ", hlist)
    ylist = []
    
    #calculate numerical derivatives for increasingly small h
    for h in hlist:
        derf = numder.derivative(fun, [Z,R,N], 'numerical', 1, d, d2, 3, h)
        exf = exact.flatten()
        error = linalg.norm(derf - exf)
        print("h", h, "error", error)
        print(derf.reshape(3,3))
        ylist.append(error)

    #plot errors in correct subplot
    if d[0] == 0:
        ax1.loglog(hlist, ylist, label = name)
    else:
        ax2.loglog(hlist, ylist, label = name)

fig.suptitle('Finite Central Difference Derivative on Coulomb Matrix of H2O Molecule')
ax1.legend()
ax2.legend()
ax1.title.set_text('by Nuclear Charge')
ax2.title.set_text('by Nuclear Coordinate')
ax1.set_xlabel("dh")
ax2.set_xlabel("dh")
ax1.set_ylabel("error")
plt.show()
