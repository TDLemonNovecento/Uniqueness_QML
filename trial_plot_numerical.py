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
which_derivative = 2
#reorder your initial data. otherwise numerical differentiation will fail
#needs to be performed with preordered data, otherwise sorting can change
Z = Z_orig[order.tolist()]
R = R_orig[order]

R = np.asarray(R, dtype = np.float64)
Z = np.asarray(Z, dtype = np.float64)
fun = jrep.CM_full_unsorted_matrix
function = 'CM'
#fun = jrep.CM_ev

#derivatives
derivative_1 =  [[0, 0],  [0,1], [1,0,1],[1,0,2], [1,1,0], [1,1,1], [1,1,2], [1,2,0], [1,2,1], [1,2,2]]

derivative_2 = [[0, 0], [0,1], [1, 0, 2]]

if which_derivative == 1:
    
    #start plotting
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    for d1 in derivative_1:
        name = "d%s" % (str(d1))
    
        #calculate exact result
        if d1[0] == 0:
            name = "dZ%i" % order[d1[1]]
            dx = "Z"
            exact = jder.sort_derivative('CM_unsrt', Z, R, N, 1, dx)[d1[1]]

        if d1[0] == 1:
            xyz = ['x', 'y', 'z']
            name = "d%s%i" %(xyz[d1[1]], d1[2])
            dx = 'R'
            exact = jder.sort_derivative('CM_unsrt', Z, R, N, 1, dx)[d1[1]][d1[2]]
    
        '''
        first derivative below
        '''
        print("\n ------------------- \n Derivative %s \n ------------------ \nAnalytical derivative" %name)
        print(exact)

        hlist = np.logspace(-1, -5, 10)
        print("hlist : ", hlist)
        ylist = []
    
        #calculate numerical derivatives for increasingly small h
        for h in hlist:
            for d2 in derivative_2:
                derf = numder.derivative(fun, [Z,R,N], 'numerical', 1, d1, d2, 3, h)
                exf = exact.flatten()
                error = linalg.norm(derf - exf)
                print("h", h, "error", error)
                #print(derf.reshape(3,3))
                ylist.append(error)
    
        #plot errors in correct subplot
        if d1[0] == 0:
            ax1.loglog(hlist, ylist, label = name)
        else:
            ax2.loglog(hlist, ylist, label = name)


elif which_derivative == 2:
    #start plotting
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    for d1 in derivative_1:
        for d2 in derivative_2:
            name = "d%sd%s" % (str(d1), str(d2))

            #calculate exact result
            if d1[0] == 0:
                name1 = "dZ%i" % order[d1[1]]
                dx1 = "Z"
                if d2[0] == 0:
                    name = name1 + "dZ%i" % order[d2[1]]
                    dx2 = "Z"
                    print("name: ", name)
                    new_exact = jder.sort_derivative('CM_unsrt', Z, R, N, 2, dx1, dx2)
                    print(new_exact)
                    exact = jder.sort_derivative('CM_unsrt', Z, R, N, 2, dx1, dx2)[d1[1]][d2[1]]

            
            if d1[0] == 1:
                xyz = ['x', 'y', 'z']
                name1 = "d%s%i" %(xyz[d1[1]], d1[2])
                dx1 = 'R'
                if d2[0] == 1:
                    name = name1 + "d%s%i" %(xyz[d2[1]], d2[2])
                    dx2 = 'R'
                    exact = jder.sort_derivative('CM_unsrt', Z, R, N, 2, dx1, dx2)[d1[1]][d1[2]][d2[1]][d2[2]]

            if (d1[0] == 0 and d2[0] == 1) or (d1[0] == 1 and d2[0] == 0):
                print("mixed derivative incoming")
                xyz = ['x', 'y', 'z']
                #translate dx1 and dx2 to numbers for functions and names
                dx1 = "Z"
                dx2 = "R"
                

                if d1[0] == 0:
                    ZR_order = True
                    name1 = "dZ%i" % order[d1[1]]
                    name2 = "d%s%i" %(xyz[d2[1]], d2[2])

                else:
                    ZR_order = False
                    name1 = "dZ%i" % order[d2[1]]
                    name2 = "d%s%i" %(xyz[d1[1]], d1[2])

                name = name1 + name2
                    
                print("name : ", name, "derive by", dx1, dx2)
                
                if ZR_order:
                    exact = jder.sort_derivative('CM', Z, R, N, 2, dx1, dx2)[d1[1]][d2[1]][d2[2]]
                else:
                    exact = jder.sort_derivative('CM_unsrt', Z, R, N, 2, dx1, dx2)[d1[1]][d1[2]][d2[1]]
                print("exact solution: \n \n",  exact)
            '''
            second derivative below
            '''
            print("\n ------------------- \n Derivative %s \n ------------------ \nAnalytical derivative" %name)
            print(exact)

            hlist = np.logspace(-1, -5, 10)
            print("hlist : ", hlist)
            ylist = []

            #calculate numerical derivatives for increasingly small h
            for h in hlist:
                derf = numder.derivative(fun, [Z,R,N], 'numerical', 2, d1, d2, 3, h)
                exf = exact.flatten()
                error = linalg.norm(derf - exf)
                print("h", h, "error", error)
                print(derf.reshape(3,3))
                ylist.append(error)

            #plot errors in correct subplot
            if d1[0] == 0:
                if d2[0] == 0:
                    ax1.loglog(hlist, ylist, label = name)
                else:
                    ax3.loglog(hlist, ylist, label = name)
            if d1[0] == 1:
                if d2[0] == 1:
                    ax2.loglog(hlist, ylist, label = name)
                else:
                    ax3.loglog(hlist, ylist, label = name)
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax1.title.set_text('Nuclear Charge')
    ax2.title.set_text('Nuclear Coordinate')
    ax3.title.set_text("Mixed")
    ax1.set_xlabel("dh")
    ax2.set_xlabel("dh")
    ax3.set_xlabel("dh")
    ax1.set_ylabel("error")




fig.suptitle('Finite Central Difference Derivative on Coulomb Matrix of H2O Molecule')
ax1.legend()
ax2.legend()
ax1.title.set_text('by Nuclear Charge')
ax2.title.set_text('by Nuclear Coordinate')
ax1.set_xlabel("dh")
ax2.set_xlabel("dh")
ax1.set_ylabel("error")
plt.show()
