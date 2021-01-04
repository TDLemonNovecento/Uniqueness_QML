'''
This program prints numerical derivative convergance
'''
import representation_ZRN as ZRNrep
from jax.config import config
config.update("jax_enable_x64", True) #increase precision from float32 to float64
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
which_derivative = 1 
#reorder your initial data. otherwise numerical differentiation will fail
#needs to be performed with preordered data, otherwise sorting can change
Z = Z_orig[order.tolist()]
R = R_orig[order]

R = np.asarray(R, dtype = np.float64)
Z = np.asarray(Z, dtype = np.float64)
functionnames = ['CM_unsrt', 'CM_EV', 'OM', 'OM_EV']
functions = [ZRNrep.Coulomb_Matrix, ZRNrep.Eigenvalue_Coulomb_Matrix, ZRNrep.Overlap_Matrix, ZRNrep.Eigenvalue_Overlap_Matrix]

def get_first_order_errors(fun, fname, hlist, d1):

    if d1[0] == 0:
        name = "dZ%i" % (d1[1] + 1)
        dx = "Z"
        exact = jder.sort_derivative(fname, Z, R, N, 1, dx)[d1[1]]
    if d1[0] == 1:
        xyz = ['x', 'y', 'z']
        name = "d%s%i" %(xyz[d1[1]], d1[2]+1)
        dx = 'R'
        exact = jder.sort_derivative(fname, Z, R, N, 1, dx)[d1[1]][d1[2]]
    print("\n ------------------- \n Derivative %s \n ------------------ \nAnalytical derivative" %name)
    print(exact)

    ylist = []
    #calculate numerical derivatives for increasingly small h
    for h in hlist:
        derf = numder.derivative(fun, [Z,R,N], 'numerical', 1, d1, [0,0], h)
        exf = exact.flatten()
        print("derf")
        print(derf)
        print("exf")
        print(exf)
        error = linalg.norm(derf - exf)
        ylist.append(error)

    return(ylist, name)

def get_second_order_errors(fun, fname, hlist, d1, d2):
    '''
    fun: function from representation_ZRN.py
    fname: string, 'CM', 'CM_EV', 'CM_unsrt', 'OM', 'OM_EV'
    '''
    #calculate exact result
    if d1[0] == 0:
        name1 = "dZ%i" % (d1[1] + 1)
        dx1 = "Z"
        if d2[0] == 0:
            name = name1 + "dZ%i" % (d2[1] + 1)
            dx2 = "Z"
            exact = jder.sort_derivative(fname, Z, R, N, 2, dx1, dx2)[d1[1]][d2[1]]

    if d1[0] == 1:
        xyz = ['x', 'y', 'z']
        name1 = "d%s%i" %(xyz[d1[1]], d1[2])
        dx1 = 'R'
        if d2[0] == 1:
            name = name1 + "d%s%i" %(xyz[d2[1]], (d2[2] + 1))
            dx2 = 'R'
            exact = jder.sort_derivative(fname, Z, R, N, 2, dx1, dx2)[d1[1]][d1[2]][d2[1]][d2[2]]

    if (d1[0] == 0 and d2[0] == 1) or (d1[0] == 1 and d2[0] == 0): 
        xyz = ['x', 'y', 'z']
        #translate dx1 and dx2 to numbers for functions and names
        dx1 = "Z"
        dx2 = "R"

        if d1[0] == 0:
            ZR_order = True
            name1 = "dZ%i" % (d1[1]+1)
            name2 = "d%s%i" %(xyz[d2[1]], d2[2]+1)

        else:
            ZR_order = False
            name1 = "dZ%i" % (d2[1]+1)
            name2 = "d%s%i" %(xyz[d1[1]], d1[2]+1)
        name = name1 + name2

        if ZR_order:
            exact = jder.sort_derivative(fname, Z, R, N, 2, dx1, dx2)[d1[1]][d2[1]][d2[2]]
        else:
            exact = jder.sort_derivative(fname, Z, R, N, 2, dx1, dx2)[d2[1]][d1[1]][d1[2]]


    print("\n ------------------- \n Derivative %s \n ------------------ \nAnalytical derivative" %name)
    print(exact)

    ylist = []
    #calculate numerical derivatives for increasingly small h
    for h in hlist:
        derf = numder.derivative(fun, [Z,R,N], 'numerical', 2, d1, d2, h)
        exf = exact.flatten()
        error = linalg.norm(derf - exf)
        #print("h", h, "error", error)
        #print(derf.reshape(3,3))
        ylist.append(error)

    return(ylist, name)

def plot_numeric_errors(fun, fname, do_first_order = True, do_second_order = True, show_legend = False):
    '''
    fun : function from representation_ZRN.py file
    fname : function name: 'CM', 'CM_unsrt', 'CM_EV', 'OM' or 'OM_EV'

    '''
    #prepare panels for plots
    fig = plt.figure(tight_layout = True, figsize=(8, 6))

    #create subfigure spacing as needed
    if do_first_order:
        if do_second_order:
            '''create 5 panel image'''
            ax1 = plt.subplot2grid((2,6),(0,0), rowspan=1, colspan=3)
            ax2 = plt.subplot2grid((2,6), (0,3), rowspan=1, colspan=3) 
            ax3 = plt.subplot2grid((2,6), (1,0), rowspan=1, colspan=2)
            ax4 = plt.subplot2grid((2,6), (1,2), rowspan=1, colspan=2)
            ax5 = plt.subplot2grid((2,6), (1,4), rowspan=1, colspan=2)

        else:
            '''create 2 panel image for dZ and dR'''
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
    elif do_second_order:
        '''create 3 panel image for dZdZ, dRdR and dZdR'''
        ax3 = fig.add_subplot(131)
        ax4 = fig.add_subplot(132)
        ax5 = fig.add_subplot(133)

    #derivatives: standard parameters by which to derive.
    '''
    [0, ...] is dZ
    [1, ...] is dR
    [...,0,...] is dZ1 or dR1
    for dR, [...,...,0] is dx, [...,...,1] dy and [...,...,2] dz
    '''

    derivative_1 =  [[0,0], [1,1, 0]]#, [0,2],[1,0,0], [1,0,1],[1,0,2],[1,1,0], [1,1,1],[1,1,2],[1,2,0], [1,2,1], [1,2,2]]

    derivative_2 = [[0,0], [0,1], [0,2], [1, 1, 2]]# [0, 1], [1,0,0], [1,2,2], [1,1,2]]
    
    #hlist: small steps that are taken for numerical derivation
    hlist = np.logspace(-1, -5, 10)
    


    
    #plot 1st order if needed
    if do_first_order:
        for d1 in derivative_1:
            ylist, name = get_first_order_errors(fun, fname, hlist, d1)
            
            #plot errors in correct subplot
            if d1[0] == 0:
                ax1.loglog(hlist, ylist, 'o-', label = name)
            else:
                ax2.loglog(hlist, ylist, 'o-', label = name)
        
        if show_legend:
            ax1.legend()
            ax2.legend()

        ax1.title.set_text('dZ (nuclear charge)')
        ax2.title.set_text('dR (nuclear coordinate)')
        ax1.set_xlabel("dh")
        ax2.set_xlabel("dh")


    if do_second_order:

        for d1 in derivative_1:
            for d2 in derivative_2:
                
                ylist, name = get_second_order_errors(fun, fname, hlist, d1, d2)

                #plot errors in correct subplot
                if d1[0] == 0:
                    if d2[0] == 0:
                        ax3.loglog(hlist, ylist, 'o-', label = name)
                else:
                    ax5.loglog(hlist, ylist, 'o-', label = name)
                if d1[0] == 1:
                    if d2[0] == 1:
                        ax4.loglog(hlist, ylist, 'o-', label = name)
                else:
                    ax5.loglog(hlist, ylist,'o-', label = name)
        
        if show_legend:
            ax3.legend()
            ax4.legend()
            ax5.legend()
        ax3.title.set_text("dZdZ")
        ax4.title.set_text("dRdR")
        ax5.title.set_text("dZdR")
        ax3.set_xlabel("dh")
        ax4.set_xlabel("dh")
        ax5.set_xlabel("dh")
    
    
    fig.suptitle('Finite Central Difference Derivative on Coulomb Matrix of H2O Molecule', fontsize = 15)
    #ad title to overall y axis

    fig.text(0.5, 0.0004, '(dh is the small change introduced in numerical differentiation)', ha='center', fontsize = 13)
    fig.text(0.0004, 0.5, 'Absolute Error w.r.t. Analytical Derivative [a.u.]', rotation = 'vertical', va='center', fontsize =13)    
    

    plt.savefig('./Images/numerical_errors_CM.png', bbox_inches = 'tight')

    return()

for fun, fname in zip(functions, functionnames):
    print("function:", fun)
    print("function name:", fname)
    plot_numeric_errors(fun, fname)
