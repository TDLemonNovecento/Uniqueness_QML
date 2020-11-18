#here we calculate the first and second derivajtives to the basis of chemical space depending on the chosen representation and placeholder. Matrix and vector reconstruction may be included, too.
import numpy as np
import jax.numpy as jnp
import jax_representation as jrep
from jax import grad, jacfwd, jacrev
import time
import database_preparation as datprep

def sort_derivative(representation, Z, R, N = 0, grad = 1, dx = "Z", ddx = "R", M = None, order = None):
    '''Easy function to handle no, one dimensional or two dimensional derivatives with grad. Issue right now: no additional arguments can be passed to function, it therefore falls back to default for any further arguments beside Z, R and N.
    Parameters
    ----------
    fun : callable function depending on (Z, R, N, ...) args
    Z : 1 x n dimensional array
        contains nuclear charges
    R : 3 x n dimensional array
        contains nuclear positions
    N : float
        number of electrons in system
        here: meaningless, can remain empty
    grad: int
        degree by which to differentiate
    dx: string
        argument by which to differentiate first time
    ddx: string
        argument by which to diff. second time
    Return
    ------
    some form of array depending on degree of differentiation
    '''

    #first, find out which representation was chosen and get appropriate function
    fn_list = {'CM': jrep.CM_full_sorted, 'CM_EV': jrep.CM_ev, 'OM' : jrep.OM_full_sorted}
    dfn_list = {'CM': d_CM, 'CM_EV' : d_CM_ev, 'OM' : d_OM, 'OM_EV' : d_OM_ev}
    ddfn_list = {'CM': dd_CM, 'CM_EV' : dd_CM_ev, 'OM' : dd_OM, 'OM_EV' : dd_OM_ev}
    

    try:
        fn = fn_list[representation]
    except ValueError:
        fn = fn_list['CM']
        print("your representation was not found. falling back to 'CM'")
    
    


    if grad == 0:
        if M == None:
            return( fn(Z, R, N)[0])
        else:
            return(M)
    
    #get correct derivative in jax_derivative function depending on which data should be derived by
    diff_list = {'Z' : 0, 'R' : 1, 'N' : 2}
    try:
        dx_index = diff_list[dx]
    except ValueError:
        dx_index = diff_list['Z']
        print("your dx value cannot be derived by. falling back to 'Z'")

    if grad == 1:#first derivative is calculated

        try:
            d_fn = dfn_list[representation]
        except ValueError:
            d_fn = dfn_list['CM']
            print("your representation was not found. falling back to 'CM' for first derivative")
        
        return(d_fn(Z, R, N, dx_index, M, order))
    

    #grad is 2 or bigger, second derivative is calculated
    print("I am calculating the second derivative")
    try: #  get derivation function
        dd_fn = ddfn_list[representation]
    except ValueError:
        dd_fn = ddfn_list['CM']
        print("your representation was not found. falling back to 'CM' for second derivative")

    try: #get second derivative index
        ddx_index = diff_list[ddx]
    except ValueError:
        ddx_index = diff_list['Z']
        print("your ddx value cannot be derived by. falling back to 'Z'")
    

    return(dd_fn(Z, R, N, dx_index, ddx_index, M, order))





'''
The following functions are for calling all derivatives and printing or processing them one by one
'''
def calculate_eigenvalues(repro, compoundlist):
    '''calculates eigenvalues of derived matrices

    Returns:
    --------
    resultlist: list of derivative_result instances, a class
                which contains both norm as well as derivatives,
                eigenvalues and fractual eigenvalue information
    '''
    resultlist = []
    #extract atomic data from compound
    for c in compoundlist:
        Z = jnp.asarray([float(i)for i in c.Z])
        R = c.R
        N = float(c.N)

        #calculate derivatives and representation
        M, order = jrep.CM_full_sorted(Z, R, N)
        #dim = M.shape[0]

        dZ = sort_derivative(repro, Z, R, N, 1, 'Z', M, order)
        dR = sort_derivative(repro, Z, R, N, 1, 'R', M, order)
        
        ddZ = sort_derivative(repro, Z, R, N, 2, 'Z', 'Z', M, order)
        dZdR = sort_derivative(repro, Z, R, N, 2, 'R', 'Z', M, order)
        ddR = sort_derivative(repro, Z, R, N, 2, 'R', 'R', M, order)

        #create derivative results instance
        der_result = datprep.derivative_results(c.filename, Z, M)
        
        #get all derivative eigenvalues for the derivatives and add to results instance
        der_result.add_all_RZev(dZ, dR, ddZ, ddR, dZdR)

        #calculate percentile results and add to results
        percentile_results = der_result.calculate_percentage()

        resultlist.append(der_result)

    return(resultlist)


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
    which_return = [False, False, True]
    '''calculates all second derivatives'''
    if which_return[0]:
        dZdZ = sort_derivative(repro, Z, R, N, 2, 'Z', 'Z')
        for i in range(dim): #3 atoms in H2O
            for j in range(dim): #second dZ over 3 atoms in H2O
                print('dZ%idZ%i' %(i+1, j+1))
                print(dZdZ[i,j])

    if which_return[1]:
        dRdR = sort_derivative(repro, Z, R, N, 2, 'R', 'R')
        print("dRdR derivatives:")

        xyz = [[0,'x'],[1,'y'],[2,'z']]
        for i in range(dim): #3 atoms in H2O
            for x in xyz:
                for j in range(dim):
                    for y in xyz:
                        print('d%s%id%s%i' %(x[1], i+1, y[1], j+1))
                        print(dRdR[i,x[0], j, y[0]])
    
    if which_return[2]:
        dZdR = sort_derivative(repro, Z, R, N, 2, 'Z', 'R')

        print("dZdR derivatives:")

        xyz = [[0,'x'],[1,'y'],[2,'z']]
        for i in range(dim): #3 atoms in H2O
            for j in range(dim):
                for x in xyz:
                    print('dZ%id%s%i' %(i+1, x[1], j+1))
                    print(dZdR[i, j, x[0]])



'''Below follow function specific derivatives with corresponding sorting'''

def d_CM(Z, R, N, dx_index, M = None, order = None):
    '''this function calculates the derivatives of the sorted Coulomb matrix
    variables:
    ----------
    Z: jnp array, nuclear charges, unsorted
    R: jnp array of 3dim arrays, xyz coordinates, unsorted
    N: total electronic charges (irrelevant, for derivative necessary)
    dx_index: int, either 0, 1, or 2, if Z, R or by N should be derived

    returns:
    --------
    dCM*: jnp array, contains sorted derivatives.
        i.e. dZidZj can be retrieved by dCM[i,j]
        dxidyj by dCM[i,x,j,y]
        
    '''
    print("calculating sorted second derivative of Coulomb Matrix")

    fM_sorted, order = jrep.CM_full_sorted(Z, R, N) #get order of sorted representation
    
    dim = len(order)
    #direct derivative as jacobian
    dCM = jacfwd(jrep.CM_full_sorted, dx_index)
    reference_dCM = dCM(Z, R, N)[0]
    
    '''reordering is correct, but signs unclear, check if values of EV are important
    '''
    if(dx_index == 0): #derivative by dZ
        #something does not work in this part
        #if the matrix was padded the derivative is weird
        #assign derivative to correct field in sorted matrix by reordering derZ_ij to derZ_kl
        dCMkl_dZkl = jnp.asarray([[[reference_dCM[l][k][m] for l in range(dim)] for k in range(dim)]for m in order])
        return(dCMkl_dZkl)
    elif(dx_index == 1):
        #unordered derivative taken from sorted matrix
        dCMkl_dRkl = jnp.asarray([[[[reference_dCM[l][k][m][x] for l in range(dim)] for k in range(dim)] for x in range(3)] for m in order])
        return(dCMkl_dRkl)
    else:
        return(reference_dCM)
    '''
    return(reference_dCM)
    '''

def d_CM_ev(Z, R, N, dx_index):
    '''Calculates first derivative of CM_ev w.r.t. dx_index
    sorts results (derivative is taken w.r.t. unsorted Z or R)
    Parameters
    ----------
    Z : 1 x n dimensional array
        contains nuclear charges
    R : 3 x n dimensional array
        contains nuclear positions
    N : float
        number of electrons in system
        here: meaningless, can remain empty
    dx_index : integer
        identifies by which variable to derive by
        (0 : Z, 1 : R, 2 : N)
    Return
    ------
    J : Jacobian of CM_ev
    '''

    print("calculating Jacobian of Coulomb Matrix eigenvalues")
    fM, order = jrep.CM_ev(Z, R, N) #get order of sorted representation
    dim = len(order)
    print('sorted CM eigenvalues :', fM)
    print('order:', order)
    #direct derivative as jacobian
    Jraw = jacfwd(jrep.CM_ev, dx_index)
    J = Jraw(Z, R, N)[0]
    print('unsorted Jacobian is:', J)

    '''Not sure about reordering below. definitely need to recheck. Not important for EV though, or is it?'''
    if(dx_index == 0):
        #assign derivative to correct field in sorted matrix by reordering derZ_ij to derZ_kl
        J_dZkl = jnp.asarray([[J[l][m] for l in range(dim)] for m in order])
        return(J_dZkl)
    elif(dx_index == 1):
        #unordered derivative taken from sorted matrix
        J_dRkl = jnp.asarray([[[J[l][m][x] for l in range(dim)] for x in range(3)] for m in order])
        return(J_dRkl)


def d_OM(Z, R, N, dx_index = 0):
    dim = jrep.OM_dimension(Z)
    Jraw = jacfwd(jrep.OM_full_sorted, dx_index)
    J = Jraw(Z, R, N)
    print('jraw', Jraw, 'J', J)
    return(J)

def dd_OM(Z, R, N, dx_index = 0, ddx_index = 0):
    dim = jrep.OM_dimension(Z)
    Hraw = hessian(jrep.OM_full_sorted, dx_index, ddx_index)(Z, R, N)[0]
    
    print('the matrix was not sorted according to derivatives')
    return(Hraw)

def d_OM_ev(Z, R, N, dx_index = 0):
    dim = jrep.OM_dimension(Z)


def dd_OM_ev(Z, R, N, dx_index = 0, ddx_index = 0):
    return()


def dd_CM(Z, R, N, dx_index = 0, ddx_index = 0, M = None, order = None, time_calculations = True):
    '''
    calculates and sorts second derivatives
    '''
    if M == None:
        fM_sorted, order = jrep.CM_full_sorted(Z, R, N)
    else:
        fM_sorted = M
    dim = len(order)
    
    #calculates dZdZ
    if (dx_index ==0):
        if (ddx_index == 0):
            HdZraw = hessian(jrep.CM_full_sorted, dx_index, ddx_index)(Z, R, N)[0]
            
            '''
            sorting function, performs the following rearrangement:
            [[[[HdZraw[k,l, m, n] for k in range(dim)] for l in range(dim)] for m in order] for n in order])
            '''
                        
            dZdZ_ordered = np.transpose(HdZraw,(2, 3, 0, 1))
            
            dZdZ_sorted = np.copy(dZdZ_ordered)
            for i in range(dZdZ_ordered.shape[0]):
                for j in range(dZdZ_ordered.shape[1]):
                    dZdZ_sorted[i, j] = dZdZ_ordered[order[i], order[j]]
            
            return(dZdZ_sorted)
    
    #calculates dRdR
    if (dx_index == 1):
        if (ddx_index == 1):
            
            HdRraw = hessian(jrep.CM_full_sorted, dx_index, ddx_index)(Z, R, N)[0]
            
            print("do dRdR sorting")
            
            '''
            sorting function, performs the following rearrangement:
            [[[[[[HdRraw[n, m, i, x, j, y] for n in range(dim)] for m in range(dim)] for y in range(3)] for j in order] for x in range(3)] for i in order])
            '''
            dRdR_ordered = np.transpose(HdRraw,(2, 3, 4, 5, 0, 1))
            
            dRdR_sorted = np.copy(dRdR_ordered)
            for i in range(dRdR_ordered.shape[0]):
                for x in range(3):
                    for j in range(dRdR_ordered.shape[2]):
                        for y in range(3):
                            dRdR_sorted[i, x, j, y] = dRdR_ordered[order[i], x, order[j], y]
            
            return(dRdR_sorted)

    if (dx_index == 0 and ddx_index == 1 ) or (dx_index == 1 and ddx_index == 0):
        print("you want to calculate dZdR or dRdZ")

        HdZdRraw = hessian(jrep.CM_full_sorted, 0, 1)(Z, R, N)[0]
        
        '''sorting function, performs the following reordering but in fast
        [[[[[HdZdRraw[n, m, i, j, x] for n in range(dim)] for m in range(dim)] for x in range(3)] for j in order] for i in order]
        '''
        dZdR_ordered = np.transpose(HdZdRraw,(2, 3, 4, 0, 1))
        dZdR_sorted = np.copy(dZdR_ordered)
        
        for i in range(dZdR_ordered.shape[0]):
            for j in range(dZdR_ordered.shape[1]):
                for x in range(3):
                    dZdR_sorted[i,j,x] = dZdR_ordered[order[i], order[j], x]
                    
        return(dZdR_sorted)

def dd_CM_ev(Z, R, N, dx_index = 0, ddx_index = 0):
    fM, order = jrep.CcM_ev(Z, R, N)
    dim = len(order)
    Hraw = hessian(jrep.CM_ev, dx_index, ddx_index)(Z, R, N)[0]
    print('Hraw:', Hraw)

def hessian(f, dx, ddx):
    H = jacfwd(jacfwd(f, dx), ddx)
    return(H)

