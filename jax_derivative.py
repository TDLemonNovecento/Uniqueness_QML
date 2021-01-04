'''
here we calculate the first and second derivatives to the basis of chemical space depending on the chosen representation and placeholder. Matrix and vector reconstruction may be included, too.
the following functions were moved to "jax_additional_derivative.py" and may cause problems upon calling:
    calculate_eigenvalues
    cal_print_1stder
    cal_print_2ndder
    update_index
    num_first_derivative
    num_second_derivative
    num_second_pure_derivative
'''
import numpy as np
import jax.numpy as jnp
import jax_derivative as jder
import jax_representation as jrep
from jax import grad, jacfwd, jacrev, ops
import time
import database_preparation as datprep
from jax.config import config
config.update("jax_enable_x64", True) #increase precision from float32 to float64


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
    ''' idea: the whole code might be more efficient if representation and order were passed on
    instead of calculated freshly at every derivative
    '''
    fn_list = {'CM': jrep.CM_full_sorted, 'CM_unsrt' : jrep.CM_full_unsorted_matrix, 'CM_EV': jrep.CM_ev, 'CM_EV_unsrt' : jrep.CM_ev_unsrt, 'OM' : jrep.OM_full_sorted}
    dfn_list = {'CM': d_CM, 'CM_unsrt': d_CM_unsrt, 'CM_EV' : d_CM_ev, 'CM_EV_unsrt': d_CM_ev_unsrt, 'OM' : d_OM, 'OM_EV' : d_OM_ev}
    ddfn_list = {'CM': dd_CM, 'CM_unsrt': dd_CM_unsrt, 'CM_EV' : dd_CM_ev, 'CM_EV_unsrt' : dd_CM_ev_unsrt, 'OM' : dd_OM, 'OM_EV' : dd_OM_ev}
    

    try:
        fn = fn_list[representation]
    except ValueError:
        fn = fn_list['CM']
        print("your representation was not found. falling back to 'CM'")
    
    


    if grad == 0:
        #print("calculating the representation itself")
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
        #print("calculating the first derivative of the representation")
        try:
            d_fn = dfn_list[representation]
        except ValueError:
            d_fn = dfn_list['CM']
            print("your representation was not found. falling back to 'CM' for first derivative")
        
        return(d_fn(Z, R, N, dx_index))
    

    #grad is 2 or bigger, second derivative is calculated
    #print("calculating the second derivative of the representation")
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
    

    return(dd_fn(Z, R, N, dx_index, ddx_index))


'''Below follow function specific derivatives with corresponding sorting'''

def d_CM(Z, R, N, dx_index):
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
    #print("calculating sorted second derivative of Coulomb Matrix")

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

def d_CM_unsrt(Z, R, N, dx_index):
    '''this function calculates the derivatives of the sorted Coulomb matrix
    variables:
    ----------
    Z: jnp array, nuclear charges, unsorted
    R: jnp array of 3dim arrays, xyz coordinates, unsorted
    N: total electronic charges (irrelevant, for derivative necessary)
    dx_index: int, either 0, 1, or 2, if Z, R or by N should be derived

    returns:
    --------
    final_dCM: jnp array, contains sorted derivatives.
        i.e. dZidZj can be retrieved by dCM[i,j]
        dxidyj by dCM[i,x,j,y]
        
    '''
    #print("calculating unsorted second derivative of Coulomb Matrix")

    #direct derivative as jacobian
    dCM_unsrt = jacfwd(jrep.CM_full_unsorted_matrix, dx_index)
   
    reference_dCM = dCM_unsrt(Z, R, N)

    dim = len(Z)
    
    '''reordering is correct, but signs unclear, check if values of EV are important
    '''
    if(dx_index == 0): #derivative by dZ
        #something does not work in this part
        #if the matrix was padded the derivative is weird
        #assign derivative to correct field in sorted matrix by reordering derZ_ij to derZ_kl
        dCMkl_dZkl = jnp.asarray([[[reference_dCM[l][k][m] for l in range(dim)] for k in range(dim)]for m in range(dim)])
        return(dCMkl_dZkl)
    elif(dx_index == 1):
        #unordered derivative taken from sorted matrix
        dCMkl_dRkl = jnp.asarray([[[[reference_dCM[l][k][m][x] for l in range(dim)] for k in range(dim)] for x in range(3)] for m in range(dim)])
        return(dCMkl_dRkl)
    else:
        return(reference_dCM)

    return(final_dCM)


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
    
    #direct derivative as jacobian
    dCM_ev = jacfwd(jrep.CM_ev, dx_index)
    ref_dCM_ev = dCM_ev(Z, R, N)

    #ref_dCM_ev = dCM_ev(Z.astype(float), R.astype(float), float(N))[0]
    
    '''Not sure about reordering and values below. definitely need to recheck'''
    if(dx_index == 0):
        #assign derivative to correct field in sorted matrix by reordering derZ_ij to derZ_kl
        J_dZkl = jnp.asarray([[ref_dCM_ev[l][m] for l in range(dim)] for m in order])
        return(J_dZkl)
    elif(dx_index == 1):
        #unordered derivative taken from sorted matrix
        J_dRkl = jnp.asarray([[[ref_dCM_ev[l][m][x] for l in range(dim)] for x in range(3)] for m in order])
        return(J_dRkl)
    else:
        return(ref_dCM_ev)

def d_CM_ev_unsrt(Z, R, N, dx_index):
    '''
    Calculates first derivative of unsorted Coulomb Matrix eigenvalues w.r.t. dx_index
    '''
    dim = len(Z)
    print("len of Z:", dim)
    print("calculating Jacobian of Coulomb Matrix eigenvalues")
    fM = jrep.CM_ev_unsrt(Z, R, N) #get order of sorted representation

    #direct derivative as jacobian
    dCM_ev = jacfwd(jrep.CM_ev_unsrt, dx_index)
    ref_dCM_ev = dCM_ev(Z, R, N)

    '''Not sure about reordering and values below. definitely need to recheck'''
    if(dx_index == 0):
        J_dZkl = jnp.asarray([[ref_dCM_ev[l][m] for l in range(dim)] for m in range(dim)])
        return(J_dZkl)
    elif(dx_index == 1):
        #unordered derivative taken from sorted matrix
        J_dRkl = jnp.asarray([[[ref_dCM_ev[l][m][x] for l in range(dim)] for x in range(3)] for m in range(dim)])
        return(J_dRkl)
    else:
        return(ref_dCM_ev)


def d_OM(Z, R, N, dx_index = 0):
    dim = jrep.OM_dimension(Z)
    print("dimension of OM is: ", dim)
    Jraw = jacfwd(jrep.OM_full_unsorted_matrix, dx_index)
    J = Jraw(Z, R, N)
    

    print("derivative: ")
    print(J)
    if(dx_index == 0): #derivative by dZ
        final_dOM = jnp.asarray([[[J[l][k][m] for l in range(dim)] for k in range(dim)]for m in range(dim)])
        return(final_dOM)

    elif(dx_index == 1):
        #unordered derivative taken from sorted matrix
        final_dOM = jnp.asarray([[[[J[l][k][m][x] for l in range(dim)] for k in range(dim)] for x in range(3)] for m in range(dim)])
        return(final_dOM)
    else:
        return(J)

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
            [[[[HdZraw[k,m, n] for k in range(dim)] for m in order] for n in order])
            '''
                        
            dZdZ_ordered = np.transpose(HdZraw,(2, 0, 1))
            
            dZdZ_sorted = np.copy(dZdZ_ordered)
            for m in range(dZdZ_ordered.shape[0]):
                for n in range(dZdZ_ordered.shape[1]):
                    dZdZ_sorted[m, n] = dZdZ_ordered[order[m], order[n]]
            
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
        print("you want to calculate dZdR or dRdZ, line 372")

        HdZdRraw = hessian(jrep.CM_full_sorted, 0, 1)(Z, R, N)[0]
        
        '''sorting function, performs the following reordering but in fast
        [[[[[HdZdRraw[n, m, i, j, x] for n in range(dim)] for m in range(dim)] for x in range(3)] for j in order] for i in order]
        '''
        #could be that I messed this up on 1.12.2020, check with push before
        dZdR_ordered = np.transpose(HdZdRraw,(1, 2, 3, 4, 0))
        dZdR_sorted = np.copy(dZdR_ordered)
 
        for i in range(dZdR_ordered.shape[0]):
            for j in range(dZdR_ordered.shape[1]):
                for x in range(3):
                    dZdR_sorted[i,j,x] = dZdR_ordered[order[i], order[j], x]
                    
        return(dZdR_sorted)


def dd_CM_unsrt(Z, R, N, dx_index = 0, ddx_index = 0, M = None, order = None):
    '''
    calculates and sorts second derivatives
    '''
    #calculates dZdZ
    if (dx_index ==0):
        if (ddx_index == 0):
            HdZraw = hessian(jrep.CM_full_unsorted_matrix, dx_index, ddx_index)(Z, R, N)
            
            return(HdZraw)

    #calculates dRdR
    if (dx_index == 1):
        if (ddx_index == 1):

            HdRraw = hessian(jrep.CM_full_unsorted_matrix, dx_index, ddx_index)(Z, R, N)
            print("do dRdR sorting")
            '''
            sorting function, performs the following rearrangement:
            [[[[[[HdRraw[n, m, i, x, j, y] for n in range(dim)] for m in range(dim)] for y in range(3)] for j in range(dim)] for x in range(3)] for i in range(dim)])
            '''
            dRdR_sorted = np.transpose(HdRraw,(2, 3, 4, 5, 0, 1))

            return(dRdR_sorted)

    if (dx_index == 0 and ddx_index == 1 ) or (dx_index == 1 and ddx_index == 0):
        print("you want to calculate dZdR or dRdZ, line 372")

        HdZdRraw = hessian(jrep.CM_full_unsorted_matrix, 0, 1)(Z, R, N)
        print("raw dRdZ derivative analytical \n", HdZdRraw)
        '''sorting function, performs the following reordering but in fast
        [[[[[HdZdRraw[n, m, i, j, x] for n in range(dim)] for m in range(dim)] for x in range(3)] for j in range(dim)] for i in range(dim)]
        '''
        #could be that I messed this up on 1.12.2020, check with push before
        dZdR_sorted = np.transpose(HdZdRraw,(2, 3, 4, 0, 1))
        
        return(dZdR_sorted)

def dd_CM_ev(Z, R, N, dx_index = 0, ddx_index = 0):

    Z = Z.astype(float)
    R = R.astype(float)
    N = float(N)

    fM, order = jrep.CM_ev(Z, R, N)
    dim = len(Z)
    Hraw = hessian(jrep.CM_ev, dx_index, ddx_index)(Z, R, N)[0]

    '''
    calculates and sorts second derivatives
    '''
    fM_sorted, order = jrep.CM_full_sorted(Z, R,N)
    dim = len(order)

    #calculates dZdZ
    if (dx_index ==0):
        if (ddx_index == 0):
            '''
            sorting function, performs the following rearrangement:
            [[[[HdZraw[k, m, n] for k in range(dim)] for m in order] for n in order])
            '''
            dZdZ_ordered = np.transpose(Hraw,(2, 0, 1))

            dZdZ_sorted = np.copy(dZdZ_ordered)
            for i in range(dZdZ_ordered.shape[0]):
                for j in range(dZdZ_ordered.shape[1]):
                    dZdZ_sorted[i, j] = dZdZ_ordered[order[i], order[j]]

            return(dZdZ_sorted)

    #calculates dRdR
    if (dx_index == 1):
        if (ddx_index == 1):

            print("do dRdR sorting")
            '''
            sorting function, performs the following rearrangement:
            [[[[[[HdRraw[n, i, x, j, y] for n in range(dim)] for y in range(3)] for j in order] for x in range(3)] for i in order])
            '''
            dRdR_ordered = np.transpose(Hraw,(1, 2, 3, 4, 0))
        
            dRdR_sorted = np.copy(dRdR_ordered)
            for i in range(dRdR_ordered.shape[0]):
                for x in range(3):
                    for j in range(dRdR_ordered.shape[0]):#why does dim not work?? 
                        for y in range(3):
                            dRdR_sorted[i, x, j, y] = dRdR_ordered[order[i], x, order[j], y]

            return(dRdR_sorted)

    if (dx_index == 0 and ddx_index == 1 ) or (dx_index == 1 and ddx_index == 0):
        print("you want to calculate dZdR or dRdZ, line 443")

        '''sorting function, performs the following reordering but in fast
        [[[[[HdZdRraw[m, i, j, x] for m in range(dim)] for x in range(3)] for j in order] for i in order]
        '''
        dZdR_ordered = np.transpose(Hraw,(1, 3, 2, 0))
        dZdR_sorted = np.copy(dZdR_ordered)

        for i in range(dZdR_ordered.shape[0]):
            for j in range(dZdR_ordered.shape[1]):
                for x in range(3):
                    dZdR_sorted[i,j,x] = dZdR_ordered[order[i], order[j], x]

        return(dZdR_sorted)

    if (dx_index == 2 or ddx_index == 2):
        return(Hraw)


def dd_CM_ev_unsrt(Z, R, N, dx_index = 0, ddx_index = 0):
   
    Z = Z.astype(float)
    R = R.astype(float)
    N = float(N)
    
    fM = jrep.CM_ev_unsrt(Z, R, N)
    dim = len(Z)
    Hraw = hessian(jrep.CM_ev_unsrt, dx_index, ddx_index)(Z, R, N)


    #calculates dZdZ
    if (dx_index ==0):
        if (ddx_index == 0):
            '''
            sorting function, performs the following rearrangement:
            [[[[HdZraw[k, m, n] for k in range(dim)] for m in order] for n in order])
            '''
            dZdZ_ordered = np.transpose(Hraw,(1, 2, 0))

            return(dZdZ_ordered)
    
    #calculates dRdR
    if (dx_index == 1):
        if (ddx_index == 1):

            print("do dRdR sorting for unsorted ddCM_ev function")
            '''
            sorting function, performs the following rearrangement:
            [[[[[[HdRraw[n, i, x, j, y] for n in range(dim)] for y in range(3)] for j in range(dim)] for x in range(3)] for i in range(dim)])
            '''
            dRdR_ordered = np.transpose(Hraw,(1, 2, 3, 4, 0))

            return(dRdR_ordered)

    if (dx_index == 0 and ddx_index == 1 ) or (dx_index == 1 and ddx_index == 0):
        print("you want to calculate dZdR or dRdZ, line 443")

        '''sorting function, performs the following reordering but in fast
        [[[[[HdZdRraw[m, i, j, x] for m in range(dim)] for x in range(3)] for j in range(dim)] for i in range(dim)]
        '''
        dZdR_ordered = np.transpose(Hraw,(1, 2, 3, 0))

        return(dZdR_ordered)
    
    if (dx_index == 2 or ddx_index == 2):
        return(Hraw)




def hessian(f, dx, ddx):
    H = jacfwd(jacfwd(f, dx), ddx)
    return(H)

