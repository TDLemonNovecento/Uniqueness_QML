#here we calculate the first and second derivajtives to the basis of chemical space depending on the chosen representation and placeholder. Matrix and vector reconstruction may be included, too.

import jax.numpy as jnp
import jax_representation as jrep
from jax import grad, jacfwd, jacrev

def sort_derivative(representation, Z, R, N = 0, grad = 1, dx = "Z", ddx = "R"):
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
        return( fn(Z, R, N)[0])
    
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
        
        return(d_fn(Z, R, N, dx_index))
    

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
    

    return(dd_fn(Z, R, N, dx_index, ddx_index))

'''Below follow function specific derivatives with corresponding sorting'''











def d_CM(Z, R, N, dx_index):
    
    print("calculating sorted second derivative of Coulomb Matrix")
    fM_sorted, order = jrep.CM_full_sorted(Z, R, N) #get order of sorted representation
    dim = len(order)
    #print('sorted CM :', fM_sorted)
    #print('order:', order)
    #direct derivative as jacobian
    dCM = jacfwd(jrep.CM_full_sorted, dx_index)
    reference_dCM = dCM(Z, R, N)[0]
    #print('reference is', reference_dCM)
    #print('unsorted derivative has successfully been calculated, starting ordering')
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
        J_dZkl = np.asarray([[J[l][m] for l in range(dim)] for m in order])
        return(J_dZkl)
    elif(dx_index == 1):
        #unordered derivative taken from sorted matrix
        J_dRkl = np.asarray([[[J[l][m][x] for l in range(dim)] for x in range(3)] for m in order])
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


def dd_CM(Z, R, N, dx_index = 0, ddx_index = 0):
    fM_sorted, order = jrep.CM_full_sorted(Z, R, N)
    dim = len(order)

    Hraw = hessian(jrep.CM_full_sorted, dx_index, ddx_index)(Z, R, N)[0]
    
    if (dx_index == 0):
        if (ddx_index == 0):
            H_ddkl = np.asarray([[[[Hraw[k][l][m][n] for  m in order] for n in order] for k in range(dim)] for l in range(dim)])
    elif (dx_index == 1):
        if (ddx_index == 1):
            H_ddkl = np.asarray([[[[[[Hraw[k][l][m][x][n][y] for l in range(dim)] for k in range(dim)] for n in order] for m in order] for x in range(3)] for y in range(3)])

    print('Hraw \n', Hraw)
    print('Hsorted\n', H_ddkl)

    return(H_ddkl)

def dd_CM_ev(Z, R, N, dx_index = 0, ddx_index = 0):
    fM, order = jrep.CM_ev(Z, R, N)
    dim = len(order)
    Hraw = hessian(jrep.CM_ev, dx_index, ddx_index)(Z, R, N)[0]
    print('Hraw:', Hraw)

def hessian(f, dx, ddx):
    return jacfwd(jacfwd(f, dx), ddx)

