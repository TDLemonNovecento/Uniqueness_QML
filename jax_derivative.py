#here we calculate the first and second derivajtives to the basis of chemical space depending on the chosen representation and placeholder. Matrix and vector reconstruction may be included, too.

import numpy as np
import jax.numpy as jnp
import jax_representation as jrep
from jax import grad, jacfwd, jacrev

def sort_derivative(representation, Z, R, N = 0, grad = 2, dx = "R", ddx = "R"):
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
    fn_list = {'CM': jrep.CM_full_sorted, 'CM_EV': jrep.CM_ev}
    dfn_list = {'CM': d_CM}
    ddfn_list = {'CM': dd_CM}

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
    print('sorted CM :', fM_sorted)
    print('order:', order)
    #direct derivative as jacobian
    dCM = jacfwd(jrep.CM_full_sorted, dx_index)
    reference_dCM = dCM(Z, R, N)[0]
    print('reference is', reference_dCM)

    '''Not sure about reordering below. definitely need to recheck. Not important for EV though, or is it?'''
    if(dx_index == 0):
        #assign derivative to correct field in sorted matrix by reordering derZ_ij to derZ_kl
        dCMkl_dZkl = np.asarray([[[reference_dCM[k][l][m] for m in order] for k in range(dim)]for l in range(dim)])
        return(dCMkl_dZkl)
    elif(dx_index == 1):
        #unordered derivative taken from sorted matrix
        dCMkl_dRkl = np.asarray([[[[reference_dCM[l][k][m][x] for l in range(dim)] for k in range(dim)] for x in range(3)] for m in order])
        return(dCMkl_dRkl)
    else:
        return(reference_dCM)


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

def hessian(f, dx, ddx):
    return jacfwd(jacfwd(f, dx), ddx)

