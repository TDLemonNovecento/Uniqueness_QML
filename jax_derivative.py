#here we calculate the first and second derivatives to the basis of chemical space depending on the chosen representation and placeholder. Matrix and vector reconstruction may be included, too.


import jax.numpy as jnp
import jax_representation as jrep

def j_derivative(fun, Z, R, N = 0, grad = 0, dx = "Z", ddx = "Z"):
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

    if grad == 0:
        return( fun(Z, R, N))
    
    difflist = ["Z", "R", "N"]
    try:
        dx_index = difflist.index(dx)
    except ValueError:
        dx_index = 0
        print("your dx value cannot be derived by. falling back to 'Z'")

    d_fun = grad(fun, dx_index)
    if grad == 1:
        return(d_fun(Z, R, N))
    
    try:
        ddx_index = difflist.index(ddx)
    except ValueError:
        ddx_index = 0
        print("your ddx value cannot be derived by. falling back to 'Z'")

    dd_fun = grad(d_fun, ddx_index)
    if grad == 2:
        return(dd_fun(Z, R, N))
    else:
        print("your grad value was out of bounds")
        return()

