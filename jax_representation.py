'''In this package representation functions are stored and their derivatives returned'''
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap

#Z and R should be jnp.array type
Z = jnp.asarray([[1,2,3]])
R = jnp.asarray([[0, 0, 0], [1, 0, 0], [2, 0 , 0]])
print(Z,R)

def CM_trial(i, j):
    if i == j:
        S_ij = (Z[i]**(2.4))/2
    else:
        S_ij = Z[i]*Z[j]/(jnp.abs(R[i]-R[j]))
    return(S_ij)


def derivative(fun, dx = [0,0]):
    if dx[0] == 0:
        d_fM = grad(fun, dx)(Z[1], Z[2], Z[3])
    elif dx[0] == 1:
        d_fM = grad(fun, dx)(R[1], R[2], R[3])
    else:
        d_fM = grad(fun, dx)(N)
    return(d_fM)


derivative(CM_trial(1,2))
