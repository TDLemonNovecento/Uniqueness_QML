'''In this package representation functions are stored and their derivatives returned'''
import jax.numpy as jnp
import numpy as np
from jax import grad, ops
import qml
#Z and R should be jnp.array type, form might differ if uploaded from xyz
Z = jnp.asarray([[1.,2.,3.]])
R = jnp.asarray([[0.,0.,0.],[1.,1.,1.],[2.,2.,2.]])


def CM_trial(Z, R):
    n = Z.shape[1]
    print("size of matrix is %i" % n)
    D = jnp.zeros((n, n))
    
    #indexes need to be adapted to whatever form comes from xyz files
    for i in range(n):
        Zi = Z[0,i]
        D = ops.index_update(D, (i,i), Zi**(2.4)/2)
        for j in range(n):
            if j != i:
                Zj = Z[0,j]
                Ri = R[i, :]
                Rj = R[j, :]
                distance = jnp.linalg.norm(Ri-Rj)
                D = ops.index_update(D, (i,j) , Zi*Zj/(distance))
    return(D)
            
def CM_ev(Z,R):
    M = CM_trial(Z,R)
    ev, vectors = jnp.linalg.eigh(M)
    print(ev)

def CM_index(Z, R, i, j):
    n = Z.shape[1]
    Zi = Z[0,i]
    if i == j:
        return(Zi**(2.4)/2)
    else:
        Zj = Z[0,j]
        Ri = R[i, :]
        Rj = R[j, :]
        distance = jnp.linalg.norm(Ri-Rj)
        return( Zi*Zj/(distance))


def derivative(fun, dx = [0,0]):
    if dx[0] == 0:
        d_fM = grad(fun, dx[1])(Z[0], Z[1], Z[2])
    elif dx[0] == 1:
        d_fM = grad(fun, dx[1])(R[1], R[2], R[3])
    else:
        d_fM = grad(fun)(N)
    return(d_fM)


def trial(i,j):
    if (i==j):
        k = i**2.4/2
    else:
        k = i*j
    return(k)

print(CM_ev(Z,R))
print(CM_trial(Z,R))
der1 = grad(CM_index)
print(der1(Z, R, 1, 0))
