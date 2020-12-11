'''
This program prints all numerical derivatives of a function
'''
import jax.numpy as jnp
import jax.ops
import jax_representation as jrep
import database_preparation as datprep
import jax_derivative as jder

#give xyz coordinates of H2O
path = "/home/linux-miriam/Uniqueness_QML/TEST/H2O.xyz"

#read xyz file and create compound instances

Z, R, N, e = datprep.read_xyzfile(path)
dim = len(Z)
xyz = [[0,'x'],[1,'y'],[2,'z']]
h1 = 0.001
h2 = 0.001
Z = jnp.asarray(Z, dtype = jnp.float32)
fun = jrep.CM_full_sorted
#fun = jrep.CM_ev
#do all Z derivatives
for i in range(dim):
    print("dZ%i derivative:" % (i+1))
    
    Zminus = jax.ops.index_add(Z, i, -h1)
    Zplus = jax.ops.index_add(Z,i, h1)
    
    der = jder.num_first_derivative(fun, [Z, R, N], [Zplus, R, N], [Zminus, R, N], 'central', h1)
    print(der)
    sys.exit()
    #do all dZdZ derivatives:
    for j in range(dim):
        print("dZ%i dZ%i derivative:" %( i+1, j+1))
        Zminusminus = jax.ops.index_add(Zminus, j, -h2)
        Zplusplus = jax.ops.index_add(Zplus, j, h2)
        if(j == i):
            der = jder.num_second_pure_derivative(fun, [Z, R, N], [Zplusplus, R, N], [Zplus, R, N], [Zminus, R, N], [Zminusminus, R, N])
        else:
            Zplusminus = jax.ops.index_add(Zplus, j, -h2)
            Zminusplus = jax.ops.index_add(Zminus, j, h2)
            der = jder.num_second_mixed_derivative(fun, [Zplusplus, R, N], [Zplusminus, R, N], [Zminusplus, R, N], [Zminusminus, R, N], 'central', h1, h2) 
        print(der)
        #do all dZdR derivatives:
        for x in xyz:
            print("dZ%i d%s%i derivative:" %(i+1, x[1], j+1))
            Rplus = jax.ops.index_update(R, j, jax.ops.index_add(R[j], x[0], h2))
            Rminus = jax.ops.index_update(R, j, jax.ops.index_add(R[j], x[0], -h2))
            der = jder.num_second_mixed_derivative(fun, [Zplus, Rplus, N], [Zplus, Rminus, N], [Zminus, Rplus, N], [Zminus, Rminus, N], 'central', h1, h2)
            print(der)

    #do all dR derivatives:
    for y in xyz:
        print("d%s%i derivative:" %(y[1], i+1))
        Rplus = jax.ops.index_update(R, i, jax.ops.index_add(R[i], y[0], h1))
        Rminus = jax.ops.index_update(R, i, jax.ops.index_add(R[i], y[0], -h1))

        der = jder.num_first_derivative(fun, [Z, R, N], [Z, Rplus, N], [Z, Rminus, N], 'central', h1)
        print(der)

        #do all dRdR derivatives:
        for k in range(dim):
            for z in xyz:
                print("d%s%i d%s%i derivative:" %(y[1], i+1, z[1], k+1))
                Rplusplus = jax.ops.index_update(Rplus, k, jax.ops.index_add(Rplus[k], z[0], h2))
                Rminusminus = jax.ops.index_update(Rminus, k, jax.ops.index_add(Rminus[k], z[0], -h2))
 

                if (i == k) and (y[0] == z[0]):
                    print("pure double derivative")
                    der = jder.num_second_pure_derivative(fun, [Z, R, N], [Z, Rplusplus, N], [Z, Rplus, N], [Z, Rminus, N], [Z, Rminusminus, N], 'central', h1, h2)

                else:
                    Rplusminus = jax.ops.index_update(Rminus, k, jax.ops.index_add(Rplus[k], z[0], -h2))
                    Rminusplus = jax.ops.index_update(Rminus, k, jax.ops.index_add(Rplus[k], z[0], h2))
                    der = jder.num_second_mixed_derivative(fun, [Z, Rplusplus, N], [Z, Rplusminus, N], [Z, Rminusplus, N], [Z, Rminusminus, N])

                print(der)
    

            


def num_derivative(f, ZRN, ZRNplus, ZRNminus, method='central', h = 0.1):
    '''Compute the difference formula for f'(a) with step size h.

    Parameters
    ----------
    f : function
        Vectorized function of one variable
    a : number
        Compute derivative at x = a
    method : string
        Difference formula: 'forward', 'backward' or 'central'
    h : number
        Step size in difference formula

    Returns
    -------
    float
        Difference formula:
            central: f(a+h) - f(a-h))/2h
            forward: f(a+h) - f(a))/h
            backward: f(a) - f(a-h))/h            
    '''
    if method == 'central':
        return (f(ZRNplus[0], ZRNplus[1], ZRNplus[2]) - f(ZRNminus[0], ZRNminus[1], ZRNminus[2]))/(2*h)
    elif method == 'forward':
        return (f(a + h) - f(a))/h
    elif method == 'backward':
        return (f(a) - f(a - h))/h
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")
