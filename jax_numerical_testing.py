'''
This program prints all numerical derivatives of a function
'''
import jax.numpy as jnp
import jax.ops
import jax_representation as jrep
import database_preparation as datprep
import jax_derivative as jder

#give xyz coordinates of H2O
path = "./TEST/H2O.xyz"

#read xyz file and create compound instances

Z_orig, R_orig, N, e_orig = datprep.read_xyzfile(path)
M, order = jrep.CM_full_sorted(Z_orig, R_orig, N)

#reorder your initial data. otherwise numerical differentiation will fail
#needs to be performed with preordered data, otherwise sorting can change

Z = Z_orig[order]
R = R_orig[order]
print("Z: ", Z, "R: ", R)
dim = len(Z)
xyz = [[0,'x'],[1,'y'],[2,'z']]
h1 = 0.1
h2 = 0.1
Z = jnp.asarray(Z, dtype = jnp.float32)
fun = jrep.CM_full_sorted
#fun = jrep.CM_ev

results = jder.num_second_mixed_derivative(fun, [Z, R, N], [0,0], [1,0,0], h1, h2)
print(results)
sys.exit()

#store all results in arrays to print them later
Zder = []
Rder = []
Z2der = []
R2der = []
ZRder = []

#do all Z derivatives
#for i in range(dim):
for i in range(1):    
    name = ("dZ%i:" % (i+1))
    Zminus = jax.ops.index_add(Z, i, -h1)
    Zplus = jax.ops.index_add(Z,i, h1)
    
    #der = jder.num_first_derivative(fun, [Z, R, N], [Zplus, R, N], [Zminus, R, N], 'central', h1)
    #Zder.append([name,der])

    #do all dZdZ derivatives:
    #for j in range(i, dim):
    for j in range(1):
        name = ("dZ%i dZ%i:" %( i+1, j+1))
        Zminusminus = jax.ops.index_add(Zminus, j, -h2)
        Zplusplus = jax.ops.index_add(Zplus, j, h2)
        #if i==j I am using a same derivative function, otherwise this is a mixed derivative
        if(j == i):
            der = jder.num_second_pure_derivative(fun, [Z, R, N], [Zplusplus, R, N], [Zplus, R, N], [Zminus, R, N], [Zminusminus, R, N], "central", h1, h2)
        else:
            Zplusminus = jax.ops.index_add(Zplus, j, -h2)
            Zminusplus = jax.ops.index_add(Zminus, j, h2)
            der = jder.num_second_mixed_derivative(fun, [Zplusplus, R, N], [Zplusminus, R, N], [Zminusplus, R, N], [Zminusminus, R, N],  h1, h2) 
        
        Z2der.append([name, der])
        
        #do all dZdR derivatives:
        for x in xyz:
            name = ("dZ%i d%s%i:" %(i+1, x[1], j+1))
            Rplus = jax.ops.index_update(R, j, jax.ops.index_add(R[j], x[0], h2))
            Rminus = jax.ops.index_update(R, j, jax.ops.index_add(R[j], x[0], -h2))
            der = jder.num_second_mixed_derivative(fun, [Zplus, Rplus, N], [Zplus, Rminus, N], [Zminus, Rplus, N], [Zminus, Rminus, N], h1, h2)
            ZRder.append([name,der])

for i in range(1):
    #do all dR derivatives:
    #for y in xyz:
    for y in [[1,'y']]:
        print(y)
        name = ("d%s%i :" %(y[1], i+1))
        Rplus = jax.ops.index_update(R, i, jax.ops.index_add(R[i], y[0], h1))
        Rminus = jax.ops.index_update(R, i, jax.ops.index_add(R[i], y[0], -h1))

        #der = jder.num_first_derivative(fun, [Z, R, N], [Z, Rplus, N], [Z, Rminus, N], 'central', h1)
        #Rder.append([name,der])

        #do all dRdR derivatives:
        for k in range(1):
        #for k in range(i, dim):
            for z in [[0, 'x']]:
            #for z in xyz:
                name = ("d%s%i d%s%i :" %(y[1], i+1, z[1], k+1))
                Rplusplus = jax.ops.index_update(Rplus, k, jax.ops.index_add(Rplus[k], z[0], h2))
                Rminusminus = jax.ops.index_update(Rminus, k, jax.ops.index_add(Rminus[k], z[0], -h2))
 

                if (i == k) and (y[0] == z[0]):
                    print("pure double derivative")
                    der = jder.num_second_pure_derivative(fun, [Z, R, N], [Z, Rplusplus, N], [Z, Rplus, N], [Z, Rminus, N], [Z, Rminusminus, N], 'central', h1, h2)

                else:
                    der = jder.num_second_mixed_derivative(fun, [Z, R, N], [i, y[0]], [k, z[0]], h1, h2)
                R2der.append([name,der])
    
#now print results properly:
print("all dR derivatives")
for i in Rder:
    print(i[0], "\n", i[1])

print("all dZ derivatives")
for i in Zder:
    print(i[0], "\n", i[1])

print("all dZdZ derivatives")
for i in Z2der:
    print(i[0], "\n", i[1])

print("all dRdR derivatives")    
for i in R2der:
    print(i[0], "\n", i[1])

print("all dZdR derivatives")    
for i in ZRder:
    print(i[0], "\n", i[1])


            


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
