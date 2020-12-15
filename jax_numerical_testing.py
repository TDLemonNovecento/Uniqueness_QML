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


#store all results in arrays to print them later
Zder = []
Rder = []
Z2der = []
R2der = []
ZRder = []

#do all Z derivatives
for i in range(dim):
    name = ("dZ%i:" % (i+1))
    
    der = jder.num_first_derivative(fun, [Z, R, N], [0, i], h1, 'central')
    Zder.append([name,der])

    #do all dZdZ derivatives:
    for j in range(dim):
        name = ("dZ%i dZ%i:" %( i+1, j+1))
        
        der = jder.num_second_derivative(fun, [Z, R, N], [0,i],[0,j], h1, h2) 
        
        Z2der.append([name, der])
        
        #do all dZdR derivatives:
        for x in xyz:
            name = ("dZ%i d%s%i:" %(i+1, x[1], j+1))

            der = jder.num_second_derivative(fun, [Z, R, N], [0, i], [1,j, x[0]], h1, h2)
            ZRder.append([name,der])
    
    #do all dR derivatives:
    for y in xyz:
        name = ("d%s%i :" %(y[1], i+1))

        der = jder.num_first_derivative(fun, [Z, R, N], [1, i, y[0]], h1, 'central')
        Rder.append([name,der])

        #do all dRdR derivatives:
        for k in range(i, dim):
            for z in xyz:
                name = ("d%s%i d%s%i :" %(y[1], i+1, z[1], k+1))

                der = jder.num_second_derivative(fun, [Z, R, N], [1, i, y[0]], [1, k, z[0]], h1, h2)
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
