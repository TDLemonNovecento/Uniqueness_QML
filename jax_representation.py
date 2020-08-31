'''In this package representation functions are stored and their derivatives returned'''
import jax.numpy as jnp
import numpy as np
from jax import grad, ops
import qml
import basis
from scipy import misc, special, linalg


#Z and R should be jnp.array type, form might differ if uploaded from xyz
Z = jnp.asarray([[8.,1., 1.]])
R = jnp.asarray([[0.,0., 0.227],[0.,1.353,-0.908],[0.,-1.353,-0.908]])
N = 3.

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
            
def CM_ev(Z, R, N =  0., i = 0):
    '''
    Parameters
    ----------
    Z : 1 x n dimensional array
        contains nuclear charges
    R : 3 x n dimensional array
        contains nuclear positions
    N : float
        number of electrons in system
        here: meaningless, can remain empty
    i : integer
        identifies EV to be returned

    Return
    ------
    ev : scalar
        Eigenvalue EV(i)
        If i out of bounds, return none and print error
    '''

    M = CM_trial(Z,R)
    ev, vectors = jnp.linalg.eigh(M)
    
    if i in range(len(ev)):
        return(ev[i])
    else:
        print("EV integer out of bounds, maximal i possible: %i" % len(ev))
        return()

def CM_index(Z, R, N, i = 0, j = 0):
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

def OM_trial(Z, R, N):
    covalent_rad = [rdkit.Chem.PeriodicTable.GetRCovalent(Z_i) for Z_i in Z]
    return(covalent_rad)


def OM_compute_norm(alpha, l, m, n):
    '''compute normalization constant for overlap matrix'''

    N = (2*alpha/np.pi)**(3/2)*(4*alpha)**(l+m+n) \
            /(factorial2(2*l-1)*factorial2(2*m-1)*factorial2(2*n-1))
    N = N**(1/2)
    return(N)

def factorial2(n):
        if n <= 0:
            return 1
        else:
            return n * factorial2(n - 2)


def OM_compute_Si(qA, qB, rPAq, rPBq, gamma):
    '''
    Computes center between two curves by employing
    Gaussian product theorem

    Variables
    ---------
    qA : integer
          quantum number (l for Sx, m for Sy, n for Sz) for atom A
    qB : integer
          same for atom B
    rP : array 3D
          center between A and B
    rAq : float
          Cartesian coordinate of dimension q for atom A (rA[0] for Sx e.g.)
    rBq : float
             
    
    Returns
    -------
    '''
    def binomial(n, k):
        '''fast way to calculate binomial coefficients by Andrew Dalke'''
        if not 0 <= k <=n: return 0
        b = 1
        for t in range(min(k, n-k)):
            b*=n
            b /= (t + 1)
            n -= 1
        return b

    def ck(l, m, a, b, k):
        c = 0
        for i in range(l+1):
            for j in range(m+1):
                if (j + i == k):
                    c += binomial(l, i)*binomial(m,j)*a**(l-i)*b**(m - j)
        return(c)

    Sq = 0
    
    for k in range(int((qA + qB)/2)+1): #loop over only even numbers for sum(qA, qB)
        c = ck(qA, qB, rPAq, rPBq, k)
        Sq += c * (np.pi/ gamma)**(1/2) ** factorial2(2*k-1)/((2*gamma)**k)
    return(Sq)


def OM_build_S(basis, K):
    S = np.zeros((K,K))
    for a, bA in enumerate(basis):      #access row a of S matrix; unpack list from tuple
        for b, bB in enumerate(basis):  #same for B
            if (a == b):
                S[a,b] = 1
            else:
                rA = np.array(bA['r']) #get atom centered coordinates of atom A
                rB = np.array(bB['r'])
                lA,mA,nA = bA['l'],bA['m'],bA['n'] #get angular momentumnumbers of A
                lB,mB,nB = bB['l'],bB['m'],bB['n']
                disAB = np.linalg.norm(rA - rB)
                aA, aB = np.array(bA['a']), np.array(bB['a']) #alpha vectors
                rP = np.add(np.dot(aA, rA),np.dot(aB, rB))/ np.add(aA, aB) #calculate weighted center
                rPA = np.subtract(rP,rA) # distance between weighted center and A
                rPB = np.subtract(rP,rB)
                

                for alphaA, dA in zip(bA['a'], bA['d']): #alpha is exp. coeff. and dA contraction coeff.
                    for alphaB, dB in zip(bB['a'], bB['d']):
                        
                        #Implement overlap element
                        gamma = alphaA + alphaB
                        normA = OM_compute_norm(alphaA, lA, mA, nA) #compute norm for A
                        normB = OM_compute_norm(alphaB, lB, mB, nB)
                        S[a,b] += dA * dB * normA * normB *\
                            np.exp(-(alphaA*alphaB)/(alphaA + alphaB) * disAB**2) *\
                            OM_compute_Si(lA, lB, rPA[0], rPB[0], gamma) *\
                            OM_compute_Si(mA, mB, rPA[1], rPB[1], gamma) *\
                            OM_compute_Si(nA, nB, rPA[2], rPB[2], gamma)

    return(S)


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



trialbasis, K = basis.build_sto3Gbasis(Z, R)
print(OM_build_S(trialbasis, K))


sys.exit()
der1 = grad(CM_index)
der2 = grad(CM_index, 1)
der3 = grad(CM_index, 2)


for i in range(2):
    for j in range(2):
        print("derivative by Z, matrix field (%i,%i)" % (i, j))
        print(der1(Z, R, i, j))
        print("derivative by R, matrix field (%i,%i)" % (i, j))
        print(der2(Z, R, i, j))
        
