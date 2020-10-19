import numpy as np

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


def factorial2(n):
        if n <= 0:
            return 1
        else:
            return n * factorial2(n - 2)

def OM_compute_norm(alpha, l, m, n):
    '''compute normalization constant for overlap matrix'''
    N = (4*alpha)**(l+m+n)
    N /= factorial2(2*l-1)*factorial2(2*m-1)*factorial2(2*n-1)
    N *= ((2*alpha)/np.pi)**(1.5)
    N = N**(0.5)
    return(N)

def OM_compute_Si(qA, qB, gamma, rAi, rBi, rPi):
    '''Variables
    ---------
    qA : integer
          quantum number (l for Sx, m for Sy, n for Sz) for atom A
    qB : integer
          same for atom B
    rP : array 3D
          center between A and B
    rAi : float
          Cartesian coordinate of dimension q for atom A (rA[0] for Sx e.g.)
    rBi : float
             
    Returns
    -------
    si
    '''
    si = 0.0

    for k in range(int((qA + qB)/2)+1): #loop over only even numbers for sum(qA, qB)
        c = ck(qA, qB, rPi-rAi, rPi - rBi, k*2) 
        si += c *factorial2(2*k-1) / (2*gamma)**k
        si *= np.sqrt(np.pi/gamma)

    return(si)

def OM_compute_Sxyz(rA, rB, alphaA, alphaB, lA, lB, mA, mB, nA, nB):
    rP = OM_Gauss_Product(rA, rB, alphaA, alphaB)
    
    sx = OM_compute_Si(lA, lB, alphaA + alphaB, rA[0], rB[0], rP[0])
    sy = OM_compute_Si(mA, mB, alphaA + alphaB, rA[1], rB[1], rP[1])
    sz = OM_compute_Si(nA, nB, alphaA + alphaB, rA[2], rB[2], rP[2])

    return(sx*sy*sz)

def IJsq(rI, rJ):
    return sum( (rI[i]-rJ[i])**2 for i in range(3))

def OM_Gauss_Product(rA, rB, alphaA, alphaB):
    gamma = alphaA + alphaB
    P = []
    for i in range(3):
        P.append((alphaA*rA[i] + alphaB * rB[i])/gamma)
    
    return(P)


def BoB_fill(sorted_bag, desired_length):
    missing_zeros = desired_length - len(sorted_bag)
    padded_bag = np.pad(sorted_bag, (0,missing_zeros), 'constant')
    return (padded_bag)


def normed(vector, maximum):
    v_max = np.amax(vector)
    normed_vector = vector / v_max 
    return(normed_vector)

def calculate_mean(list_nparrays):
    'calculates average of list of arrays'
    n = len(list_nparrays)
    added = sum(list_nparrays)

    #calculate average value
    average = np.true_divide(added, n)
    #calculate mean squared error
    sqrd_errors = [(nparray - average)**2 for nparray in list_nparrays]
    error = np.true_divide(sum(sqrd_errors), n)
   
    return(average, error)
    
