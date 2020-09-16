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

    N = (2*alpha/np.pi)**(3/2)*(4*alpha)**(l+m+n) \
            /(factorial2(2*l-1)*factorial2(2*m-1)*factorial2(2*n-1))
    N = N**(1/2)
    return(N)

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

    Sq = 0

    for k in range(int((qA + qB)/2)+1): #loop over only even numbers for sum(qA, qB)
        c = ck(qA, qB, rPAq, rPBq, k)
        Sq += c * (np.pi/ gamma)**(1/2) ** factorial2(2*k-1)/((2*gamma)**k)
    return(Sq)

