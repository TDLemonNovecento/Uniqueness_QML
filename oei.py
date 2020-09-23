import math
import numpy as np
from jax_math import factorial2
from scipy import special


def buildS(basis, S):
  """
  Pass in an empty KxK overlap integral matrix, S.
  Calling the function (Sxyz) to calculate each Gaussian primitive overlap integral,
  compute all elements of the S = (A|B) 2-dimensional matrix.
  (Handout 4, Eq. 1)
  """
  for A, bA in enumerate(basis):    # retrieve atomic orbital A from the basis
    for B, bB in enumerate(basis):  # retrieve atomic orbital B from the basis

      for a, dA in zip(bA['a'],bA['d']):    # retrieve A.O. A alpha and contraction coefficients
        for b, dB in zip(bB['a'],bB['d']):  # retrieve A.O. B alpha and contraction coefficients

          RA, RB = bA['r'], bB['r']  # 

          lA, mA, nA = bA['l'], bA['m'], bA['n'] # instantiate angular momentum exponent variables, A.O. A
          lB, mB, nB = bB['l'], bB['m'], bB['n'] # instantiate angular momentum exponent variables, A.O. B
            
          # using all of the collected information on A.O.s A and B, calculate Handout 4, Eq. 1
          S[A,B] += (
                     math.exp(-a*b*IJsq(RA,RB)/(a+b)) * #
                     N(a,lA,mA,nA) * N(b,lB,mB,nB) * # normalization constant
                     dA * dB * Sxyz(RA,RB,a,b,lA,lB,mA,mB,nA,nB)
                    )

  return S

def Sxyz(RA,RB,a,b,lA,lB,mA,mB,nA,nB):
  """
  Calculate the Gaussian primitive overlap integral.
  (Handout 4, Eq. 2)
  """
  RP = gaussianProduct(a,RA,b,RB,a+b) # g = a + b

  sx = si(lA,lB,a+b,RA[0],RB[0],RP[0])
  sy = si(mA,mB,a+b,RA[1],RB[1],RP[1])
  sz = si(nA,nB,a+b,RA[2],RB[2],RP[2])

  return sx*sy*sz

def si(lA,lB,g,Ai,Bi,Pi):
  """
  Calculate the i-th component (x, y, or z) contribution to (A|B).
  (Handout 4, Eq. 3)
  """
  si = 0.0
  for j in range(0,int((lA+lB)/2)+1):
    si += ck(2*j,lA,lB,Pi-Ai,Pi-Bi) * factorial2(2*j-1) / (2*g)**j
  si *= math.sqrt(math.pi/g)

  return si 

def buildT(basis, T):
  """
  Build the kinetic energy integral matrix. 
  Call on function Kxyz to calculate integrals over primitives.
  (part of Handout 4, Eq. 10)
  """
  for A, bA in enumerate(basis):
    for B, bB in enumerate(basis):

      for a, dA in zip(bA['a'],bA['d']):
        for b, dB in zip(bB['a'],bB['d']):

          RA, RB = bA['R'], bB['R']

          lA, mA, nA = bA['l'], bA['m'], bA['n']
          lB, mB, nB = bB['l'], bB['m'], bB['n']

          T[A,B] += (
                     math.exp(-a*b*IJsq(RA,RB)/(a+b)) * 
                     N(a,lA,mA,nA) * N(b,lB,mB,nB) * 
                     dA * dB 
                    ) * Kxyz(RA,RB,a,b,lA,lB,mA,mB,nA,nB)

  return T

def Kxyz(RA,RB,a,b,lA,lB,mA,mB,nA,nB):
  """
  Calculate the kinetic energy integral over primitives, for components x, y, z.
  (the "integral" part of Handout 4, Eq. 10)
  """
  K  = b*(2*(lB+mB+nB)+3)*Sxyz(RA,RB,a,b,lA,lB,mA,mB,nA,nB) # line 1 of Eq. 10

  K -= (2*b**2)*Sxyz(RA,RB,a,b,lA,lB+2,mA,mB,nA,nB) # line 2
  K -= (2*b**2)*Sxyz(RA,RB,a,b,lA,lB,mA,mB+2,nA,nB)
  K -= (2*b**2)*Sxyz(RA,RB,a,b,lA,lB,mA,mB,nA,nB+2)

  K -= (1/2)*(lB*(lB-1))*Sxyz(RA,RB,a,b,lA,lB-2,mA,mB,nA,nB) # line 3
  K -= (1/2)*(mB*(mB-1))*Sxyz(RA,RB,a,b,lA,lB,mA,mB-2,nA,nB)
  K -= (1/2)*(nB*(nB-1))*Sxyz(RA,RB,a,b,lA,lB,mA,mB,nA,nB-2)

  return K

def buildV(basis, V, R, Z, atoms):
  """
  Calculate the product of the x, y, and z components of the nuclear-attraction 
  integral over the Gaussian primitives.
  (part of Handout 4, Eq. 13)
  """  
  for A, bA in enumerate(basis):
    for B, bB in enumerate(basis):
      for C, rC in enumerate(R):

        for a, dA in zip(bA['a'],bA['d']):
          for b, dB in zip(bB['a'],bB['d']):

            RA, RB, RC = bA['R'], bB['R'], rC

            lA, mA, nA = bA['l'], bA['m'], bA['n']
            lB, mB, nB = bB['l'], bB['m'], bB['n']

            V[C,A,B] += dA*dB*Vxyz(lA,mA,nA,lB,mB,nB,a,b,RA,RB,RC,Z[atoms[C]])

  return V

def Vxyz(lA,mA,nA,lB,mB,nB,a,b,RA,RB,RC,Z):
  """
  Calculate the product of the x, y, and z components of the nuclear-attraction 
  integral over the Gaussian primitives.
  (the "integral" part of Handout 4, Eq. 13)
  """
  g = a + b

  RP = gaussianProduct(a,RA,b,RB,g)

  ABsq = IJsq(RA,RB)
  PCsq = IJsq(RP,RC)
  Vxyz = 0.0
  for l in range(0,lA+lB+1):
    for r in range(0,int(l/2)+1):
      for i in range(0,int((l-2*r)/2)+1):
        vx = vi(l,r,i, lA,lB,RA[0],RB[0],RC[0],RP[0],g)

        for m in range(0,mA+mB+1):
          for s in range(0,int(m/2)+1):
            for j in range(0,int((m-2*s)/2)+1):
              vy = vi(m,s,j, mA,mB,RA[1],RB[1],RC[1],RP[1],g)

              for n in range(0,nA+nB+1):
                for t in range(0,int(n/2)+1):
                  for k in range(0,int((n-2*t)/2)+1):
                    vz = vi(n,t,k, nA,nB,RA[2],RB[2],RC[2],RP[2],g)

                    nu  = l+m+n-2*(r+s+t)-(i+j+k)
                    F   = BoysFunction(nu, g*abs(PCsq))

                    Vxyz  += vx * vy * vz * F

  Na = N(a,lA,mA,nA) # calculate the normalization constant for orbital A
  Nb = N(b,lB,mB,nB) # calculate the normalization constant for orbital B

  Vxyz *= 2*math.pi/g
  Vxyz *= math.exp(-a*b*abs(ABsq)/g)
  Vxyz *= Na * Nb 
  Vxyz *= -Z
  return Vxyz

def vi(l,r,i, lA,lB,Ai,Bi,Ci,Pi,g):
  """
  Calculate the $i-th$ component of the nuclear-attraction integral over
  Gaussian primitives.
  (Handout 4, Eq. 14)
  """

  eps = 1/(4*g) # (Handout 4, Eq. 15)
  
  vi  = (-1)**l
  vi *= ck(l,lA,lB,Pi-Ai,Pi-Bi)
  vi *= (-1)**i * misc.factorial(l,exact=True)
  vi *= (Pi-Ci)**(l-2*r-2*i) * eps**(r+i)
  vi /= misc.factorial(r,exact=True)
  vi /= misc.factorial(i,exact=True)
  vi /= misc.factorial(l-2*r-2*i,exact=True)

  return vi
 
def BoysFunction(nu, x):
  """
  The analytical function coded herein was suggested by CCL forum; similar to
  result when evaluating the Boys Function integral in Mathematica. 
  Depends on gamma functions, which are easily computed with SciPy's 
  library of special functions.
  """
  if x < 1e-7:
    return (2*nu+1)**(-1) - x*(2*nu+3)**(-1) # (Handout 4, Eq. 17)
  else:
    return (1/2) * x**(-(nu+0.5)) * special.gamma(nu+0.5) * special.gammainc(nu+0.5,x) # (Handout 4, Eq. 16)
 
def ck(j,l,m,a,b):
  """
  Calculate the coefficient 'ck' factor within the theta expression,
  associated with a third center between position vectors
  of the nuclei A and B.
  (Handout 4, Eq. 8)
  """
  coefficient = 0.0
 
  for k in range(0,l+1):
    for i in range(0,m+1):
      if i + k == j:
        coefficient += special.binom(l,k) * special.binom(m,i) * a**(l-k) * b**(m-i)

  return coefficient

def N(a,l,m,n):
  """
  Calculate the normalization factors.
  (Handout 4, Eq. 9)
  """
  N  = (4*a)**(l+m+n)
  N /= factorial2(2*l-1) * factorial2(2*m-1) * factorial2(2*n-1)
  N *= ((2*a)/math.pi)**(1.5)
  N  = N**(0.5)

  return N

def gaussianProduct(a,RA,b,RB,g):
  """
  The product of two Gaussians is a third Gaussian.
  (Handout 4, Eq. 5)
  """
  P = []
  for i in range(3):
    P.append( (a*RA[i]+b*RB[i])/g )

  return P

def IJsq(RI,RJ):
  """
  Calculate the square of the distance between two points.
  (Handout 4, Eq. 6)
  """
  return sum( (RI[i]-RJ[i])**2 for i in (0,1,2) )
