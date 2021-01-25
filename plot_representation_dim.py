import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from math import factorial
import database_preparation as datprep

'''
standard size for BoB:
    {'C': 7, 'H': 16, 'N': 3, 'O': 3, 'S': 1}
    #number of bags has no effect on dimensionality?
size of OM for max and min number of orbitals.
# of orbitals:
    H, He: 1
    Li, Be, B, C, N, O, F, Ne: 5
    Na, Mg, Al, Si, P, S, Cl, Ar: 9
    K, Ca: 13
'''

def nchoose2(vec):
    V = []
    for i in vec:
        nfac = factorial(i)
        try:
            kfac = 2* factorial(i-2)
        except ValueError:
            nfac = 1
            kfac = 1
        V.append(nfac/kfac)
    return(np.asarray(V))

#Data for plotting
N = np.arange(1,23)
relevant_dim = 3*N - 6
BoB = N + nchoose2(N)
OM = (2*N)**2
CM = N**2
all_CCS = 4*N-6

print("BoB:", BoB)
print("CM:", CM)
print("freedom CCS degrees", all_CCS)
print(relevant_dim)


#prepare canvas for plots
'''standard settings for matplotlib plots'''
fontsize = 24
plt.rc('font',       size=fontsize) # controls default text sizes
plt.rc('axes',  titlesize=fontsize) # fontsize of the axes title
plt.rc('axes',  labelsize=fontsize) # fontsize of the x and y labels
plt.rc('xtick', labelsize=fontsize*0.8) # fontsize of the tick labels
plt.rc('ytick', labelsize=fontsize*0.8) # fontsize of the tick labels
plt.rc('legend', fontsize=fontsize*0.8) # legend fontsize
plt.rc('figure',titlesize=fontsize*1.2) # fontsize of the figure title


fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(12,8))

#add line plot of maximum possible values
ax.plot(N, all_CCS, label = "dimensions of CCS")
ax.plot(N, relevant_dim, label = "internal degrees of freedom")
ax.plot(N, CM, label = "CM")
ax.plot(N, N, label = "EVCM")
ax.plot(N, BoB, label = "BoB")
ax.plot(N, OM, label = "minimum for OM")

plt.xlabel("Number of Atoms in Molecule")
plt.ylabel("Degrees of Freedom") 

ax.legend()

plt.title("Exponential Growth of Fingerprint Dimensions")
fig.tight_layout()

plt.savefig("dZ_derivatives_CM_byatomno", transparent = True, bbox_inches = 'tight')


