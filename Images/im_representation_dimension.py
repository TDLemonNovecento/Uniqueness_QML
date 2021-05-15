import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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


#Data for plotting
N = np.arange(1,23)
relevant_dim = 3*N - 6
BoB = N*(N + 1)/2
EVOM_upper = 5*N
EVOM_lower = np.append(N[:16],N[16:]*5)
OM_upper = EVOM_upper**2
OM_lower = EVOM_lower **2
CM = N**2
CM_triag = N*(N-1)/2

all_CCS = 4*N-6

print("BoB:", BoB)
print("CM:", CM)
print("EVOM lower:", EVOM_lower)
print("OM_upper", OM_upper)
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
ax.plot(N, all_CCS, "x", label = "CCS (4n+1)")
ax.plot(N, relevant_dim,"x", label = "Internal D.o.F. (3n - 6)")
ax.plot(N, CM, 'o', label = "CM")
ax.plot(N, CM_triag, '>', label = "CM triag")
ax.plot(N, N, 'o', label = "EVCM")
ax.plot(N, BoB, 'p', label = "BOB, CM diag")
ax.plot(N, OM_upper, 'v', label = "OM upper limit")
ax.plot(N, OM_lower,'<',  label = "OM lower limit")
ax.plot(N, EVOM_upper,'v',  label = "EVOM upper limit")
ax.plot(N, EVOM_lower,'P', label = "EVOM lower limit")

plt.xlabel("Number of Atoms in Molecule")
plt.ylabel("Degrees of Freedom") 
plt.yscale('log')

ax.legend()

#plt.title("Dimensions of Representations")
fig.tight_layout()

plt.savefig("./Representation_dimensions", transparent = True, bbox_inches = 'tight')


