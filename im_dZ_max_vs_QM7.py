import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import database_preparation as datprep

#Data for plotting
'''
was computed using wolframalpha.com
{{1.4a^(1.4),b/p,c/q, d/r, e/s},{b/p,0,0,0,0},{c/q,0,0,0,0}, {d/r, 0, 0, 0,0},{e/s,0,0,0,0}}
'''

dim = np.arange(1,24)
zeros_in_evofder = np.asarray([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
size_of_matrix = dim**2

zeros_in_der = 2/dim - 1/(dim**2)
CM_diag = dim*(dim + 1)/2
CM_triag = dim*(dim-1)/2




fraction_of_nonzero_direct = np.ones(dim.size)-(zeros_in_der / size_of_matrix)
fraction_of_nonzero_EV = np.ones(dim.size)-(zeros_in_evofder / dim)

print(size_of_matrix)
print("fraction nonzero in matrix:")
print(fraction_of_nonzero_direct)
print(fraction_of_nonzero_EV)


#get exact results from CM analytical derivatives
results_file_CM = "/home/linux-miriam/Databases/Pickled/qm7_CM_results.pickle"

results_CM = datprep.read_compounds(results_file_CM)

dZ_frac_CM = []
norms = []

for i in range(len(results_CM)):
    norms.append(results_CM[i].Z.size)

    all_fractions = results_CM[i].calculate_smallerthan(0)
    dZ_frac_CM.append(results_CM[i].dZ_perc)


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

#add scatter plot of QM7 data
ax.scatter(np.asarray(norms), np.asarray(dZ_frac_CM), label = "Analytical Derivatives of QM7 Dataset")
#add line plot of maximum possible values
ax.plot(dim, fraction_of_nonzero_EV, label = "Maximum Fraction of Nonzero Eigenvalues of dZ Matrix")
ax.plot(dim, fraction_of_nonzero_direct, label = "Maximum Fraction of Nonzero dZ Matrix fields")

plt.xlabel("Number of Atoms in Molecule")
plt.ylabel("Fraction of Nonzero Derivatives") 

ax.legend()

plt.title("Coulomb Matrix Representation")
fig.tight_layout()

plt.savefig("dZ_derivatives_CM_byatomno", transparent = True, bbox_inches = 'tight')


