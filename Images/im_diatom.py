'''in this program the off diagonal elements of OM and CM are calculated and plotted.
This is done primarily for an application on diatomics to see the decrease
for larger radii'''
import os
import jax_representation as jrep
import matplotlib.pyplot as plt
import qml
import itertools
#import jax_math as jmath
import numpy as np

#path to xyz files
database = "/home/stuke/Databases/XYZ_diatom/"
distance_vector = []
OM_overlap_vector = []
CM_overlap_vector = []
for xyzfile in os.listdir(database):
    if xyzfile.endswith(".xyz"):
        xyz_fullpath = database + xyzfile
        compound = qml.Compound(xyz_fullpath)
        distance_vector.append(xyzfile[10:13]) #distance is given in 'name...i.xyz', retrieve i here
        print('file:', xyzfile, 'distance:', xyzfile[10:13])
        Z = compound.nuclear_charges.astype(float)
        R = compound.coordinates
        N = float(len(Z))
    
        #Calculate Overlap matrix and determine dimensionality dim
        OM, order = jrep.OM_full_sorted(Z, R, N)
        CM, order = jrep.CM_full_sorted(Z, R, N)
        dim  = len(order)
    
        #loop over OM and add all off-diagonal elements
        OM_overlap = 0
        CM_overlap = 0
        for i in range(dim):
            for j in range(dim):
                if (i != j):
                    OM_overlap += OM[i][j]
                    CM_overlap += CM[i][j]
        OM_overlap_vector.append(float(OM_overlap))
        CM_overlap_vector.append(float(CM_overlap))


OM_lists = sorted(itertools.zip_longest(*[distance_vector, OM_overlap_vector]))
CM_lists = sorted(itertools.zip_longest(*[distance_vector, CM_overlap_vector]))

new_x, OM_y = list(itertools.zip_longest(*OM_lists))
new_x, CM_y = list(itertools.zip_longest(*CM_lists))

OM_y = jmath.normed(np.array(OM_y), 1)
CM_y = jmath.normed(np.array(CM_y), 1)
print(CM_y)
'''plot that stuff'''
import seaborn as sns
fontsize = 30

plt.rc('font',       size=fontsize) # controls default text sizes
plt.rc('axes',  titlesize=fontsize) # fontsize of the axes title
plt.rc('axes',  labelsize=fontsize) # fontsize of the x and y labels
plt.rc('xtick', labelsize=fontsize*0.8) # fontsize of the tick labels
plt.rc('ytick', labelsize=fontsize*0.8) # fontsize of the tick labels
plt.rc('legend', fontsize=fontsize*0.8) # legend fontsize
plt.rc('figure',titlesize=fontsize*1.2) # fontsize of the figure title
f, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 8))
#ax.plot(new_x, OM_y, label = 'Overlap Matrix')
ax.plot(new_x, CM_y, label = 'Coulomb Matrix', linewidth=fontsize/8)
ax.plot(new_x, OM_y, label = 'Overlap Matrix', linewidth=fontsize/8)

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
#ax.spines['bottom'].set_position(('axes', -0.05))
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
#ax.spines['left'].set_position(('axes', -0.05))
plt.rcParams["legend.loc"] = 'upper right'

ax.set_xlabel('Distance [$\AA$]')
ax.legend(frameon=False)
    
plt.setp(ax, xticks = [0, 15, 30], xticklabels = ['0.0',  '1.5',  '3.0'], yticks = [0, 0.3, 0.6, 0.9], yticklabels = ['0.0', '0.3', '0.6', '0.9'])
ax.set_ylabel('Relative Overlap [a.u.]')
f.suptitle('Cross interactions of HCl')

sns.set_style('whitegrid', {'grid.linestyle': '--'})
sns.set_context("poster")

f.savefig("HCl_overlap_OM_CM.png", bbox_inches = 'tight')
f.savefig("HCl_overlap_OM_CM.pdf", bbox_inches = 'tight')

