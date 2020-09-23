'''in this program the off diagonal elements of OM and CM are calculated and plotted.
This is done primarily for an application on diatomics to see the decrease
for larger radii'''
import os
import jax_representation as jrep
import matplotlib.pyplot as plt
import qml
import itertools

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


'''plot that stuff'''
f, ax = plt.subplots()
#ax.plot(new_x, OM_y, label = 'Overlap Matrix')
ax.plot(new_x, CM_y, label = 'Coulomb Matrix')

every_nth = 4
for n, label in enumerate(ax.xaxis.get_ticklabels()):
    if n % every_nth != 0:
        label.set_visible(False)

ax.set_xlabel('Distance [$\AA$]')
ax.set_ylabel('Overlap')
ax.set_title('Cross interactions of HCl')
ax.legend()
plt.show()
print(OM_overlap_vector)
print(CM_overlap_vector)

