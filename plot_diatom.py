'''in this program the off diagonal elements of OM and CM are calculated and plotted.
This is done primarily for an application on diatomics to see the decrease
for larger radii'''
import os
import jax_representation as jrep
import qml

#path to xyz files
database = "/home/stuke/Databases/XYZ_diatom/"

for xyzfile in os.listdir(database):
    xyz_fullpath = database + xyzfile
    compound = qml.Compound(xyz_fullpath)
    
    print('file:', xyzfile)
    Z = compound.nuclear_charges.astype(float)
    R = compound.coordinates
    N = float(len(Z))
    print('Z', Z, 'R', R, 'N', N)
    OM = jrep.OM_full_sorted(Z, R, N)
    print(OM)
