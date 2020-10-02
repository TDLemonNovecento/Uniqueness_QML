import os, time
import jax_derivative as jder
import qml
import numpy as np
import derivative_plot as derp

#path to xyz files
database = "/home/stuke/Databases/XYZ_ethin/"
ethin_image = "./ethin.svg"

#the ethin files are sorted by index. the numbering shows how big angle theta is:
'''we start with a straight molecule stretching out on the x axis:

      H--C===C--H

to then move both H simultaneously anti-clockwise by an angle phi:
    
                H
               / phi
         C===C.......
        /
        H

'''
name_list = [] #identificator for xyz file. here: name identifying fraction of angle of torsion
dZ_eigenvalues = [] #list of eigenvalue vectors. length is same as len of name_vector
dimZ_list = [] #dimension of files may vary. store all dimensions here


listof_series_dR = []

for xyzfile in os.listdir(database):
    if xyzfile.endswith(".xyz"):
        xyz_fullpath = database + xyzfile
        compound = qml.Compound(xyz_fullpath)
        Z = compound.nuclear_charges.astype(float)
        name_list.append(xyzfile[6:9]) #distance is given in 'name...i.xyz', retrieve i here
        R = compound.coordinates
        N = float(len(Z))
        dimZ = len(Z)
        dimZ_list.append(dimZ)
        
        #Calculate CM derivative matrix and determine eigenvalues thereof. Store accordingly.
        dZ = jder.sort_derivative('CM', Z, R, N, grad = 1, dx = "R")
        eigenvalues, eigenvectors = np.linalg.eig(dZ)

        dZ_eigenvalues.append(eigenvalues)
        
        #prepare for plotting
        series = derp.pandaseries_dR(eigenvalues, dimZ)
        listof_series_dR.append(series)
       

figname = derp.plot_dR(listof_series_dR, name_list, max(dimZ_list))
time.sleep(5)
#derp.merge_plot_with_svg(figname, ethin_image)
