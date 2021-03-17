import os, time
import jax_derivative as jder
import qml
import numpy as np
import plot_derivative as derp
import pandas as pd

#path to xyz files
database_opt = "/home/linux-miriam/Databases/Ethyne_invariant/Optimized/"
ethin_image = "/home/linux-miriam/Databases/Ethyne_invariant/ehtyne.png"
database_raw = "/home/linux-miriam/Databases/Ethyne_invariant/Raw"
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
listof_pandaseries = []

database = database_opt



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
        #series = derp.pandaseries_dR(eigenvalues, dimZ)
        

        #do pandas series for faster plotting
        iterables = [['d1', 'd2', 'd3', 'd4'], ['dx', 'dy', 'dz'], [1, 2, 3, 4]]
        index = pd.MultiIndex.from_product(iterables, names=['dAtom', 'dxyz', 'atoms'])

        pd_series = pd.DataFrame(eigenvalues.flatten(), index = index, columns = ['values'])
        
        listof_pandaseries.append(pd_series)
        #listof_series_dR.append(series)
       

figname = derp.plot_pandas_ethin(listof_pandaseries, name_list, max(dimZ_list))
#time.sleep(5)
print(figname)
#derp.merge_plot_with_svg(figname, ethin_image)
