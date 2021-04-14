import os, time
import jax_derivative as jder
import qml
import numpy as np
import plot_derivative as pltder
import jax_representation as jrep
import numerical_derivative as numder
from matplotlib import colors as mcolors


#path to xyz files
database_opt = "/home/linux-miriam/Databases/Ethyne_invariant/Optimized/"
ethin_image = "/home/linux-miriam/Databases/Ethyne_invariant/ehtyne.png"
database_raw = "/home/linux-miriam/Databases/Ethyne_invariant/Raw"

'''define folder of .xyz files'''
datapath = database_opt
reference = database_raw + "/ethin_00.xyz"

indexes = range(20)


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

#calculate reference values
compound = qml.Compound(reference)
Z = compound.nuclear_charges.astype(float)
R = compound.coordinates

ref_M_EVCM = jrep.CM_ev_unsrt(Z, R, size = 4)




dx_slot1_list = [[],[],[],[]]
dx_slot2_list = [[],[],[],[]]
dx_slot3_list = [[],[],[],[]]
dx_slot4_list = [[],[],[],[]]

dy_slot1_list = [[],[],[],[]]
dy_slot2_list = [[],[],[],[]]
dy_slot3_list = [[],[],[],[]]
dy_slot4_list = [[],[],[],[]]

dz_slot1_list = [[],[],[],[]]
dz_slot2_list = [[],[],[],[]]
dz_slot3_list = [[],[],[],[]]
dz_slot4_list = [[],[],[],[]]



EVCM_aberration = []

for i in indexes:
    num = str(i).zfill(2)
    filename = datapath + "ethin_" + num + ".xyz"
    compound = qml.Compound(filename)
    Z = compound.nuclear_charges.astype(float)
    R = compound.coordinates
    
    #calculate difference to reference constitution
    M_EVCM = jrep.CM_ev_unsrt(Z, R, N = 0, size = 4)

    diff = (ref_M_EVCM - M_EVCM)
    error = 0
    for d in diff:
        error += d*d

    EVCM_aberration.append(error)


    #calculate CM
    M_CM = jrep.CM_full_unsorted_matrix(Z, R, N=0, size = 4)
    
    #collect all dZ for plotting
    for j in range(4):
        dxj = numder.derivative(jrep.CM_ev_unsrt,[Z, R, 0], order = 1, d1 = [1,j, 0])
        dx_slot1_list[j].append(dxj[0])
        dx_slot2_list[j].append(dxj[1])
        dx_slot3_list[j].append(dxj[2])
        dx_slot4_list[j].append(dxj[3])

        dyj = numder.derivative(jrep.CM_ev_unsrt,[Z, R, 0], order = 1, d1 = [1,j, 1])
        dy_slot1_list[j].append(dyj[0])
        dy_slot2_list[j].append(dyj[1])
        dy_slot3_list[j].append(dyj[2])
        dy_slot4_list[j].append(dyj[3])
            
        dzj = numder.derivative(jrep.CM_ev_unsrt,[Z, R, 0], order = 1, d1 = [1,j, 2])
        dz_slot1_list[j].append(dzj[0])
        dz_slot2_list[j].append(dzj[1])
        dz_slot3_list[j].append(dzj[2])
        dz_slot4_list[j].append(dzj[3])

#dZ1 is dZ[0] and so forth, it contains a list of arrays in 4 dimensions
#for every atom, extract it's data


#prepare list of lists for plotting
yvalues = [[],[],[]]


#define colors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

# Sort colors by hue, saturation, value and name.
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                for name, color in colors.items())
sorted_names = [name for hsv, name in by_hsv]

dx_cor = 130 #start with reds

dy_cor = 68 #start with greens

dz_cor = 91 #start with blues

for i in range(4):
    dwhich = "dx"

    yvalues[0].append([np.asarray(dx_slot1_list[i]), dwhich + "%i D1" %(i + 1), colors[sorted_names[dx_cor]]])
    yvalues[0].append([np.asarray(dx_slot2_list[i]), dwhich + "%i D2" %(i + 1), colors[sorted_names[dx_cor+1]]])
    
    dwhich = "dy"

    yvalues[1].append([np.asarray(dy_slot1_list[i]), dwhich + "%i D1" %(i + 1), colors[sorted_names[dy_cor]]])
    yvalues[1].append([np.asarray(dy_slot2_list[i]), dwhich + "%i D2" %(i + 1), colors[sorted_names[dy_cor+1]]])


    dx_cor += 2
    dy_cor += 2


yvalues[2].append([np.asarray(dz_slot1_list[1]), "dz D1, dz D2", colors[sorted_names[dz_cor]]])
yvalues[2].append([[0], "D3, D4"])


#change color of one line to red for visibility
yvalues[1][4][2] = 'r'

#add error of EVCM list compared to reference to lineplots
linevalues = [[np.asarray(EVCM_aberration), "MSE to Reference"]]

finalyvalues = []
for ylist in yvalues:
    finalyvalues.extend(ylist)

pltder.plot_ethyne(indexes, finalyvalues,\
        savetofile = "./Images/Final/dR_Ethyne.png",\
        #lineplots = linevalues,\
        plot_title = False,\
        plot_dZ = False)
