import os
import jax_derivative as jder
import qml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg
#path to xyz files
database = "/home/stuke/Databases/XYZ_ethin/"

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

def pandaseries_dR(eigenvalues, labels):
    listof_series = []
    for i in zip(eigenvalues, labels): #combines correct axis (x,y,z) with derivatives
        for j in zip(i[0],i[1]): #combines correct atom (1, 2, 3...) with derivative
            s = pd.Series(j[0], name = j[1])  #creates panda series with index being range(len(j[0]))
            listof_series.append(s)
    return(listof_series)

def pandaseries_dZ(eigenvalues, labels):
    listof_series = []
    for i in zip(eigenvalues, labels):
        s = pd.Series(j[0], name = j[1])
    
    return(listof_series)

def pandaseries_dN(eigenvalues, labels):
    s = pd.Series(eigenvalues, labels)
    return(s)

name_vector = [] #some kind of identificator for xyz file. here: name identifying fraction of angle of torsion
dZ_eigenvalues = [] #list of eigenvalue vectors. length is same as len of name_vector



x = 9*np.arange(9)
print('x ', x)
dimZ =4 
fontsize = 20


label_dN = ['dN']
label_dZ = ['dZ%i' % (i + 1) for i in range(dimZ)]
otherlabel_dR = [['dx%i' %(i+1) , 'dy%i' %(i+1) , 'dz%i' %(i+1)]  for i in range(dimZ)]
label_dR = [['dx%i' % (i + 1) for i in range(dimZ)], ['dy%i' %(j + 1) for j in range(dimZ)], ['dz%i' %(k+1) for k in range(dimZ)]]

print(label_dN, '\n', label_dZ, '\n', label_dR)
print('other R', otherlabel_dR)

#prepare plotting frame
def plot_dR(listof_series, listof_names):
    '''plots window over all dRs that orders them as follws:

        dx1 dy1 dz1
        dx2 dy2 dz2
         :   :   : 

        Variables
        ---------
        listof_series : just that, a list containing pandas series ready to be plotted.
                        every series should also have a name for titles of the subplots
        listof_names :  list of names that correspond to each series. This is aimed at screening
                        multiple xyz files for example.

        plots dR plots. no return so far

    '''


    #create plot
    plt.rc('font',       size=fontsize) # controls default text sizes
    plt.rc('axes',  titlesize=fontsize) # fontsize of the axes title
    plt.rc('axes',  labelsize=fontsize) # fontsize of the x and y labels
    plt.rc('xtick', labelsize=fontsize*0.8) # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontsize*0.8) # fontsize of the tick labels
    plt.rc('legend', fontsize=fontsize*0.8) # legend fontsize
    plt.rc('figure',titlesize=fontsize*1.2) # fontsize of the figure title

    
    figR, axR = plt.subplots(nrows = dimZ, ncols = 3, figsize=(12,8))
    figR.tight_layout()
    st = figR.suptitle("First Derivative of Ethin with Rotating H")

    # shift subplots down and to the left to give title and legend space:
    st.set_y(0.95)
    figR.subplots_adjust(top=0.85, left = 0.05, right = 0.88, wspace = 0.1)
    
    #Add image of rotating ethin
    beautiful = mpimg.imread('Ethin_Rotation.png')
    # Place the image in the upper-right corner of the figure
    #--------------------------------------------------------
    # We're specifying the position and size in _figure_ coordinates, so the image
    # will shrink/grow as the figure is resized. Remove "zorder=-1" to place the
    # image in front of the axes.
    newax = figR.add_axes([0.8, 0.8, 0.2, 0.2], anchor='NE', zorder=-1)
    newax.imshow(beautiful)
    newax.axis('off')
    

    print('dim of series info is ', len(listof_series))
    lines = [] #stores info for legend
    #loop over all series that were passed and plot to correct window
    for idx in range(len(listof_series)):
        series = listof_series[idx]

        for i in range(dimZ): #chooses x, y, z aka ncols
            for j in range(3): #chooses 1, 2, 3... aka nrows
                print('i = ', i, 'j = ', j,'series[i + j]', series[i + j]) 
                
                newline = axR[i][j].plot(series[i*3 + j], label = int(float(listof_names[idx])))[0]
                lines.append(newline)
    #set title and show legend for all subplots
    for i in range(dimZ):
        for j in range(3):
            axR[i][j].set_title(series[i*3+j].name)
    
    handles, labels = axR[0][0].get_legend_handles_labels()
    #for numeric labels, they can be sorted by the following two lines of code
    ascending = sorted(zip(listof_names, handles, labels))
    sorted_listof_names, sorted_handles, sorted_labels = zip(*ascending)
    print(sorted_handles, 'unsorted handles:', handles)

    figR.legend(sorted_handles, sorted_labels, loc="center right", title = 'x for phi = 90/x')
    plt.show()
    return()

listof_series_dR = []

for xyzfile in os.listdir(database):
    if xyzfile.endswith(".xyz"):
        xyz_fullpath = database + xyzfile
        compound = qml.Compound(xyz_fullpath)
        Z = compound.nuclear_charges.astype(float)
        name_vector.append(xyzfile[6:9]) #distance is given in 'name...i.xyz', retrieve i here
        R = compound.coordinates
        N = float(len(Z))
        dimZ = len(Z)
        print('dimZ is ', dimZ)
        #Calculate CM derivative matrix and determine eigenvalues thereof. Store accordingly.
        
        dZ = jder.sort_derivative('CM', Z, R, N, grad = 1, dx = "R")
        eigenvalues, eigenvectors = np.linalg.eig(dZ)

        dZ_eigenvalues.append(eigenvalues)
        
        #prepare for plotting
        series = pandaseries_dR(eigenvalues, otherlabel_dR)
        listof_series_dR.append(series)
        
plot_dR(listof_series_dR, name_vector)
        


    
sys.exit()



print(dZ_eigenvalues)    
CM_dxi = np.array([i[0][0] for i in dZ_eigenvalues])
print(CM_dxi)




'''
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
for j in range(4):
    y = [k[j] for k in CM_dxi]
    ax.scatter(x, y, label = 'Coulomb Matrix', linewidth=fontsize/8)

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
f.show()
f.savefig("ethin_derivative_CM.png", bbox_inches = 'tight')
f.savefig("ethin_derivative_CM.pdf", bbox_inches = 'tight')

'''

