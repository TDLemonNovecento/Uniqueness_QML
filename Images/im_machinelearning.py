import sys
sys.path.insert(0, "..")


import matplotlib.pyplot as plt
import numpy as np


#standard settings for plotting:
fontsize = 30

plt.rc('font',       size=fontsize) # controls default text sizes
plt.rc('axes',  titlesize=fontsize) # fontsize of the axes title
plt.rc('axes',  labelsize=fontsize) # fontsize of the x and y labels
plt.rc('xtick', labelsize=fontsize*0.8) # fontsize of the tick labels
plt.rc('ytick', labelsize=fontsize*0.8) # fontsize of the tick labels
plt.rc('legend', fontsize=fontsize*0.8) # legend fontsize
plt.rc('figure',titlesize=fontsize*1.2) # fontsize of the figure title
plt.rcParams['axes.titlepad'] = 20




fig = plt.figure(figsize = (12, 8))

ax = fig.gca(projection='3d')
#ax1 = fig.add_subplot(1, 2, 2, projection = '3d')
#ax2 = fig.add_subplot(1, 2, 1)


#plot surface
# create x,y
xx,yy=np.meshgrid([0,1], [0,1])
 
zz = 0.5*xx + 0.5
ax.plot_surface(xx, yy, zz, alpha = 0.5)



#define all 3 axes:
x = np.linspace(0, 1, 100)
y = 0.5*x  + 0.5 
z = np.random.uniform(0, 1, size = 100)

ax.plot(x, z, zdir='z', label='Curve in (X,Y)')

ax.scatter(x,z,y, zdir ='z', label = 'Points in (X,Y,Z)')




# Make legend, set axes limits and labels
ax.legend()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

plt.savefig("ML_figure.png", bbox_inches="tight",
            pad_inches=0.3, transparent=True)
