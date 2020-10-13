import numpy as np
import kernel_learning as krnl
import matplotlib.pyplot as plt

datapath = "/home/stuke/Databases/QM9_XYZ_below10/"

class LearningResults(lamda, sigma, set_sizes, maes, perc_maes):
    def __init__(self):
        self.lamda = lamda
        self.sigma = sigma
        self.set_sizes = set_sizes
        self.maes = maes
        self.perc_maes = perc_maes

def my_learning_curves(folder, sigma = 20, set_sizes = [10, 20, 40, 80, 160, 300], lamda = 1.0e-12):
    maes = []
    percent_maes = []

    for size in set_sizes:
        training_e, test_e, result, errors = krnl.my_kernel_ridge(folder, size)
        percent_errors = np.divide(errors, test_e)*100
        percent_mae = sum(abs(percent_errors))/(len(errors))
        mae = sum(abs(errors))/(len(errors))
        maes.append(mae)
        percent_maes.append(percent_mae)
    

    print('set_sizes\n', set_sizes)
    print('maes\n', maes)
    print('percentile maes\n', percent_maes)
    return(set_sizes, maes, percent_maes, lamda)
    
#for plotting:
fontsize = 30

plt.rc('font',       size=fontsize) # controls default text sizes
plt.rc('axes',  titlesize=fontsize) # fontsize of the axes title
plt.rc('axes',  labelsize=fontsize) # fontsize of the x and y labels
plt.rc('xtick', labelsize=fontsize*0.8) # fontsize of the tick labels
plt.rc('ytick', labelsize=fontsize*0.8) # fontsize of the tick labels
plt.rc('legend', fontsize=fontsize*0.8) # legend fontsize
plt.rc('figure',titlesize=fontsize*1.2) # fontsize of the figure title
plt.rcParams['axes.titlepad'] = 20 


f, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (12, 8))
st = f.suptitle('Learning Curves of CM Eigenvector Repro on QM9 Dataset\n of Molecules with 9 Atoms or less')
# shift subplots down and to the left to give title and legend space:
st.set_y(0.95)
f.subplots_adjust(top=0.8, left = 0.05, right = 0.78, wspace = 0.1)

ax[0].set_xlabel('Training Set Size')
ax[1].set_xlabel('Training Set Size')
ax[0].set_ylabel('MAE')
ax[1].set_ylabel('percentile MAE')

def plot_learning(set_sizes = [10, 20, 40, 80, 160, 300], maes = [2.035934924620435, 1.8125458722942258, 1.7146791116661697, 1.6779313630086368, 1.8454600048725978, 1.8763117260488518], perc_maes = [232.7964404444149, 205.87841291639506, 190.275572697472, 162.50375206243325, 254.048604095239, 64.66622415061042], title = "sigma = 20\nlambda = 1e-3"):

    ax[0].loglog(set_sizes, maes, label = title, linewidth = fontsize/8)
    ax[1].loglog(set_sizes, perc_maes, label = title, linewidth = fontsize/8)
    
#all the plotting has to be done below
for l in [1.e-7,1.e-9,1.e-11,1.e-13]:
    sigma = 4
    set_sizes, maes, percent_maes, thislamda = my_learning_curves(datapath, sigma, [10, 20, 40, 80, 160, 300],l)
    learning_results = kernel_obj(l, sigma, set_sizes, maes, percent_maes)
    plot_learning(set_sizes, maes, percent_maes, title = "s = %s; l = %s" % (str(sigma), str(l)))



#all the plotting has to be done above

handles, labels = ax[0].get_legend_handles_labels()

f.legend(handles, labels, title = 'Variables:', loc = 'center right')

f.savefig('learning.png', bbox_inches = 'tight')
f.savefig('learning.pdf', bbox_inches = 'tight')
