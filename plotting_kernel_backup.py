import numpy as np
import kernel_learning as kler
import matplotlib.pyplot as plt

datapath = "/home/stuke/Databases/QM9_XYZ_below10/"

class CurveObj:
    def __init__(self, name):
        self.xlist = []
        self.ylist = []
        self.xerror = []
        self.yerror = []
        self.name = name

def cleanup_results(resultsfile, multiple_runs = False):
    ''' gets data from resultfile and returns plottable Curve objects
    Variables
    ---------
    resultsfile : string, path to file containing pickled Result objects
    multiple_runs : if True, calculate mean of runs with same lamda and sigma
    
    Returns
    -------
    this_curve : LearningResults object
    '''

    plottable_curves = []
    
    if multiple_runs:
        lamdas = []
        sigmas = []
        xlists = []
        ylists = []


    results_list = kler.get_all_Results(resultsfile)
    
    for result in results_list:
        lamda = result.lamda
        sigma = result.sigma
        xlist = result.set_sizes
        ylist = result.maes
    
        if not multiple_runs:
            name = curve_name(sigma, lamda)
            curve = CurveObj(name)
            curve.xlist = xlist
            curve.ylist = ylist

            plottable_curves.append(curve)
            


    return(plottable_curves)

def curve_name(sigma, lamda):
    name = 'sigma = %s, lambda = %s'% (str(sigma), str(lamda))
    return(name)

def plot_learning(set_sizes = [10, 20, 40, 80, 160, 300], maes = [2.035934924620435, 1.8125458722942258, 1.7146791116661697, 1.6779313630086368, 1.8454600048725978, 1.8763117260488518], perc_maes = [232.7964404444149, 205.87841291639506, 190.275572697472, 162.50375206243325, 254.048604095239, 64.66622415061042], title = "sigma = 20\nlambda = 1e-3"):

    ax[0].loglog(set_sizes, maes, label = title, linewidth = fontsize/8)
    ax[1].loglog(set_sizes, perc_maes, label = title, linewidth = fontsize/8)
    return()

def plot_curves(curve_list):
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
    
    
    f, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (12, 8))
    st = f.suptitle('Learning Curves of CM Eigenvector Repro on QM9 Dataset\n of Molecules with 9 Atoms or less')
    # shift subplots down and to the left to give title and legend space:
    st.set_y(0.95)
    f.subplots_adjust(top=0.8, left = 0.05, right = 0.78, wspace = 0.1)

    ax[0].set_xlabel('Training Set Size')
    ax[1].set_xlabel('Training Set Size')
    ax[0].set_ylabel('MAE')
    ax[1].set_ylabel('percentile MAE')

    
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
    return()
