import numpy as np
import kernel_learning as kler
import matplotlib.pyplot as plt
import jax_math as jmath
import database_preparation as datprep
datapath = "/home/stuke/Databases/QM9_XYZ_below10/"

class CurveObj:
    def __init__(self, name):
        self.xnparray = None
        self.ynparray = None
        self.xerror = None
        self.yerror = None
        self.name = name

def cleanup_results(result_file, multiple_runs = False, Choose_Folder = False, rep_no = 1):
    ''' gets data from resultfile and returns plottable Curve objects
    Variables
    ---------
    resultsfile : string, path to file containing pickled Result objects
    multiple_runs : if True, calculate mean of runs with same lamda and sigma
    Choose_Folder: boolean, if True, file is directly stored to result_file.
                    if not, result file is stored in ./Pickled/Kernel_Results folder


    Returns
    -------
    this_curve : LearningResults object
    '''
    
    if not Choose_Folder:
        #print("your results were stored to ./Pickled/Kernel_Results/")
        result_file = "./Pickled/Kernel_Results/" + result_file + "_" + str(rep_no)+ "reps"


    plottable_curves = []
    if rep_no > 1:
        multiple_runs = True


    if multiple_runs:
        lamdas = []
        sigmas = []

    results_list = datprep.read_compounds(result_file)
    #print("len results_list:", len(results_list)) 
    
    for result in results_list:
        #print("type result:", type(result))
        lamda = result.lamda
        sigma = result.sigma
        xlist = result.set_sizes
        ylist = result.maes
        if not multiple_runs:
            name = curve_name(sigma, lamda)
            curve = CurveObj(name)
            curve.xnparray = xlist
            curve.ynparray = ylist

            plottable_curves.append(curve)
            
        else:
            lamdas.append(lamda)
            sigmas.append(sigma)
    
    #probably plottable_curves could already be returned here for False
    if multiple_runs:
        for l in list(set(lamdas)): #get all unique occurances for lamda
            for s in list(set(sigmas)): #get all unique occurances for sigma
                same_x = []
                same_y = []

                #find all results with these s and l
                for result in results_list:
                    if result.lamda == l and result.sigma == s:
                        same_x.append(result.set_sizes)
                        same_y.append(result.maes)
                #print("all arrays of same y:\n", same_y)        
                #calculate average now
                av_ylist, yerror = jmath.calculate_mean(same_y)
                print("the calculated mean and it's error are:\n mean:", av_ylist, "\n error:", yerror)
                #add Curve object
                name = curve_name(s,l)
                curve = CurveObj(name)
                curve.xnparray = same_x[0]
                curve.ynparray = av_ylist
                curve.yerror = yerror
                
                plottable_curves.append(curve)


    return(plottable_curves)

def curve_name( sigma, lamda):
    name = ', sigma = %s, lambda = %s'% (str(sigma), str(lamda))
    return(name)

def plot_learning(set_sizes = [10, 20, 40, 80, 160, 300], maes = [2.035934924620435, 1.8125458722942258, 1.7146791116661697, 1.6779313630086368, 1.8454600048725978, 1.8763117260488518], perc_maes = [232.7964404444149, 205.87841291639506, 190.275572697472, 162.50375206243325, 254.048604095239, 64.66622415061042], title = "sigma = 20\nlambda = 1e-3"):

    ax[0].loglog(set_sizes, maes, label = title, linewidth = fontsize/8)
    ax[1].loglog(set_sizes, perc_maes, label = title, linewidth = fontsize/8)
    return()

def plot_curves(curve_list, file_title = "TrialLearning",\
        plottitle = 'Learning Curves of CM Eigenvector Representation on QM9 Dataset\n 1000 molecules, 2 Runs Averaged',\
        xtitle = 'Training Set Size',\
        ytitle = 'MAE [hartree]',\
        multiple_runs = True,\
        include_title = False):

    '''plots learning curves
    curve_list: lsit of data
    file_title: string, where to store plot to
    plottitle = title over plot
    xtitle : string, title of x axis
    ytitle : string, title of y axis
    mutliple_runs : boolean, if true, plot curves with variation shadowed
    include_title : boolean, if true, plot title
    '''

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
    
    
    f, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 8))
    
    #plot title if included
    if include_title:
        st = f.suptitle(plottitle)

        # shift subplots down and to the left to give title and legend space:
        st.set_y(0.95)
    f.subplots_adjust(top=0.8, left = 0.05, right = 0.78, wspace = 0.1)

    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    #all the plotting has to be done below
    for c in range(len(curve_list)):
        curve = curve_list[c]
        
        ax.plot(curve.xnparray, curve.ynparray, linewidth = 2, label = curve.name)
        print('x:' ,curve.xnparray,'\ny:', curve.ynparray)

        '''
        #check whether learning worked
        if curve.ynparray[8] > curve.ynparray[0]:
            #print('a curve with the following x and y arrays was plotted')
            #print('x:' ,curve.xnparray,'\ny:', curve.ynparray)
            ax.plot(curve.xnparray, curve.ynparray,  label = curve.name)
        else:
            print("this curve was excluded:", curve.name)
        #if multiple_runs:
        #    #ax.plot(curve.xnparray, curve.ynparray, 'o')
        #    ax.errorbar(curve.xnparray, curve.ynparray, yerr = curve.yerror, fmt = '-o')
        '''
    #all the plotting has to be done above
    
    #f.legend = ax.legend(loc = 'center right')
    handles, labels = ax.get_legend_handles_labels()

    f.legend(handles, labels) #, bbox_to_anchor=(1,1), loc="upper left")
    
    figuretitle = "./Images/" + file_title + ".png"

    f.savefig(figuretitle, bbox_inches = 'tight')
    return(print("figure was saved to", figuretitle))
