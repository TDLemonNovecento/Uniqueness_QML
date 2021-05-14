import numpy as np
import itertools
import kernel_learning as kler
import matplotlib.pyplot as plt
import jax_math as jmath
import database_preparation as datprep
datapath = "/home/stuke/Databases/QM9_XYZ_below10/"

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

def plot_learning(set_sizes ,\
        maes,\
        labels = [],\
        xtitle = 'Training Set Size',\
        ytitle = 'MAE [kcal/mol]',\
        title = "sigma = 20\nlambda = 1e-3",\
        filename = "QML_learning"):
    
    '''
    plots x and y values of learning curves
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
    
    for m in range(len(maes)):
        mae_list = maes[m]
        name = labels[m]
        ax.loglog(set_sizes, mae_list, label = name, linewidth = 2)
    
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('axes', -0.05))
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['left'].set_position(('axes', -0.05))

    ax.xaxis.set_tick_params(width=2.5, length=20)
    ax.yaxis.set_tick_params(width=2.5, length=20)


    #set x and y axis label
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)

    #add legend
    ax.legend()

    #save figure
    f.savefig(filename, bbox_inches = 'tight')

    return(print("figure was saved to", filename))



def plot_scatter(y_test, y_predicted, label = "OM",\
        title = "OM Representation Gaussian Kernel",\
        figuretitle = "Scatterplot_OM",\
        xtitle = "Atomic Energies [kcal/mol]",\
        ytitle = "Predicted Atomic Energies [kcal/mol]"):
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

    #prep figure
    f, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 8))
    
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('axes', -0.05))
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['left'].set_position(('axes', -0.05))

    ax.xaxis.set_tick_params(width=2.5, length=20)
    ax.yaxis.set_tick_params(width=2.5, length=20)


    st = f.suptitle(title)

    #plot results
    ax.scatter(y_test, y_predicted, label = label)
    
    #make x and y ticks the same
    multiplier = 10 ** -2
    min_tick = int(min(y_test)*multiplier) / multiplier

    plt.xticks(np.arange(min_tick, max(y_test)+100, 200))
    plt.yticks(np.arange(min_tick, max(y_test)+100, 200))
    #save figure

    f.savefig(figuretitle, bbox_inches = 'tight')
    return(print("figure was saved to", figuretitle))


def plot_curves(curve_list, file_title = "TrialLearning",\
        plottitle = 'Learning Curves of CM Eigenvector Representation on QM9 Dataset\n 1000 molecules, 2 Runs Averaged',\
        xtitle = 'Training Set Size',\
        ytitle = 'MAE [kcal/mol]',\
        multiple_runs = False,\
        include_title = False):

    '''plots learning curves from hartree to kcal/mol
    curve_list: list of CurveObj class objects
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


    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('axes', -0.05))
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['left'].set_position(('axes', -0.05))

    ax.xaxis.set_tick_params(width=2.5, length=20)
    ax.yaxis.set_tick_params(width=2.5, length=20)

    #assign marker for better differentiation
    marker = itertools.cycle((',', '+', '.', 'o', '*', "1", "2", "3", "4", "8", "H", "D")) 

    #all the plotting has to be done below
    for curve in curve_list:
    
        yarray = curve.ynparray
        
        ax.plot(curve.xnparray, curve.ynparray, linewidth = 2, marker = next(marker), label = curve.name)
        print('x:' ,curve.xnparray,'\ny:', yarray)

    #all the plotting has to be done above
    
    f.legend = ax.legend(loc = 'lower left')
    
    #handles, labels = ax.get_legend_handles_labels()

    #f.legend(handles, labels, bbox_to_anchor=(1,1), loc="upper left")
    
    figuretitle = "./Images/" + file_title + ".png"
    f.savefig(figuretitle, bbox_inches = 'tight')
    return(print("figure was saved to", figuretitle))
