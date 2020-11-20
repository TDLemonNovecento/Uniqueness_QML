'''contains all sorts of functions used to plot derivatives'''
#import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import sys
import numpy as np
#import svgutils.transform as sg

'''standard settings for matplotlib plots'''
fontsize = 24
plt.rc('font',       size=fontsize) # controls default text sizes
plt.rc('axes',  titlesize=fontsize) # fontsize of the axes title
plt.rc('axes',  labelsize=fontsize) # fontsize of the x and y labels
plt.rc('xtick', labelsize=fontsize*0.8) # fontsize of the tick labels
plt.rc('ytick', labelsize=fontsize*0.8) # fontsize of the tick labels
plt.rc('legend', fontsize=fontsize*0.8) # legend fontsize
plt.rc('figure',titlesize=fontsize*1.2) # fontsize of the figure title



def plot_percentage_zeroEV(norm_xaxis, percentages_yaxis, title, savetofile = "perc_nonzeroEV_CM_test", oneplot = False):
    #general figure settings
    if oneplot:
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(12,8))
    else:
        fig = plt.figure(figsize = (12,8))
        ax = fig.add_subplot(111) #main subfigure for titles and stuff
        ax_d1 = fig.add_subplot(221) #first single derivative
        ax_d2 = fig.add_subplot(223) #second single derivative
        ax_dd1 = fig.add_subplot(322) #first double derivative
        ax_dd2 = fig.add_subplot(324) #second double derivative
        ax_dd3 = fig.add_subplot(326) #third double derivative

    fig.tight_layout()
 
    #add all plots
    if oneplot:
        for y in percentages_yaxis:
            ax.scatter(norm_xaxis, y[0], label = y[1])
    else:
        y = percentages_yaxis #for simplicity
        ax_d1.scatter(norm_xaxis, y[0][0])
        ax_d1.title.set_text(y[0][1])
        ax_d2.scatter(norm_xaxis, y[1][0])
        ax_d2.title.set_text(y[1][1])
        ax_dd1.scatter(norm_xaxis, y[2][0])
        ax_dd1.title.set_text(y[2][1])
        ax_dd2.scatter(norm_xaxis, y[3][0])
        ax_dd2.title.set_text(y[3][1])
        ax_dd3.scatter(norm_xaxis, y[4][0])
        ax_dd3.title.set_text(y[4][1])
        
        #turn off ticks and so on for outer subfigure
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')

        ax.set_xlabel('Norm of Coulomb Matrix', labelpad = 30)
        ax.set_ylabel('Fraction of Nonzero Eigenvalues', labelpad = 50)



    #title, axis and legend
    st = fig.suptitle(title)

    if oneplot:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title = 'Derivatives')
    
        plt.xlabel('Norm of Coulomb Matrix')
        plt.ylabel('Fraction of Nonzero Eigenvalues')
        fig.subplots_adjust(top=0.92, bottom = 0.1, left = 0.12, right = 0.97)

        # shift subplots down and to the left to give title and legend space:
        #hspace, wspace increases space between subplots
    else:
        fig.subplots_adjust(top=0.87, left = 0.10, right = 0.97,  wspace = 0.2, hspace = 0.5) #right = 0.82
    
    
    #save and display plot
    name = savetofile
    plt.savefig(name, transparent = True)
    

    return(print("plots have been saved to %s" % name))

def pandaseries_dR(eigenvalues, dimZ):
    label_dR = [['dx%i' %(i+1) , 'dy%i' %(i+1) , 'dz%i' %(i+1)]  for i in range(dimZ)]
    listof_series = []
    for i in zip(eigenvalues, label_dR): #combines correct axis (x,y,z) with derivatives
        for j in zip(i[0],i[1]): #combines correct atom (1, 2, 3...) with derivative
            s = pd.Series(j[0], name = j[1])  #creates panda series with index being range(len(j[0]))
            listof_series.append(s)
    return(listof_series)

def pandaseries_dZ(eigenvalues, dimZ):
    label_dZ = ['dZ%i' % (i + 1) for i in range(dimZ)]
    listof_series = []
    for i in zip(eigenvalues, labels_dZ):
        s = pd.Series(j[0], name = j[1])

    return(listof_series)

def pandaseries_dN(eigenvalues, dimZ):
    label_dN = ['dN']
    s = pd.Series(eigenvalues, labels)
    return(s)

def plot_dR(listof_series, listof_names, dimZ):
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

        plots dR plots

        Return
        ------
        figurename: name of stored figure

    '''
    
    #general figure settings
    fig, ax = plt.subplots(nrows = dimZ, ncols = 3, figsize=(12,8), sharey = True)
    figurename = "Ethin_CM_dR.svg"
    
    fig.tight_layout()
    st = fig.suptitle("Eigenvalues of First Derivative of Ethin with Rotating H")

    # shift subplots down and to the left to give title and legend space:
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85, left = 0.05, right = 0.82, wspace = 0.1)
    
    #start with plotting:
    #loop over all series that were passed and plot to correct window
    for idx in range(len(listof_series)):
        series = listof_series[idx]

        for i in range(dimZ): #chooses x, y, z aka ncols
            for j in range(3): #chooses 1, 2, 3... aka nrows
                #print('i = ', i, 'j = ', j,'series[i + j]', series[i + j]) 
                newline = ax[i][j].plot(series[i*3 + j], label = int(float(listof_names[idx])))[0]

    #set title and show legend for all subplots
    for i in range(dimZ):
        for j in range(3):
            ax[i][j].set_title(series[i*3+j].name)
    
    #prepare legend based on labels
    handles, labels = ax[0][0].get_legend_handles_labels()

    #for numeric labels, they can be sorted by the following two lines of code
    ascending = sorted(zip(listof_names, handles, labels))
    listof_names, handles, labels = zip(*ascending)
    
    fig.legend(handles, labels, loc="lower right", title = '  $x$\n$\phi = 90/x$')
    
    plt.savefig(figurename, transparent = True)

    return(figurename)



def merge_plot_with_svg(figname, imagepath):
    #create new SVG figure
    fig = sg.SVGFigure("16cm", "6.5cm")

    # load matpotlib-generated figures
    # load matpotlib-generated figures
    fig1 = sg.fromfile(figname)
    fig2 = sg.fromfile(imagepath)

    # get the plot objects
    plot1 = fig1.getroot()
    plot2 = fig2.getroot()
    plot2.moveto(280, 0, scale=0.5)

    # add text labels
    txt1 = sg.TextElement(25,20, "A", size=12, weight="bold")
    txt2 = sg.TextElement(305,20, "B", size=12, weight="bold")

    # append plots and labels to figure
    fig.append([plot1, plot2])
    fig.append([txt1, txt2])
   
    # save generated SVG files
    fig.save("fig_final.svg")

