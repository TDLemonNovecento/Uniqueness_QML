import numpy as np
import kernel_learning as krnl
import matplotlib.pyplot as plt

datapath = "/home/stuke/Databases/QM9_XYZ_below10/"

def my_learning_curves(folder, sigma = 20, set_sizes = [10, 20, 40, 80, 160, 300], lamda = 1.0e-3):
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
    

def plot_learning(set_sizes = [10, 20, 40, 80, 160, 300], maes = [2.035934924620435, 1.8125458722942258, 1.7146791116661697, 1.6779313630086368, 1.8454600048725978, 1.8763117260488518], perc_maes = [232.7964404444149, 205.87841291639506, 190.275572697472, 162.50375206243325, 254.048604095239, 64.66622415061042], title = "QM9 dataset with up to 9 Atoms"):
    fontsize = 30

    plt.rc('font',       size=fontsize) # controls default text sizes
    plt.rc('axes',  titlesize=fontsize) # fontsize of the axes title
    plt.rc('axes',  labelsize=fontsize) # fontsize of the x and y labels
    plt.rc('xtick', labelsize=fontsize*0.8) # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontsize*0.8) # fontsize of the tick labels
    plt.rc('legend', fontsize=fontsize*0.8) # legend fontsize
    plt.rc('figure',titlesize=fontsize*1.2) # fontsize of the figure title
    


my_learning_curves(datapath)

