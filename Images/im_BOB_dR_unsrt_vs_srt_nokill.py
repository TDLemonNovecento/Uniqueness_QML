import sys
sys.path.insert(0, "..")

import database_preparation as datprep
import jax_derivative as jder
import jax_additional_derivative as jader
import jax_representation as jrep
import plot_derivative as pltder
import jax.numpy as jnp
import os
from plot_derivative import prepresults
import matplotlib.pyplot as plt
import numpy as np


'''with the following variable the derivatives to be included in the final plot are chosen'''
which_d = [0] # 0 = dZ, 1 = dR, 2 = ddR, 3 = dZdR, 4 = dZdZ

'''this variable sets the xaxis for plotting. norm of CM matrix or number of atoms in molecule can be chosen'''
xnorm = "nuc" #"norm" #"nuc"

'''this variable defines how nonzero values are counted, as absolutes or as percentage'''
yvalues = "abs" #"perc" #"abs"

der = ["dZ", "dR", "dRdR", "dZdR", "dZdZ"]

dname = der[which_d[0]]

'''which representation is going to be plotted'''
representations =[1, 2] #0 = BOB broken dZ, 1 = sorted Bob, 2 = unsorted BOB

'''this list is necessary to assign the correct labels'''
reprolist = ['BOB broken dZ', 'sorted BOB', "unsorted BOB"]
colorlist = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

results_file = ["../Databases/Pickled/BOB2_numder_res100-800",\
        "../Databases/Pickled/BOB_sorted_rep/BoB_numder_res",\
        "../Databases/Pickled/BOB_unsorted_rep/BOB_numder_res"]



#start figure to prevent storage overflow
#general figure settings
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(12,8))
fig.tight_layout()

#add reference line of degrees of freedom
N = np.arange(1,23)

relevant_dim = 3*N - 6
ax.plot(N, relevant_dim, label = "Internal Degrees of Freedom (3n - 6)")


if xnorm == "nuc":
    xtitle = "Number of Atoms in Molecule"
else: #xnorm = "norm" or something else that is not yet defined
    xtitle = "Norm of Coulomb Matrix"

if yvalues == "abs": 
    ytitle = "Number of Nonzero Values"
else:
    ytitle = "Fraction of Nonzero Values"

plt.xlabel(xtitle)
plt.ylabel(ytitle)
fig.subplots_adjust(top=0.92, bottom = 0.1, left = 0.12, right = 0.97)

for i in representations:
    repro = reprolist[i]

    filename = results_file[i]

    if i == 0:
        results = datprep2.read_compounds(filename)

        #get xdata and ydata with labels (ydatalist[i][1] stores labels)
        xdata, ydata, newresults = prepresults(results, rep = repro,\
                dwhich = which_d, repno = i,\
                norm = xnorm, yval = yvalues,\
                with_whichd = False)

        #datprep.store_compounds(newresults, partialfilename)
        for yd in ydata:
            ax.scatter(xdata, yd[0], c = colorlist[i], label = repro)
            del(xdata)
            del(ydata)
            del(newresults)
            del(results)

        
    else:
        j = 0 #important so only one label is displayed
        unsrt_numbers = ["100-120", "120-140", "140-160", "160-180", "180-200",\
                "220-240", "240-260", "260-280", "280-300",\
                "300-320", "320-340", "340-360", "360-380", "380-400",\
                "400-420", "420-440", "440-460", "460-480", "480-500",\
                "520-540", "540-560", "560-580", "580-600",\
                "600-620", "620-640", "640-660", "660-680", "680-700",\
                "700-720", "720-740", "740-760", "760-780", "780-800",\
                "800-820", "820-840", "840-860", "880-900",\
                "920-940", "940-960", "980-1000", "1000-1020"]

        range_1000 = range(0, 3800, 100)
        srt_numbers = ["%i-%i"%(j, j+100) for j in range_1000]
        print(srt_numbers)
        number_ends = [srt_numbers, unsrt_numbers]
        print("number of molecules total:", 20*len(unsrt_numbers))
        for k in range(len(unsrt_numbers)):#range(0, 4000, 100):
            
                    
            partialfilename = filename + number_ends[i-1][k]
            print("file: ", partialfilename)

            if os.path.isfile(partialfilename):
                results = datprep.read_compounds(partialfilename)
            else:
                print(partialfilename, "was not found")
                continue

            xdata, ydata, newresults = prepresults(results, rep = repro,\
                    dwhich = which_d, repno = 2,\
                    norm = xnorm, yval = yvalues,\
                    with_whichd = False)
            
            #datprep.store_compounds(newresults, partialfilename)
            for yd in ydata:
                ax.scatter(xdata, yd[0], c = colorlist[i], label = repro if j == 0 else "")
                #np.savez(outfile, xdata = x, yd[0] = y)
                del(xdata)
                del(ydata)
                del(newresults)
                del(results)
            
            j += 1

        


#add labels
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc = "upper left")#, title = 'Derivatives')


plt.savefig("./BOB_srt_vs_unsrt_%s.png" %dname, transparent = True, bbox_inches = 'tight')
print("file has been saved to", "./BOB_srt_vs_unsrt_%s.png" %dname)
