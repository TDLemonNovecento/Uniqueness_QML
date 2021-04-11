import database_preparation as datprep
import jax_derivative as jder
import jax_additional_derivative as jader
import jax_representation as jrep
import plot_derivative as pltder
import jax.numpy as jnp
import sys, os
from plot_derivative import prepresults

'''with the following variable the derivatives to be included in the final plot are chosen'''
which_d = [4] #[0, 1, 2, 3, 4] # 0 = dZ, 1 = dR, 2 = ddR, 3 = dZdR, 4 = dZdZ

'''this variable sets the xaxis for plotting. norm of CM matrix or number of atoms in molecule can be chosen'''
xnorm = "norm" #"norm" #"nuc"

by_nuc = False
if xnorm == "nuc":
    by_nuc = True

'''this variable defines how nonzero values are counted, as absolutes or as percentage'''
yvalues = "perc" #"perc" #"abs"

der = ["dZ", "dR", "dRdR", "dZdR", "dZdZ"]

dname = der[which_d[0]]

representations = [0, 1, 2]#, 3, 4] #0 = CM, 1 = EVCM, 2 = BOB, 3 = OM, 4 = EVOM

'''this list is necessary to assign the correct labels'''
reprolist = ['CM', 'EVCM', 'BOB', 'OM', 'EVOM']


'''below are lists of pickled results files or their partial paths'''
results_file =["/home/linux-miriam/Databases/Pickled/qm7_CM_results.pickle",\
        "/home/linux-miriam/Databases/Pickled/qm7_CM_EV_results.pickle",\
        "/home/linux-miriam/Databases/Pickled/BoB_numder_res",\
        "/home/linux-miriam/Databases/Pickled/OM_numder_res",\
        "/home/linux-miriam/Databases/Pickled/EVOM_numder_res",\
        "./Pickled/fourcompounds_EV_results.pickle",\
        "./Pickled/trial_numder.pickle"]



#for OM and BoB no full pickled file could be made
completeresults = []
ydatalist = []
xdatalist = []


for i in representations:
    repro = reprolist[i]
    #print("representation:", repro)
    #print("len ydatalist:", len(ydatalist))

    filename = results_file[i]

    if i < 2:
        #repro is pickled in one file
        results = datprep.read_compounds(filename)
        
        #get xdata and ydata with labels (ydatalist[i][1] stores labels)
        xdata, ydata, newresults = prepresults(results, rep = repro,\
                dwhich = which_d, repno = i,\
                norm = xnorm,\
                yval = yvalues,\
                with_repro = True,\
                with_whichd = False)
        
        xdatalist.append(xdata)
        ydatalist.extend(ydata)


    else:
        #for BOB, EVOM and OM results were stored in files containing 100 compounds each
        j = 0

        #collect and append numbers to one array, to later merge to one dataset in yaxis
        fullydata = []
        fullxdata = []
        ydatalabel = ""

        for k in [100, 200, 300, 400, 500, 600, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900,\
                2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600,\
                3700, 3800, 3900]:
            
            if k < 3900:
                numbers = "%i-%i" % (k, k+100)
            else:
                numbers = "%i-3993" %(k)

            partialfilename = filename + numbers

            if os.path.isfile(partialfilename):
                results = datprep.read_compounds(partialfilename)
            else:
                print("this file does not exist, skip: ", partialfilename)
                continue

            results = datprep.read_compounds(partialfilename)
            j += 1

            xdata, ydata, newresults = prepresults(results, rep = repro,\
                    dwhich = which_d, repno = i,\
                    norm = xnorm,\
                    yval = yvalues,\
                    with_repro = True,\
                    with_whichd = False)
            
            #datprep.store_compounds(newresults, partialfilename)

            fullxdata.extend(xdata) 
            print("attention: if there are problems with the xdata list and ydatalist length when plotting\
                    check whether different lengths of data are included\
                    for BOB; EVOM and OM the length may vary depending on the range depickted")
            
            for y in ydata:
                fullydata.extend(y[0])
                ydatalabel = y[1]

            #fullydata contains list of [ydata, ylabel] pairs, which need to be merged to one
        
        ydatalist.append([fullydata, ydatalabel])
        xdatalist.append(fullxdata)



print("len xdatalist:", len(xdatalist))
print("len ydatalist:", len(ydatalist))


#all derivatives dZ of all repros in one panel
if xnorm == "nuc":
    xtitle = "Number of Atoms in Molecule"
else: #xnorm = "norm" or something else that is not yet defined
    xtitle = "Norm of Coulomb Matrix"

pltder.plot_percentage_zeroEV(xdatalist, ydatalist,\
        title = dname + " Derivatives on QM7 Dataset",\
        savetofile = "./Images/Final/" + dname +"_derivatives_" + xnorm,\
        oneplot = True,\
        representations = representations,\
        xaxis_title = xtitle,\
        Include_Title = False,\
        by_nuc = by_nuc)


'''
#all derivatives of one repro in one image
plotname = "./Images/QM7_derivatives/%s_derivatives"%repro

pltder.plot_percentage_zeroEV(xdata, ydata,\
        title = repro + " Derivatives on QM7 Dataset",\
        savetofile = plotname + "one",\
        oneplot = True,\
        representations = 1)
       # xaxis_title = "Number of Atoms in Molecule")



pltder.plot_percentage_zeroEV(xdata, ydata,\
        title = repro +" Derivatives on QM7 Dataset", \
        savetofile = plotname + "panel",\
        oneplot = False,\
        representations = 1)
       # xaxis_title = "Number of Atoms in Molecule")

'''
