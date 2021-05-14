import database_preparation as datprep
import database_preparation2 as datprep2


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

representations = [2]#, 3, 4] #0 = CM, 1 = EVCM, 2 = BOB, 3 = OM, 4 = EVOM

'''this list is necessary to assign the correct labels'''
reprolist = ['CM', 'EVCM', 'BOB', 'OM', 'EVOM']


'''below are lists of pickled results files or their partial paths'''
results_file =["/home/linux-miriam/Databases/Pickled/qm7_CM_results.pickle",\
        "/home/linux-miriam/Databases/Pickled/qm7_CM_EV_results.pickle",\
        "/home/linux-miriam/Databases/Pickled/BOB_numder_res",\
        "/home/linux-miriam/Databases/Pickled/OM_numder_res",\
        "/home/linux-miriam/Databases/Pickled/EVOM_numder_res",\
        "./Pickled/fourcompounds_EV_results.pickle",\
        "./Pickled/trial_numder.pickle"]

new_filename = "/home/linux-miriam/Databases/Pickled/BOB2_numder_res"

for i in representations:
    repro = reprolist[i]
    #print("representation:", repro)
    #print("len ydatalist:", len(ydatalist))

    filename = results_file[i]
    

    for k in range(100, 1000, 100):

        new_res_array = []

        print("k is: ", k)
        if k < 3900:
            numbers = "%i-%i" % (k, k+100)
        else:
            numbers = "%i-3993" %(k)
        partialfilename = filename + numbers

        if os.path.isfile(partialfilename):
            results = datprep.read_compounds(partialfilename)

            print("len results:", len(results))
            continue
        else:
            print("this file does not exist, skip: ", partialfilename)
            continue

        print("len results:", len(results))
        for res in results:
            res2 = datprep2.derivative_results(res.filename, res.Z)
            res2.norm = res.norm
            res2.representation_form = res.representation_form
            res2.dZ_ev = res.dZ_bigger
            res2.dR_ev = res.dR_bigger
            res2.dZdR_ev = res.dZdR_bigger
            res2.dZdZ_ev = res.dZdZ_bigger
            res2.dRdR_ev = res.dRdR_bigger

            res2.dZ_perc = res.dZ_perc
            res2.dR_perc = res.dR_perc
            res2.dZdR_perc = res.dZdR_perc
            res2.dRdR_perc = res.dRdR_perc
            res2.dZdZ_perc = res.dZdZ_perc

            new_res_array.append(res2)
            #print(vars(res))

        datprep2.store_compounds(new_res_array, new_filename + numbers) 
