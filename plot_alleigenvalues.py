import database_preparation as datprep
import jax_derivative as jder
import jax_additional_derivative as jader
import jax_representation as jrep
import plot_derivative as pltder
import jax.numpy as jnp
import sys
from plot_derivative import prepresults

results_file =["/home/linux-miriam/Databases/Pickled/qm7_CM_results.pickle",\
        "/home/linux-miriam/Databases/Pickled/qm7_CM_EV_results.pickle",\
        "./Pickled/fourcompounds_EV_results.pickle",\
        "./Pickled/trial_numder.pickle"]

results = datprep.read_compounds(results_file[3])
repro = "OM"

xdata, ydata, newresults = prepresults(results, rep = repro)

#datprep.store_compounds(newresults, results_file)


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

