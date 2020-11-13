'''In this program routines and subroutines are called depending on what shoud be done. Unused parts should be hashed and parts that conflict with each other marked with exclamation marks and special characters respectively twice that character. Written by Miriam Stuke.
'''
import time
import database_preparation as datprep
import jax_derivative as jder
import jax_representation as jrep
import jax.numpy as jnp

#define path to folder containing xyz files. All files are considered.
database = "/home/linux-miriam/Databases/QM9_XYZ/"
database_file = "/home/linux-miriam/Uniqueness_QML/Pickled/qm9.pickle"
dat_ha_file = "/home/linux-miriam/Uniqueness_QML/Pickled/qm7.pickle"
trial_file = "/home/linux-miriam/Uniqueness_QML/Pickled/XYZ_random_ha5.txt"
result_file = "/home/linux-miriam/Uniqueness_QML/results.pickle"


#qm7_compounds = datprep.read_compounds(dat_ha_file)
results = datprep.read_compounds(result_file)



for result in results:
    max_ev = 9

    dZ_ev = result.dZ_ev
    dR_ev = result.dR_ev
    ddZ_ev = result.ddZ_ev
    ddR_ev = result.ddR_ev
    dZdR_ev = result.dZdR_ev
    
    perc_dR = []
    perc_ddZ = []
    perc_ddR = []
    perc_dZdR = []
    nonzero_dZ = []
    perc_dZ = []

    #for loops much faster
    start_t = time.time()
    for eigenvals in dZ_ev:
        EV = jnp.real(eigenvals)
        nz = jnp.count_nonzero(EV)
        perc = nz /max_ev
        perc_dZ.append(perc)

    #print("time reals: ", start_t - middle_t, "results: ", nonzero_dZ)
    #print("time imag: ", middle_t - end_t, "results: ", perc_dZ)


    for eigenvals in perc_dR:
        nz = jnp.count_nonzero(EV)
    
    end_time = time.time()
    #print("stacked calculation: ", start_t - middle_time, "for loop: ", middle_time - end_time)
    
