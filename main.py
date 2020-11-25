'''In this program routines and subroutines are called depending on what shoud be done. Unused parts should be hashed and parts that conflict with each other marked with exclamation marks and special characters respectively twice that character. Written by Miriam Stuke.
'''
import time
import database_preparation as datprep
import jax_derivative as jder
import jax_representation as jrep
import jax.numpy as jnp
import plot_derivative as pltder

#define path to folder containing xyz files. All files are considered.
database = "/home/linux-miriam/Uniqueness_QML/Pickled/qm7.pickle"
small_data_file = "/home/linux-miriam/Uniqueness_QML/Pickled/fourcompounds.pickle"
dat_ha_file = "/home/linux-miriam/Uniqueness_QML/Pickled/qm7.pickle"

compounds = datprep.read_compounds(small_data_file)

for c in compounds:
    ev, vectors = jrep.CM_ev(c.Z, c.R, c.N)
    print("name of compound:", c.filename)
    print("eigenvalue repro:\n", ev)

    derivative = jder.sort_derivative('CM_EV', c.Z, c.R)
