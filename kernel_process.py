import kernel_learning as kler
import kernel_plot as kplot
import database_preparation as datprep
import kernel_representations as krep
from time import time as tic


def kernel_learning(datapath, final_file, representation_no = 0, maxnumber = 3993):
    start = tic()

    #define parameters for learning
    set_sizes = [2, 4, 10] #not bigger than the total number of instances in set
    sigmas = [80]       #how tight is the fit? needs to be tested depending on data varies widely
    lambdas = [1e-15]  #how much variation to the initial data is introduced? 1e-17 - 1e-13 good to try
    number_of_runs = 3 #how many times should the learning be done before averaging and plotting?

    #define representation
    representation_list = [krep.Coulomb_Matrix,\
            krep.Eigenvalue_Coulomb_Matrix,\
            krep.Bag_of_Bonds,\
            krep.Overlap_Matrix,\
            krep.Eigenvalue_Overlap_Matrix]

    repro = representation_list[representation_no]

    #unpack pickled data to compounds list
    compounds = datprep.read_compounds(datapath)
    print("len of compoundlist:", len(compounds))

    #shorten compounds to make stuff faster
    compounds = compounds[:maxnumber]
    print("len of compoundlist used for this run:", len(compounds))


    #for compounds create list of energies and list of fingerprints in "repro" representation
    energylist = []
    fingerprintlist = []

    for c in compounds:

        #get properties from compound class
        energy = float(c.energy)
    
        Z = c.Z
        R = c.R
        N = c.N
    
        #calculate fingerprint of molecule
        fingerprint = repro(Z, R, N)
         
        #add energy and fingerprint to lists
        energylist.append(energy)
        fingerprintlist.append(fingerprint)

    t_compounds = tic()

    print("time start to compounds: ", t_compounds - start)

    #run learning
    results, metadata = kler.full_kernel_ridge(fingerprintlist,\
            energylist,\
            final_file,\
            set_sizes,\
            sigmas,\
            lambdas,\
            rep_no = 1,\
            upperlimit = maxnumber)

    return()
