import kernel_learning as kler
#import kernel_plot as kplot
import database_preparation as datprep
import representation_ZRN as ZRN_rep
from time import time as tic


def kernel_learning(datapath, final_file, representation_no = 0, maxnumber = 3993):
    start = tic()

    #define parameters for learning
    set_sizes = [5, 10, 25, 70, 120, 300, 600, 1500, 3000] #not bigger than the total number of instances in set
    sigmas = [ 4, 10, 25, 80]       #how tight is the fit? needs to be tested depending on data varies widely
    lambdas = [1e-15]  #how much variation to the initial data is introduced? 1e-17 - 1e-13 good to try
    number_of_runs = 3 #how many times should the learning be done before averaging and plotting?

    #define (hashed) representation
    representation_list = [ZRN_rep.Coulomb_Matrix_h,\
            ZRN_rep.Eigenvalue_Coulomb_Matrix_h,\
            ZRN_rep.Bag_of_Bonds,\
            ZRN_rep.Overlap_Matrix_h,\
            ZRN_rep.Eigenvalue_Overlap_Matrix_h]

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
            rep_no = number_of_runs,\
            upperlimit = maxnumber)

    return(results)
