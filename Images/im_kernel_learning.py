import kernel_learning as kler
import plot_kernel as kplot
import database_preparation as datprep
#import representation_ZRN as ZRN_rep
from time import time as tic
import kernel_process as kproc


#number of runs:
repno = 1

final_file_list = ['CM_QM7', \
        'EVCM_QM7', \
        'BOB_QM7',\
        'OM_QM7',\
        'EVOM_QM7']

repnames = ["CM","EVCM", "BOB", "OM", "EVOM"]
max_no = 3993 #150

final_file = "./tmp/Curves.pickle"
curve_list = datprep.read_compounds(final_file)


kplot.plot_curves(curve_list, file_title = "Final/ML_kernel_laplacian", plottitle = "Learning of Molecular Energies on QM7 Dataset")
