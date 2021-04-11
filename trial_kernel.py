import kernel_learning as kler
import plot_kernel as kplot
import database_preparation as datprep
#import representation_ZRN as ZRN_rep
from time import time as tic
import kernel_process as kproc

#define datapath to a pickled list of compound instances
datapath = "./Pickled/qm7.pickle"

#number of runs:
repno = 3

#final_file_list = ["./Results/trial.obj"]
final_file_list = ['CM_QM7', \
        'EVCM_QM7', \
        'BOB_QM7',\
        'OM_QM7',\
        'EVOM_QM7']

repnames = ["CM","EVCM", "BOB", "OM", "EVOM"]
max_no = 3993 #150

for i in [0, 1, 2]:
    results = kproc.kernel_learning(datapath, final_file_list[i], representation_no = i, maxnumber = max_no, repno = repno)

curve_list = []
for i in [0, 1, 2]:
    final_file = final_file_list[i]
    curves = kplot.cleanup_results(final_file, rep_no = repno)
    for curve in curves:
        curve.name = repnames[i] + curve.name
        curve_list.append(curve)

        #print("curve", curve)
    #kplot.plot_curves(curves, file_title = final_file[11:], plottitle = final_file + "Learning on 200 QM7 datapoints")


kplot.plot_curves(curve_list, file_title = "ML_trial", plottitle = "Learning of Molecular Energies on QM7 Dataset")
