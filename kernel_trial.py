import kernel_learning as kler
import kernel_plot as kplot
import database_preparation as datprep
import kernel_representations as krep
from time import time as tic
import kernel_process as kproc

#define datapath to a pickled list of compound instances
datapath = "./Pickled/qm7.pickle"
final_file_list = ['./Results/CM_learningresults.obj', \
        './Results/EVCM_learningresults.obj', \
        './Results/BOB_learningresults.obj',\
        './Results/OM_learningresults.obj',\
        './Results/EVOM_learningresults.obj']

repnames = ["CM","EVCM", "BOB", "OM", "EVOM"]

#for i in [4]:
#    kproc.kernel_learning(datapath, final_file_list[i], representation_no = i, maxnumber = 12)


curve_list = []
for i in range(5):
    final_file = final_file_list[i]
    curves = kplot.cleanup_results(final_file, multiple_runs = True)
    for curve in curves:
        curve.name = repnames[i] + curve.name
        curve_list.append(curve)
    kplot.plot_curves(curves, file_title = final_file[11:], plottitle = final_file + "Learning on 200 QM7 datapoints")


kplot.plot_curves(curve_list, file_title = "Trialplot", plottitle = "Multiple Learning Curves on 200 QM7 datapoints")
