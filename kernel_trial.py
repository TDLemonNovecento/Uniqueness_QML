import kernel_learning as kler
import kernel_plot as kplot
datapath = "/home/stuke/Databases/QM9_XYZ/" #always has to end with /
final_file = datapath+'/trial_learningresults.obj'

#results, metadata = kler.full_kernel_ridge(datapath, final_file, [10, 50, 100, 250, 500, 750, 950], [4], [1e-13], rep_no = 1)


curves = kplot.cleanup_results(final_file)
print(curves)
kplot.plot_curves(curves)
