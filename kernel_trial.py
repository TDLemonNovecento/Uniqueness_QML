import kernel_learning as kler
import kernel_plot as kplot
datapath = "/home/stuke/Databases/XYZ_ethin/" #always has to end with /
final_file = datapath+'/trial_learningresults.obj'

#results, metadata = kler.full_kernel_ridge(datapath, final_file, [1, 3, 5, 8, 11], [4, 40], [1e-13, 1e-11], rep_no = 2 )
#print(results)

curves = kplot.cleanup_results(final_file)
kplot.plot_curves(curves)
