'''In this program routines and subroutines are called depending on what shoud be done. Unused parts should be hashed and parts that conflict with each other marked with exclamation marks and special characters respectively twice that character. Written by Miriam Stuke.
The numbers in the comments indicate which parts of the programm can be executed independently - 
A) xyz data is read and stored as compound information.
B) compound information is read and results are calculated.
C) results are plotted
'''
import database_preparation as datprep
import jax_derivative as jder
import jax_additional_derivative as jader
import jax_representation as jrep
import plot_derivative as pltder
import jax.numpy as jnp
import sys
import representation_ZRN as ZRNrep


#for numerical calculations:
numerical_representations = [ZRNrep.Coulomb_Matrix, ZRNrep.Eigenvalue_Coulomb_Matrix, ZRNrep.Overlap_Matrix, \
        ZRNrep.Eigenvalue_Overlap_Matrix, ZRNrep.Bag_of_Bonds]

###ABC)

#If you want to plot from multiple pickle results file, use this code:
#It can be used after a compound list has been created to
#efficiently distribute the results calculations over multiple cores.
#this may be done via "nohup python3 -u trial_CMderivatives_plot.py 200 300 > job1.out &"
#which would put all the terminal information into the job1.out file. 
#In this example, compounds 200 - 299 would be processed and saved to the specified results file below 

try:
	init, end = int(sys.argv[1]), int(sys.argv[2])
except IndexError:
	init = int(input("starting point"))
	end = int(input("end point"))


###A)
###define path to folder containing xyz files. All files are considered.
database = "/home/miriam/Databases/QM9_XYZ/"
qm7_file = "./Pickled/qm7.pickle"
four_compounds = "./Pickled/fourcompounds.pickle"
CM_EV_results_file = "/home/linux-miriam/Databases/Pickled/qm7_CM_EV_results.pickle"
CM_results_file = "/home/linux-miriam/Databases/Pickled/qm7_CM_results.pickle"
OM_results_file = "/home/linux-miriam/Databases/Pickled/qm7_OM_results.pickle"
results_path = "/home/linux-miriam/Databases/Pickled/"
result_folder = "./Pickled/"

results_file_EV = CM_EV_results_file
result_file_CM = CM_results_file
results_file_OM = OM_results_file

data_file = four_compounds

'''
#If you want to plot from multiple pickle results file, use this code:
#result_file = "/home/linux-miriam/Uniqueness_QML/Pickled/results" 
'''

###A)
###read xyzfiles and store into a molecular list as well as a compound list
#mol_ls, compound_ls = datprep.read_xyz_energies(database)

###A)
###store compounds to database_file
#datprep.store_compounds(compound_ls, database_file)

###A) supplement: you don't need to do this step
###take info from database_file and extract all molecules with less than 7 heavy atoms to dat_ha_file
#max_atoms = datprep.sortby_heavyatoms(database_file, dat_ha_file, 7)

###A) supplement: if in doubt, just choose 23 for QM9 dataset. max_atoms is just the maximal size of your representation
###max_atoms is maximal number of atoms in file. needed to set size of CM
#print("all CM should have size " , max_atoms)
#input("Press enter once you have made sure the size of the unsorted CM matrix has been adapted accordingly")

###B)
###read list of compounds from data file
full_compound_ls = datprep.read_compounds(data_file)
print(len(full_compound_ls), " compounds in full data file")

###B)

#If you want to plot only part of all compounds, use this code:
try:
	compound_ls = full_compound_ls[init : end]
except IndexError:
	print("Your indices were out of bound, restart. min: 0, max: ", len(full_compound_ls))
	exit()



print(len(compound_ls), " of which are being processed")

###B)
###create new list of results from list of compounds

###analytical derivative:
#results = jader.calculate_eigenvalues('CM_EV', compound_ls)

###numerical derivative:
results, resultaddition = jader.calculate_num_der(numerical_representations[2], compound_ls)


###B)
###store list of results in result_file
'''
#If you want to plot from multiple pickle results file, use this code:
#result_file = result_folder + "results_%i-%i.pickle" %(init, end)
'''
print("results:", results)
print("len of results:", len(results))
datprep.store_compounds(results, results_file_OM)


#C) 

#If you want to plot from multiple pickle results file, use this code:
#these are used as file identifiers for the results_%i-%i.pickle files
#numbers = ["0-200", "200-400", "400-600", "600-800", "800-1000", "1000-1200", "1200-1400", "1400-1600", "1600-1800", "1800-2000", "2000-2200", "2200-2400", "2400-2600", "2600-2800", "2800-3000", "3000-3200", "3200-3400", "3400-3600", "3600-3800", "3800-3993"]

'''
#read list of compounds from data file
full_compound_ls = datprep.read_compounds(results_file)
#print(len(full_compound_ls), " compounds in full data file")

'''
###use if only part of dataset should be processed
#try:
#	compound_ls = full_compound_ls[init : end]
#except IndexError:
#	print("Your indices were out of bound, restart. min: 0, max: ", len(full_compound_ls))
#	exit()

#print(len(compound_ls), " of which are being processed")

#we'll be reading the results anew.
#This part of the program should therefore be executed after the top part
#has been executed for all the files which can then be called by referencing to the "numbers" list

#results = []


#for n in numbers:
#    filename = "/home/linux-miriam/Copy/results_%s.pickle" %n
#    print("filename: ", filename)
#    compoundlist = datprep.read_compounds(filename)
#    print("number of compounds in sublist: ", len(compoundlist[0]))
#    
#    results.extend(compoundlist[0])

#print("number of compounds: ", len(results))

#datprep.store_compounds(results, result_file)

#data has now been stored to resultfile


#C)
#read data from result file

#If you want to plot from multiple pickle results file, use this code:
#result_file = resultfile

results_OM = datprep.read_compounds(results_file_OM)
#results_EV = datprep.read_compounds(results_file_EV)
#results_CM = datprep.read_compounds(results_file_CM)
'''
try:
       results_EV = results_EV[init : end]
       results_CM = results_CM[init : end]
except IndexError:
       print("Your indices were out of bound, restart. min: 0, max: ", len(full_compound_ls))
       exit()
'''

#for res in results_EV:
#    res.convert_ev_to_np()
#    fractions, numbers = res.calculate_smallerthan()

#datprep.store_compounds(results_EV, results_file)

print("\n \n \n")
#C)
#prepare plotting
#y-axis information
dZ_percentages_EV = []
dR_percentages_EV = []
dZdZ_percentages_EV = []
dRdR_percentages_EV = []
dZdR_percentages_EV = []

'''
dZ_percentages_CM = []
dR_percentages_CM = []
dZdZ_percentages_CM = []
dRdR_percentages_CM = []
dZdR_percentages_CM = []
'''

#x-axis information
#CM_norms = []
norms = []

#C)
#get all the data from our results list
for i in range(len(results_OM)):

    #print(results_EV)
    #CM_norms.append(results_CM[i].norm)
    norms.append(results_OM[i].norm)


    results_perc_OM = results_OM[i].calculate_percentage()
    #results_perc_EV, somethingelse = results_EV[i].calculate_smallerthan()
    '''
    for res in results_perc_OM:
        if res < 0.5:
            print("this file has EV derivatives smaller than 1:")
            print( results_OM[i].filename)
    '''
    
    '''
    dZ_percentages_CM.append(results_CM[i].dZ_perc)
    dR_percentages_CM.append(results_CM[i].dR_perc)
    dZdZ_percentages_CM.append(results_CM[i].dZdZ_perc)
    dRdR_percentages_CM.append(results_CM[i].dRdR_perc)
    dZdR_percentages_CM.append(results_CM[i].dZdR_perc)
    '''

    dZ_percentages_EV.append(results_OM[i].dZ_perc)
    dR_percentages_EV.append(results_OM[i].dR_perc)
    dZdZ_percentages_EV.append(results_OM[i].dZdZ_perc)
    dRdR_percentages_EV.append(results_OM[i].dRdR_perc)
    dZdR_percentages_EV.append(results_OM[i].dZdR_perc)
    

#C)
# create list of data that suits our plot_derivatives.plot_percentage_zeroEV function
'''
CM_ylist_toplot = [[jnp.asarray(dZ_percentages_CM), "CM dZ"],[jnp.asarray(dR_percentages_CM), "CM dR"],[jnp.asarray(dRdR_percentages_CM), "CM dRdR"] ,[jnp.asarray(dZdR_percentages_CM), "CM dZdR"], [jnp.asarray(dZdZ_percentages_CM), "CM dZdZ"]]

CM_EV_ylist_toplot = [[jnp.asarray(dZ_percentages_EV), "EVCM dZ"],[jnp.asarray(dR_percentages_EV), "EVCM dR"],[jnp.asarray(dRdR_percentages_EV), "EVCM dRdR"] ,[jnp.asarray(dZdR_percentages_EV), "EVCM dZdR"], [jnp.asarray(dZdZ_percentages_EV), "EVCM dZdZ"]]

CM_ylist_toplot.extend(CM_EV_ylist_toplot)
'''

OM_ylist_toplot = [[jnp.asarray(dZ_percentages_OM), "OM dZ"],\
        [jnp.asarray(dR_percentages_OM), "OM dR"],\
        [jnp.asarray(dRdR_percentages_OM), "OM dRdR"] ,\
        [jnp.asarray(dZdR_percentages_OM), "OM dZdR"],\
        [jnp.asarray(dZdZ_percentages_OM), "OM dZdZ"]]


#C)
#plot and save all datapoints in one and in multiple panels
pltder.plot_percentage_zeroEV(jnp.asarray(norms), CM_ylist_toplot,\
        "Nonzero Values of CM and CMEV Derivatives", "./Images/perc_nonzero_CM_CMEV_panel", True, representations = 2)
pltder.plot_percentage_zeroEV(jnp.asarray(norms), CM_ylist_toplot,\
        "Nonzero Values of CM and EVCM Derivatives", "./Images/perc_CM_CMEV_oneplot", False, representations = 2)

            

    

