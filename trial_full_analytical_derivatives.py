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


#if only some little part of the dataset should be used, define as input variables
try:
	init, end = int(sys.argv[1]), int(sys.argv[2])
except IndexError:
	init = int(input("starting point"))
	end = int(input("end point"))


###B)
'''
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
'''

###A)
###define path to folder containing xyz files. All files are considered.
database = "/home/miriam/Databases/QM9_XYZ/"
database_file = "./Pickled/qm9.pickle"
dat_ha_file = "./Pickled/qm7.pickle"
result_folder = "./Pickled/"
result_file = "./Pickled/results.pickle"
'''
#If you want to plot from multiple pickle results file, use this code:
result_file = "/home/linux-miriam/Uniqueness_QML/Pickled/results" 
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
full_compound_ls = datprep.read_compounds(dat_ha_file)
print(len(full_compound_ls), " compounds in full data file")

###B)

#If you want to plot only part of all compounds, use this code:
try:
	compound_ls = full_compound_ls[init : end]
except IndexError:
	print("Your indices were out of bound, restart. min: 0, max: ", len(full_compound_ls))
	exit()



#print(len(compound_ls), " of which are being processed")

#define path to folder containing xyz files. All files are considered.
database = "/home/miriam/Databases/QM9_XYZ/"
database_file = "/home/miriam/Uniqueness_QML/Pickled/qm9.pickle"
dat_ha_file = "/home/miriam/Uniqueness_QML/Pickled/qm7_CM_EV.pickle"
result_folder = "/home/miriam/Uniqueness_QML/Pickled/"

###B)
###create new list of results from list of compounds
results = jader.calculate_eigenvalues('CM_EV', compound_ls)

###B)
###store list of results in result_file
'''
#If you want to plot from multiple pickle results file, use this code:
result_file = result_folder + "results_%i-%i.pickle" %(init, end)
'''
datprep.store_compounds(results, result_file)


#C) 
'''
#If you want to plot from multiple pickle results file, use this code:
#these are used as file identifiers for the results_%i-%i.pickle files
numbers = ["0-166", "166-332", "332-498", "498-664", "664-830", "830-996", "996-1162", "1162-1328", "1328-1494", "1494-1660", "1660-1826", "1494-1660", "1660-1826", "1826-1992", "1992-2158", "2158-2324", "2324-2490", "2490-2656", "2656-2822", "2822-2988", "2988-3154", "3154-3320", "3320-3486", "3486-3652", "3652-3818", "3818-3993"] 

#read list of compounds from data file
full_compound_ls = datprep.read_compounds(dat_ha_file)
print(len(full_compound_ls), " compounds in full data file")
try:
	compound_ls = full_compound_ls[init : end]
except IndexError:
	print("Your indices were out of bound, restart. min: 0, max: ", len(full_compound_ls))
	exit()

print(len(compound_ls), " of which are being processed")

#we'll be reading the results anew.
#This part of the program should therefore be executed after the top part
#has been executed for all the files which can then be called by referencing to the "numbers" list

results = []

#store list of results in result_file
result_file = result_folder + "results_%i-%i.pickle" %(init, end)
datprep.store_compounds(results, result_file)

for n in numbers:
    filename = result_file + "_" + n + ".pickle"
    compoundlist = datprep.read_compounds(filename)
    #print("number of compounds in sublist: ", len(compoundlist))
    
    results.extend(compoundlist)

#print("number of compounds: ", len(compounds))
resultfile = result_file + ".pickle"
datprep.store_compounds(compounds, resultfile)

#data has now been stored to resultfile
'''

#C)
#read data from result file
'''
#If you want to plot from multiple pickle results file, use this code:
result_file = resultfile
'''
results = datprep.read_compounds(result_file)[0]

#C)
#prepare plotting
#y-axis information
dZ_percentages = []
dR_percentages = []
dZdZ_percentages = []
dRdR_percentages = []
dZdR_percentages = []

#x-axis information
norms = []

#C)
#get all the data from our results list
for i in range(len(results)):
    print("type of rsults[i]", type(results[i]))
    print(results[i])
    norms.append(results[i].norm)
    results_perc = results[i].calculate_percentage()
    dZ_percentages.append(results[i].dZ_perc)
    dR_percentages.append(results[i].dR_perc)
    dZdZ_percentages.append(results[i].dZdZ_perc)
    dRdR_percentages.append(results[i].dRdR_perc)
    dZdR_percentages.append(results[i].dZdR_perc)

    #some results were off, check for filenames
    if results[i].dZdR_perc > 0.8 or results[i].dZ_perc > 0.8:
        print(results[i].filename)

#C)
# create list of data that suits our plot_derivatives.plot_percentage_zeroEV function
ylist_toplot = [[jnp.asarray(dZ_percentages), "dZ"],[jnp.asarray(dR_percentages), "dR"],[jnp.asarray(dRdR_percentages), "dRdR"] ,[jnp.asarray(dZdR_percentages), "dZdR"], [jnp.asarray(dZdZ_percentages), "dZdZ"]]

#C)
#plot and save all datapoints in one and in multiple panels
pltder.plot_percentage_zeroEV(jnp.asarray(norms), ylist_toplot, "Nonzero Values of CM Eigenvalue Derivatives", "./Images/perc_nonzeroEV_CM_ev_one", True)
pltder.plot_percentage_zeroEV(jnp.asarray(norms), ylist_toplot, "Nonzero Values of CM Eigenvalue Derivatives", "./Images/perc_nonzero_CM_ev_panel", False)

            

    

