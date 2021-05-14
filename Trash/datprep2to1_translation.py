import database_preparation as datprep
import database_preparation2 as datprep2
import numpy as np
import shutil

pickled_qm7 = "/home/linux-miriam/Databases/Pickled/qm7.pickle" 

'''below are lists of pickled results files or their partial paths'''
results_file_list =["/home/linux-miriam/Databases/Pickled/qm7_CM_results.pickle",\
        "/home/linux-miriam/Databases/Pickled/qm7_CM_EV_results.pickle",\
        "/home/linux-miriam/Databases/Pickled/BOB_unsorted_rep/BOB_numder_res",\
        "/home/linux-miriam/Databases/Pickled/OM_numder_res",\
        "/home/linux-miriam/Databases/Pickled/EVOM_numder_res",\
        "/home/linux-miriam/Databases/Pickled/BOB2_numder_res100-800",\
        "/home/linux-miriam/Databases/Pickled/BOB_sorted_rep/BoB_numder_res",\
        "./Pickled/fourcompounds_EV_results.pickle",\
        "./Pickled/trial_numder.pickle"]


repno = 2
repname_list = ["CM", "EVCM", "BOB", "OM", "EVOM"]
repname = repname_list[repno]

how_count = "perc"

qm7_xyz_path = "/home/linux-miriam/Databases/QM7_XYZ/"
special_compounds = "/home/linux-miriam/Databases/Special/"

tag = "%s_dZdZ_d_norm_%s" %(repname, how_count)

results_file = results_file_list[repno]


if repno < 2:
    results_list = datprep.read_compounds(results_file)

elif repno == 2:
    results_list = []
    bob_numbers = ["100-120", "120-140", "140-160", "160-180", "180-200",\
        "220-240", "240-260", "260-280", "280-300",\
        "300-320", "320-340", "340-360", "360-380", "380-400",\
        "400-420", "420-440", "440-460", "460-480", "480-500",\
        "520-540", "540-560", "560-580", "580-600",\
        "600-620", "620-640", "640-660", "660-680", "680-700",\
        "700-720", "720-740", "740-760", "760-780", "780-800",\
        "800-820", "820-840", "840-860", "880-900",\
        "920-940", "940-960", "980-1000", "1000-1020"]
    for i in range(len(bob_numbers)):
        this_res_file = results_file + bob_numbers[i]
        results_list.extend(datprep2.read_compounds(this_res_file))

else:
    results_list = []
    for i in range(400, 800, 100):
        results = datprep.read_compounds(results_file + "%i-%i" %(i, i+100))
        print("i:", i)
        results_list.extend(results)
        print(results_file)
        del(results)

total_res_list = []
for result in results_list:
    
    result.calculate_smallerthan(repro = 2) 
    
    #search criteria
    #if len(result.Z) < 6 and result.dR_perc > 0.2:
    if result.dZdZ_perc > 0.4 and result.norm > 150:
        #print("less atoms than 5:")
        print(result.filename, len(result.Z))
        print("dR percentage:")
        print(result.dZ_perc)
        print("norm:", result.norm)
        print("absolute:", result.dR_perc * result.calculate_dim(repno))

        filename = result.filename
        #now copy file
        original = qm7_xyz_path + filename
        target = special_compounds + tag + filename 
    
        shutil.copyfile(original, target)
        
        print("file was found")
