import os
import qml
import numpy as np
import database_preparation as datprep
import kernel_learning as kler
import plot_kernel as pltker
import representation_ZRN as ZRN_rep
import jax_math as jmath


#how many compounds should be screened?
training_no = [100, 500, 1000, 2000, 3500]

#list of representations to be considered, 0 = CM, 1 = EVCM, 2 = BOB, 3 = OM, 4 = EVOM
representation_list = [1]#, 3]#3, 4]# 2]
rep_names = ["CM", "EVCM", "BOB", "OM", "EVOM"]

#what do you want to do if the same variables were already used for a run?
plot_learning = True #False
plot_scatter = False
calculate_kernel = False #True

#how many are to be predicted?
test_no =100 

#set hyperparameters, sigma is kernel width and lamda variation
sigma = [[2048.0], [64.0], [1024.0], [500.0], [1000.0]] #optimal sigmas for every representation
lamda = [1e-8, 1e-8, 1e-8, 1e-8, 1e-8] #optimal lamdas for every representation

#hyperparameters for screening
sigma[representation_list[0]] = [01.*2**i for i in range(25)][4:]
lamda_list = [1e-6, 1e-8, 1e-10, 1e-12, 1e-14]

#get maximum number of compounds for which representations need to be calculated
total = training_no[-1] + test_no
this_job_name = "%i_%s" %(test_no, str(lamda))
for i in representation_list:
    this_job_name += rep_names[i] + "_%i_" %(len(sigma[i]))
this_job_name += (str(training_no))


filepath_thisjob = "./tmp/Kernel_Results_" + this_job_name
print("filepath:", filepath_thisjob)

if os.path.isfile(filepath_thisjob):
    print("this file already exists")
    
    if plot_scatter:
        #plot scatter plots
        results = datprep.read_compounds(filepath_thisjob)
    
        for i in range(len(training_no)):
        
            name = "%i Training Instances, OM representation" % training_no[i] 
            y_test = results[i].y_test
            y_predicted = results[i].test_predicted_results
        
            pltker.plot_scatter(y_test, y_predicted, title = name, figuretitle = "Scatterplot_OM_%i" %i)
    
    if plot_learning:
        results = datprep.read_compounds(filepath_thisjob)
        print("trying to plot learning from existing file") 
        
        #create CurveObj class list 
        sigma_list = []
        lamda_list = []

        
        for res in results:
            sigma_list.append(res.sigma)
            lamda_list.append(res.lamda)

        #get all indices of unique values in sigma_list
        sigmas = np.array(sigma_list)
        lamdas = np.array(lamda_list)
        
        #for unique sigma/lamda pairs, add both arrays
        sl_value = sigmas+lamdas

        curve_indices = jmath.unique_indices(sigmas+lamdas)
        
        #sort sigma_list by lamda indices:

        curves = []
        
        for c in curve_indices:
            c = np.sort(c)
            print("sorted c:", c)
            co = datprep.CurveObj("lamda = %.2e, sigma = %.1f" % (lamdas[c[0]], sigmas[c[0]]))
            
            for i in c:
                co.xnparray = np.append(co.xnparray , results[i].training_size())
                co.ynparray = np.append(co.ynparray, results[i].mae)
            
            curves.append(co) 
        
        pltker.plot_curves(curves, file_title = "Trial_Automatic_Kernel.png")

    if calculate_kernel:
        print("trying to calculate kernel")
        
    
    else:
        sys.exit()

#make list with all the representations
X_list = [[],[],[],[],[]]
Y_energy_list = []


# Get raw Kernel_Results class list with represented compounds
for rep in representation_list:
    kernel_class_path = "./Pickled/Represented_Compounds_Kernel/%s_raw_Kernel_Results" % rep_names[rep]
    
    raw_representation = datprep.read_compounds(kernel_class_path)

    #add representation array to X_list
    X_list[rep] = raw_representation[0].x
    Y_energy_list = raw_representation[0].y


print("len Y_energy_list:", len(Y_energy_list))


'''prepare training and test sets '''

training_list = []
test_list = []

for no in training_no:
    #randomly divide data into training and test set
    training, test = kler.make_training_test(total, training_size = no, upperlim = no + test_no)
    
    training_list.append(training)
    test_list.append(test)


X_training = [[],[],[],[],[]]
X_test = [[],[],[],[],[]]

Y_training = []
Y_test = []

#form training and test arrays
for no in range(len(training_no)):
    
    Y_training.append(np.array([Y_energy_list[t] for t in training_list[no]]))
    Y_test.append(np.array([Y_energy_list[t] for t in test_list[no]]))

    for rep in representation_list:
        #make 2D arrays
        X_training[rep].append(np.array([X_list[rep][t] for t in training_list[no]]))
        X_test[rep].append(np.array([X_list[rep][t] for t in test_list[no]]))



'''create Kernel, run learning and store results '''
maes = [[],[],[],[],[]] #maes for each representation (CM, EVCM, BOB, OM, EVOM)


#create list with results
learning_list = []


#do learning
for rep in representation_list:
    for no in range(len(training_no)):
        for s in sigma[rep]:
            #for l in [1]: 
            for l in lamda_list:
                print("l:", l, "s:", s)
                #create new Kernel_Result object
                m = datprep.Kernel_Result()
                #for finding optimal sigma/lamda
                m.lamda = l
                m.sigma = s
                #for final run:
                #m.sigma = sigma[rep][0]
                #m.lamda = lamda[rep] 

                #further info
                m.representation_name = rep_names[rep]
                m.x_training = X_training[rep][no]
                m.x_test = X_test[rep][no]
                m.y_training = Y_training[no]
                m.y_test = Y_test[no]

                m.do_qml_gaussian_kernel()
        
                #save results
                learning_list.append(m)
                maes[rep].append(m.mae)


datprep.store_compounds(learning_list, filepath_thisjob)


#plot results
plottable_maes = [maes[i] for i in representation_list]
labels = [rep_names[i] for i in representation_list]

pltker.plot_learning(set_sizes = training_no, maes = plottable_maes, labels = labels) 
print("sigmas:")
print(sigma)
print("maes:")
for i in representation_list:
    print("representation: ", rep_names[i])
    print(maes[i])
