import os
import qml
import numpy as np
import database_preparation as datprep
import kernel_learning as kler
import plot_kernel as pltker
import representation_ZRN as ZRN_rep
import jax_math as jmath
from copy import deepcopy
from qml.math import cho_solve


#how many compounds should be screened?
training_no = [100, 500, 1000, 2000, 3500]
test_no = 10

sigma_grid = np.array([0])
lamda_grid = np.logspace(-11, -11, 1)

#list of representations to be considered, 0 = CM, 1 = EVCM, 2 = BOB, 3 = OM, 4 = EVOM
representation_list = [0, 1, 2, 3, 4]#, 3]#3, 4]# 2]
rep_names = ["CM", "EVCM", "BOB", "OM", "EVOM"]

#what do you want to do if the same variables were already used for a run?
plot_learning = False
plot_scatter = False
calculate_kernel = True

#set hyperparameters, sigma is kernel width and lamda variation
sigma_list = [[2400.0], [80.0], [1700.0], [2600.0], [8.0]] #optimal sigmas for every representation
lamda_list = [1e-11, 1e-8, 1e-11, 1e-11, 1e-5] #optimal lamdas for every representation


#get maximum number of compounds for which representations need to be calculated

final_file_list = ['CM_QM7', \
        'EVCM_QM7', \
        'BOB_QM7',\
        'OM_QM7',\
        'EVOM_QM7']

final_curve_file = "./tmp/Curves.pickle"
filepath_thisjob = "./tmp/trial"
    
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
        
    curve_list = []
    learning_list = []#
    

    #make list with all the representations
    X_list = [[],[],[],[],[]]
    Y_energy_list = []

    maes = [[],[],[],[],[]] #maes for each representation (CM, EVCM, BOB, OM, EVOM)

    nModels = test_no #10 #how many times do you want to repeat?

    # Get raw Kernel_Results class list with represented compounds
    for rep in representation_list:
        kernel_class_path = "./Pickled/Represented_Compounds_Kernel/%s_raw_Kernel_Results" % rep_names[rep]
    
        raw_representation = datprep.read_compounds(kernel_class_path)
        
        #add representation array to X_list
        X_list[rep] = np.array(raw_representation[0].x)
        Y_energy_list = np.array(raw_representation[0].y)

        dim = len(Y_energy_list)

        print("len Y_energy_list:", len(Y_energy_list))


        '''create Kernel, run learning and store results '''

        
        for i, sigma in enumerate(sigma_grid):
            
            #create new Kernel_Result object
            m = datprep.Kernel_Result()
            m.sigma = sigma_list[rep]
            #m.sigma = sigma
            m.representation_name = rep_names[rep]

            K = m.laplacian_kernel_matrix(X_list[rep], X_list[rep])
            
            for j, lamda in enumerate(lamda_grid):
                print("lamda, sigma:", lamda, sigma)
                lamda = lamda_list[rep]
                mae_list = []
                for number in training_no: 
                    
                    total_tested = 0
                    total_training = 0
                    
                    mae_nmodels = 0

                    for i in range(nModels):
                        
                        #copy Kernel_Results class object
                        m_c = deepcopy(m)
                                            
                        #make training and test indices
                        training_indices, test_indices = kler.make_training_test(dim = dim,\
                            training_size = number, upperlim = dim)

                        total_tested += len(test_indices)
                        total_training += len(training_indices)
                        
                        #copy relevant rows&columns from K for learning
                        C = deepcopy(K[training_indices][:,training_indices])
                        
                        #add slight alteration lambda

                        C[np.diag_indices_from(C)] += lamda

                        #further info
                        m_c.x_training = X_list[rep][training_indices]
                        m_c.x_test = X_list[rep][test_indices]
                    
                        m_c.y_training = Y_energy_list[training_indices]
                        m_c.y_test = Y_energy_list[test_indices]
                        
                        #solve for alphas
                        alphas = cho_solve(C, m_c.y_training)
                        
                        K_test = m_c.laplacian_kernel_matrix(x_training = m_c.x_training, x_test = m_c. x_test)

                        m_c.test_predicted_results = np.dot(K_test, alphas)
                        
                        mae = m_c.calculate_mae()
                        
                        mae_nmodels += mae

               
                    print("mae_nmodels ", mae_nmodels)
                    avg_mae = mae_nmodels / float(nModels)
                    print("avg mae:", avg_mae)
                    m_c.mae = avg_mae
                    learning_list.append(m_c)

                    mae_list.append(avg_mae)

                    print("totally tested:", total_tested)

                
                name = rep_names[rep]#"sigma: %.2e, lambda: %.2e" %(sigma, lamda)
                curve = datprep.CurveObj(name)
                curve.xnparray = training_no
                curve.ynparray = np.array(mae_list)
                curve_list.append(curve)
    
        final_file = "./tmp/Kernel_Results/" + final_file_list[rep]
         

    datprep.store_compounds(curve_list, final_curve_file)
    
