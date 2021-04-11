import jax_derivative as jder
import qml
import numpy as np
import plot_derivative as pltder
import jax_representation as jrep
import numerical_derivative as numder
import representation_ZRN as ZRNrep
import database_preparation as datprep
import jax_additional_derivative as jader

#what to do?
do_fingerprint_distance = False
do_derivative_calculation = False
do_plot_derivatives = True

#path to xyz files
database = "/home/linux-miriam/Databases/BOB/"

'''define folder of .xyz files'''
names = [database + "BOB1.xyz", database + "BOB2.xyz"]

#which representations?
namelist = ["CM", "EVCM", "BOB", "OM", "EVOM"]



compound1 = qml.Compound(names[0])
compound2 = qml.Compound(names[1])


if do_fingerprint_distance:

    Z1 = compound1.nuclear_charges.astype(float)
    R1 = compound1.coordinates
    
    Z2 = compound2.nuclear_charges.astype(float)
    R2 = compound2.coordinates
    
    #calculate difference to reference constitution
    M_CM1 = jrep.CM_full_unsorted_matrix(Z1, R1, size = 4)
    M_CM2 = jrep.CM_full_unsorted_matrix(Z2, R2, size=4)

    M_EVCM1 = jrep.CM_ev_unsrt(Z1, R1, N = 0, size = 4)
    M_EVCM2 = jrep.CM_ev_unsrt(Z2, R2, size = 4)

    M_BOB1 = np.asarray(ZRNrep.Bag_of_Bonds(Z1, R1))
    M_BOB2 = np.asarray(ZRNrep.Bag_of_Bonds(Z2, R2))

    M_OM1 = jrep.OM_full_unsorted_matrix(Z1, R1)
    M_OM2 = jrep.OM_full_unsorted_matrix(Z2, R2)

    M_EVOM1 = jrep.OM_ev(Z1, R1)[0]
    M_EVOM2 = jrep.OM_ev(Z2, R2)[0]

    diff1 = (M_EVCM1 - M_EVCM2)
    diff2 = (M_BOB1 - M_BOB2)
    diff3 = (M_CM1.flatten() - M_CM2.flatten())
    diff4 = (M_OM1.flatten() - M_OM2.flatten())
    diff5 = (M_EVOM1 - M_EVOM2)

    CM_error = 0
    EVCM_error = 0
    BOB_error = 0
    OM_error = 0
    EVOM_error = 0

    for d in diff1:
        EVCM_error += d*d

    for d in diff2:
        BOB_error += d*d

    for d in diff3:
        CM_error += d*d

    for d in diff4:
        OM_error += d*d

    for d in diff5:
        EVOM_error += d*d




    print("Difference between Compounds by representation:")
    print("BOB", BOB_error)
    print("CM", CM_error)
    print("EVCM", EVCM_error)
    print("OM", OM_error)
    print("EVOM", EVOM_error)


if do_derivative_calculation:
    #create results instances
    mols, compounds = datprep.read_xyz_energies(database)

    datprep.store_compounds(compounds, database+"compounds.pickle")

    #prepare derivatives of all representations
    replist = [ZRNrep.Coulomb_Matrix, ZRNrep.Eigenvalue_Coulomb_Matrix, ZRNrep.Bag_of_Bonds, ZRNrep.Overlap_Matrix, \
        ZRNrep.Eigenvalue_Overlap_Matrix]

    
    for i in range(5):
        results, resultaddition = jader.calculate_num_der(replist[i], compounds)
        res_file = database + namelist[i] + "der_results.pickle"
        datprep.store_compounds(results, res_file)
        print("results were successfully stored to ", res_file)


if do_plot_derivatives:
    yvals = []
    norms_nuc = []

    for i in range(5):
        resfile = database + namelist[i] + "der_results.pickle"

        res_list = datprep.read_compounds(resfile)

        xlist, ylist, results = \
                pltder.prepresults(results = res_list,\
                rep = namelist[i],\
                repno = i,\
                yval = "perc",\
                with_whichd = True)

        yvals.extend(ylist)
        for i in ylist:
            norms_nuc.append(xlist)

    print(yvals)

    pltder.plot_percentage_zeroEV([1,2], yvals, title = "4 C Atoms",\
            savetofile = "./Images/Final/BOB_conformations.png", oneplot = False,\
            representations = [0,1,2,3,4],\
            xaxis_title = "Compound",\
            Include_Title = False,\
            BOB = True)
