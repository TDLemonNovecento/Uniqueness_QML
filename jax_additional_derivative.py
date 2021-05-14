'''
This file contains functions related to the analytical derivatives contained in jax_derivative
They either print derivatives (cal_print_1stder, cal_pring_2ndder) or interact with the
database_preparation objects by adding derivative results to them (calculate_eigenvalues, 
update_index).

'''
import numpy as np
from jax import ops
import jax_representation as jrep
import jax.numpy as jnp
import database_preparation as datprep
import jax_derivative as jder
import numerical_derivative as numder
from jax.config import config
config.update("jax_enable_x64", True) #increase precision from float32 to float64
import representation_ZRN as ZRNrep

def presort(Z_orig, R_orig, order):
    
    Z = Z_orig[order.tolist()]
    R = R_orig[order]

    R = np.asarray(R, dtype = np.float64)
    Z = np.asarray(Z, dtype = np.float64)
    return(Z, R)

def calculate_num_der(repro, compoundlist, matrix = False, do_dZ = True, do_dR = True):
    '''calculates eigenvalues of derived matrices from compounds
    only functions as translator between the compound object, the derivative_result object
    and the sorted_derivative function.

    Arguments:
    ----------
    repro: callable representation function that takes Z, R as first two args
            all functions in "representations_ZRN.py" work
    compoundlist: list of database_preparation.compound objects
    matrix: If repro returns matrix like shape, unravel
    do_dZ : boolean,
            if true, do dZ derivative
    do_dR : boolean,
            if true, do dR derivative

    Returns:
    --------
    resultlist: list of derivative_result instances, a class
                which contains both norm as well as derivatives,
                eigenvalues and fractual eigenvalue information
    results: list of fractions of nonzero eigenvalues,
            structure: [[compound 1: dZ_ev, dR_ev, ...], [compound 2: ...],...]
    '''
    resultlist = []
    results = []
    
    print("Your representation is ", repro, "this is a matrix is set to: ", matrix)
    print("if this is wrong, change in function 'calculate_num_der' in file jax_additional_derivative.py")
     
    #extract atomic data from compound
    for c in compoundlist:
        Z_orig = np.asarray([float(i)for i in c.Z])
        R_orig = np.asarray(c.R)
        N = float(c.N)
        

        #calculate derivatives and representation
        #M needed to calculate norm for molecule, here always using CM matrix nuclear norm
        #order needed to preorder molecules (numerical derivative)
        
        dim = len(Z_orig)
        M, order = jrep.CM_full_sorted(Z_orig, R_orig, N, size = dim)
        
        #preorder Z and R as numerical derivation may be disturbed by sorted representations
        Z, R = presort(Z_orig, R_orig, np.asarray(order))
        

        dZ = [[0] for i in range(dim)]
        dR = [[[0] for j in range(3)] for i in range(dim)]
        dZdZ = [[[0] for j in range(dim)] for i in range(dim)]
        dRdR = [[[[[0] for l in range(3)] for k in range(dim)] for j in range(3)] for i in range(dim)]
        dZdR = [[[[0] for k in range(3)] for j in range(dim)] for i in range(dim)]
        
        print("checkpoint2: now starting numerical differentiation, jax_additional line 82")
        for i in range(dim):
            if do_dZ:
                try:
                    dZ[i] = numder.derivative(repro, [Z, R, N], order = 1, d1 = [0, i])
                    print("subcheck2: dZ derivative calculated")
                except TypeError:
                    print("this representation cannot be derived by dZ")
            for j in range(3):
                if do_dR:
                    dR[i][j] = numder.derivative(repro, [Z, R, N], order = 1, d1 = [1, i, j])

                for k in range(dim):
                    if do_dZ:
                        try:
                            dZdR[i][k][j] = numder.derivative(repro, [Z, R, N], order = 2, d1 = [0, i], d2 = [1, k, j])
                        except TypeError:
                            print("this representation cannot be derived by dZdR")
                    if do_dR:
                        for l in range(3):
                            dRdR[i][j][k][l] = numder.derivative(repro, [Z, R, N], order = 2, d1 = [1, i, j], d2 = [1, k,l])
            
            for m in range(dim):
                if do_dZ:
                    dZdZ[i][m] = numder.derivative(repro, [Z,R,N], order = 2, d1 = [0,i], d2 = [0, m])
            
        print("all derivatives were calculated successfully for compound ", c.filename)
        #create derivative results instance
        der_result = datprep.derivative_results(c.filename, Z, M)

        #get all derivative eigenvalues for the derivatives and add to results instance
        der_result.add_all_RZev(np.asarray(dZ), np.asarray(dR), np.asarray(dZdZ),\
                np.asarray(dRdR), np.asarray(dZdR))

        #calculate percentile results and add to results
        res, addition = der_result.calculate_smallerthan()

        results.append(res)
        resultlist.append(der_result)

    return(resultlist, results)


'''
The following functions are for calling all derivatives and printing or processing them one by one
'''
def calculate_eigenvalues(repro, compoundlist):
    '''calculates eigenvalues of derived matrices from compounds
    only functions as translator between the compound object, the derivative_result object
    and the sorted_derivative function.

    Arguments:
    ----------
    repro: representation, such as 'CM' for coulomb matrix ect, as is used in sorted_derivative function
    compoundlist: list of database_preparation.compound objects

    Returns:
    --------
    resultlist: list of derivative_result instances, a class
                which contains both norm as well as derivatives,
                eigenvalues and fractual eigenvalue information
    results: list of fractions of nonzero eigenvalues,
            structure: [[compound 1: dZ_ev, dR_ev, ...], [compound 2: ...],...]
    '''
    resultlist = []
    results = []
    #extract atomic data from compound
    for c in compoundlist:
        Z = jnp.asarray([float(i)for i in c.Z])
        R = c.R
        N = float(c.N)

        #calculate derivatives and representation
        #M needed to calculate norm for molecule, here always using CM matrix nuclear norm
        M, order = jrep.CM_full_sorted(Z, R, N)
        #dim = M.shape[0]

        dZ = jder.sort_derivative(repro, Z, R, N, 1, 'Z', M, order)
        dR = jder.sort_derivative(repro, Z, R, N, 1, 'R', M, order)

        ddZ = jder.sort_derivative(repro, Z, R, N, 2, 'Z', 'Z', M, order)
        dZdR = jder.sort_derivative(repro, Z, R, N, 2, 'R', 'Z', M, order)
        ddR = jder.sort_derivative(repro, Z, R, N, 2, 'R', 'R', M, order)

        print("all derivatives were calculated successfully for compound ", c.filename)
        #create derivative results instance
        der_result = datprep.derivative_results(c.filename, Z, M)
        
        #get all derivative eigenvalues for the derivatives and add to results instance
        der_result.add_all_RZev(dZ, dR, ddZ, ddR, dZdR)

        #calculate percentile results and add to results
        res = der_result.calculate_percentage()

        results.append(res)
        resultlist.append(der_result)

    return(resultlist, results)


def cal_print_1stder(repro, Z, R, N):
    dim = len(Z)
    '''calculates all derivatives and prints them nicely'''
    dZ = jder.sort_derivative(repro, Z, R, N, 1, 'Z')
    dN = jder.sort_derivative(repro, Z, R, N, 1, 'N')
    dR = jder.sort_derivative(repro, Z, R, N, 1, 'R')

    print('first derivatives:\n------------------')
    for i in range(dim): #3 atoms in H2O
        print('dZ%i' % (i+1))
        print(dZ[i])

    xyz_labels = ['x', 'y', 'z']
    for xyz in range(3): #x, y and z
        for i in range(dim): #3 atoms in H2O
            print('d%s%i' % (xyz_labels[xyz], (i+1)))
            print(dR[i][xyz]) #derivatives are unsorted

def cal_print_2ndder(repro, Z, R, N):
    dim = len(Z)
    which_return = [True, True, True]
    matrix = False
    try:
        if(repro  == 'CM' or repro == 'OM'):
            matrix = True
    except IndexError:
        print("your representation has not the shape of a matrix")


    '''calculates all second derivatives'''
    if which_return[0]:
        dZdZ = jder.sort_derivative(repro, Z, R, N, 2, 'Z', 'Z')
        for i in range(dim): #3 atoms in H2O
            for j in range(dim): #second dZ over 3 atoms in H2O
                print('dZ%idZ%i' %(i+1, j+1))
                print(dZdZ[i,j])


    if which_return[1]:
        dRdR = jder.sort_derivative(repro, Z, R, N, 2, 'R', 'R')
        print("dRdR derivatives:")

        xyz = [[0,'x'],[1,'y'],[2,'z']]
        for i in range(dim): #3 atoms in H2O
            for x in xyz:
                for j in range(dim):
                    for y in xyz:
                        print('d%s%id%s%i' %(x[1], i+1, y[1], j+1))
                        print(dRdR[i,x[0], j, y[0]])

    if which_return[2]:
        dZdR = jder.sort_derivative(repro, Z, R, N, 2, 'Z', 'R')

        print("dZdR derivatives:")

        xyz = [[0,'x'],[1,'y'],[2,'z']]
        for i in range(dim): #3 atoms in H2O
            for j in range(dim):
                for x in xyz:
                    print('dZ%id%s%i' %(i+1, x[1], j+1))
                    print(dZdR[i, x[0], j])



def update_index(ZRN, d, h):
    if d[0] == 0:
        new = ops.index_add(ZRN[d[0]], d[1], h)
        new_ZRN = [new, ZRN[1], ZRN[2]]
    elif d[0] == 1:
        new = ops.index_update(ZRN[d[0]], d[1], ops.index_add(ZRN[d[0]][d[1]], d[2], h))
        new_ZRN = [ZRN[0], new, ZRN[2]]
    return(new_ZRN)

def num_first_derivative(f, ZRN, d = [0, 0], h = 0.1, method = 'central', dim =3):
    '''Compute the difference formula for f'(a) with step size h.

    Parameters
    ----------
    f : function
        Vectorized function of one variable
    ZRN : list
        Contains Z, R and N information of where derivative should be taken
    d : list
        contains integers referring to variable w.r.t. which the derivative is taken
        1st place: 0 = dZ, 1 = dR, 2 = dN;
        2nd place: which dZ/dR, e.g. 2 == dZ3, 0 == dR1, ect.;
        3rd place: which dR, 0 = dx, 1 = dy, 2 = dz
    method : string
        Difference formula: 'forward', 'backward' or 'central'
    h : number
        Step size in difference formula

    Returns
    -------
    float
        Difference formula:
            central: f(a+h) - f(a-h))/2h
            forward: f(a+h) - f(a))/h
            backward: f(a) - f(a-h))/h
    '''
    if method == 'central':
        p = update_index(ZRN, d, h)
        m = update_index(ZRN, d, -h)

        fp = f(p[0], p[1], p[2])
        fm = f(m[0], m[1], m[2])

        return( (fp - fm)/(2*h))

    elif method == 'forward':
        p = update_index(ZRN, d, h)

        fp = f(p[0], p[1], p[2])
        fnormal = f(ZRN[0], ZRN[1], ZRN[2])

        return ((fp - fnormal)/h)

    elif method == 'backward':
        m = update_index(ZRN, d, -h)

        fnormal = f(ZRN[0], ZRN[1], ZRN[2])
        fm = f(m[0], m[1], m[2])

        return ((fnormal - fm)/h)
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")

def num_second_derivative(f, ZRN, d1 = [0, 0], d2 = [1, 2, 1], h1 = 0.01, h2 = 0.01, method = 'simplecentral', dim = 3):
    '''Compute the difference formula for f'(a) with step size h.
    works best for h = 0.01 on CM

    Parameters
    ----------
    f : function
        Vectorized function of one variable
    ZRN : list containing Z, R, N
    Z : np array nx1
        nuclear charges
    R : np array nx3
        positions of nuclear charges
    N : float
        total no of electrons, irrelevant
    d1,d2 : list
        first place determines which variable is derived by (0 = dZ, 1 = dR, 2 = dN)
        second place determines which dR (0 = dx, 1 = dy, 2 = dz)
    method : string
        Difference formula: 'forward', 'backward' or 'central'
    h1, h2 : number
        Step size in difference formula

    Returns
    -------
    result of f


        with a being the unaltered ZRN:
        Difference formula for mixed second derivative:
            simplecentral: f(a+h_1, b+h_2) - f(a+h_1, b-h_2) - f(a-h_1, b+h_2) + f(a-h_1, b-h_2))/(4h_1*h_2)

    '''
    if d1[0] == d2[0] and d1[1] == d2[1]:
        p = update_index(ZRN, d1, h1)
        m = update_index(ZRN, d1, -h1)

        fp = f(p[0], p[1], p[2], dim, True)[0]
        fm = f(m[0], m[1], m[2], dim, True)[0]
        fnormal = f(ZRN[0], ZRN[1], ZRN[2], dim, True)[0]

        return((fp - 2*fnormal + fm)/(h1*h1))


    if method == 'simplecentral':
        pp = update_index(update_index(ZRN, d1, h1), d2, h2)
        pm = update_index(update_index(ZRN, d1, h1), d2, -h2)
        mp = update_index(update_index(ZRN, d1, -h1), d2, h2)
        mm = update_index(update_index(ZRN, d1, -h1), d2, -h2)

        fpp = f(pp[0], pp[1], pp[2], dim, True)[0]
        fpm = f(pm[0], pm[1], pm[2], dim, True)[0]
        fmp = f(mp[0], mp[1], mp[2], dim, True)[0]
        fmm = f(mm[0], mm[1], mm[2], dim, True)[0]
        return((fpp - fpm - fmp + fmm)/(4*h1*h2))

    if method == 'felix':
        pp = update_index(update_index(ZRN, d1, h1), d2, h2)
        pm = update_index(update_index(ZRN, d1, h1), d2, -h2)
        mp = update_index(update_index(ZRN, d1, -h1), d2, h2)
        mm = update_index(update_index(ZRN, d1, -h1), d2, -h2)

        p2p = update_index(update_index(ZRN, d1, 2*h1), d2, h2)
        p2m = update_index(update_index(ZRN, d1, 2*h1), d2, -h2)
        m2p = update_index(update_index(ZRN, d1, -2*h1), d2, h2)
        m2m = update_index(update_index(ZRN, d1, -2*h1), d2, -h2)

        pp2 = update_index(update_index(ZRN, d1, h1), d2, 2*h2)
        pm2 = update_index(update_index(ZRN, d1, h1), d2, -2*h2)
        mp2 = update_index(update_index(ZRN, d1, -h1), d2, 2*h2)
        mm2 = update_index(update_index(ZRN, d1, -h1), d2, -2*h2)

        p2p2 = update_index(update_index(ZRN, d1, 2*h1), d2, 2*h2)
        p2m2 = update_index(update_index(ZRN, d1, 2*h1), d2, -2*h2)
        m2p2 = update_index(update_index(ZRN, d1, -2*h1), d2, 2*h2)
        m2m2 = update_index(update_index(ZRN, d1, -2*h1), d2, -2*h2)


        fp2p2 = f(p2p2[0], p2p2[1], p2p2[2], dim, True)[0]
        fp2m2 = f(p2m2[0], p2m2[1], p2m2[2], dim, True)[0]
        fm2p2 = f(m2p2[0], m2p2[1], m2p2[2], dim, True)[0]
        fm2m2 = f(m2m2[0], m2m2[1], m2m2[2], dim, True)[0]

        fpp2 = f(pp2[0], pp2[1], pp2[2], dim, True)[0]
        fpm2 = f(pm2[0], pm2[1], pm2[2], dim, True)[0]
        fmp2 = f(mp2[0], mp2[1], mp2[2], dim, True)[0]
        fmm2 = f(mm2[0], mm2[1], mm2[2], dim, True)[0]

        fp2p = f(p2p[0], p2p[1], p2p[2], dim, True)[0]
        fp2m = f(p2m[0], p2m[1], p2m[2], dim, True)[0]
        fm2p = f(m2p[0], m2p[1], m2p[2], dim, True)[0]
        fm2m = f(m2m[0], m2m[1], m2m[2], dim, True)[0]

        fpp = f(pp[0], pp[1], pp[2], dim, True)[0]
        fpm = f(pm[0], pm[1], pm[2], dim, True)[0]
        fmp = f(mp[0], mp[1], mp[2], dim, True)[0]
        fmm = f(mm[0], mm[1], mm[2], dim, True)[0]

        return((1/(144*h1*h2))*(8*(fpm2 + fp2m + fm2p + fmp2) - 8*(fmm2+fm2m+fpp2+fp2p) - (fp2m2+fm2p2-fm2m2-fp2p2) + 64*(fmm+fpp-fpm-fmp)))




def num_second_pure_derivative(f, ZRN, ZRNplusplus, ZRNplus, ZRNminus, ZRNminusminus, method='central', h1 = 0.1, h2 = 0.1, dim = 3):
    '''Compute the difference formula for f'(a) with step size h.

    Parameters
    ----------
    f : function
        Vectorized function of one variable
    ZRN : list
        compute derivative at Z, R, N as in list ZRN
    ZRNplus, ZRNminus: list
        altered original ZRN list with small changes in the variable
        with respect to which the derivative is taken
        ZRNplus: h is added
        ZRNminus: h is subtracted
    method : string
        Difference formula: 'forward', 'backward' or 'central'
    h : number
        Step size in difference formula

    Returns
    -------
    result of f

    Difference formula for three point second derivative:
            central: (f(a+h) - 2f(a) + f(a-h))/h²
            forward: (f(a) - 2f(a+h) +f(a+2h))/h
            backward: (f(a-2h) - 2f(a-h) + f(a))/h

        five point centered difference:
            central: (-f(a+2h) + 16f(a+h)-30f(a) +16f(a-h) - f(a-2h))/12h²
    '''
    normal = f(ZRN[0], ZRN[1], ZRN[2], dim, True)[0]
    #print("ZRNplus", ZRNplus, "ZRNplusplus", ZRNplusplus)
    if method == 'central':
        plusplus = f(ZRNplusplus[0], ZRNplusplus[1], ZRNplusplus[2], dim, True)[0]
        minusminus = f(ZRNminusminus[0], ZRNminusminus[1], ZRNminusminus[2], dim, True)[0]
        plus = f(ZRNplus[0], ZRNplus[1], ZRNplus[2],dim, True)[0]
        minus = f(ZRNminus[0], ZRNminus[1], ZRNminus[2], dim, True)[0]
        #print("plusplus \n", plusplus, "\nplus\n", plus, "\nminus\n", minus, "\nminusminus\n", minusminus)

        return (-plusplus + 16*plus -30*normal + 16*minus - minusminus)/(12*h1*h2)
    elif method == 'central3':
        plus = f(ZRNplus[0], ZRNplus[1], ZRNplus[2], dim, True)[0]
        minus = f(ZRNminus[0], ZRNminus[1], ZRNminus[2],dim, True)[0]
        return((plus - 2*normal *minus) /( h1*h2))
    elif method == 'forward':
        return (f(a + h) - f(a))/h
    elif method == 'backward':
        return (f(a) - f(a - h))/h
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")

