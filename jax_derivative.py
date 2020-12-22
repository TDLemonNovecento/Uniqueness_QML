#here we calculate the first and second derivajtives to the basis of chemical space depending on the chosen representation and placeholder. Matrix and vector reconstruction may be included, too.
import numpy as np
import jax.numpy as jnp
import jax_derivative as jder
import jax_representation as jrep
from jax import grad, jacfwd, jacrev, ops
import time
import database_preparation as datprep
from jax.config import config
config.update("jax_enable_x64", True) #increase precision from float32 to float64


def sort_derivative(representation, Z, R, N = 0, grad = 1, dx = "Z", ddx = "R", M = None, order = None):
    '''Easy function to handle no, one dimensional or two dimensional derivatives with grad. Issue right now: no additional arguments can be passed to function, it therefore falls back to default for any further arguments beside Z, R and N.
    Parameters
    ----------
    fun : callable function depending on (Z, R, N, ...) args
    Z : 1 x n dimensional array
        contains nuclear charges
    R : 3 x n dimensional array
        contains nuclear positions
    N : float
        number of electrons in system
        here: meaningless, can remain empty
    grad: int
        degree by which to differentiate
    dx: string
        argument by which to differentiate first time
    ddx: string
        argument by which to diff. second time
    Return
    ------
    some form of array depending on degree of differentiation
    '''

    #first, find out which representation was chosen and get appropriate function
    ''' idea: the whole code might be more efficient if representation and order were passed on
    instead of calculated freshly at every derivative
    '''
    fn_list = {'CM': jrep.CM_full_sorted, 'CM_EV': jrep.CM_ev, 'OM' : jrep.OM_full_sorted}
    dfn_list = {'CM': d_CM, 'CM_EV' : d_CM_ev, 'OM' : d_OM, 'OM_EV' : d_OM_ev}
    ddfn_list = {'CM': dd_CM, 'CM_EV' : dd_CM_ev, 'OM' : dd_OM, 'OM_EV' : dd_OM_ev}
    

    try:
        fn = fn_list[representation]
    except ValueError:
        fn = fn_list['CM']
        print("your representation was not found. falling back to 'CM'")
    
    


    if grad == 0:
        #print("calculating the representation itself")
        if M == None:
            return( fn(Z, R, N)[0])
        else:
            return(M)
    
    #get correct derivative in jax_derivative function depending on which data should be derived by
    diff_list = {'Z' : 0, 'R' : 1, 'N' : 2}
    try:
        dx_index = diff_list[dx]
    except ValueError:
        dx_index = diff_list['Z']
        print("your dx value cannot be derived by. falling back to 'Z'")

    if grad == 1:#first derivative is calculated
        #print("calculating the first derivative of the representation")
        try:
            d_fn = dfn_list[representation]
        except ValueError:
            d_fn = dfn_list['CM']
            print("your representation was not found. falling back to 'CM' for first derivative")
        
        return(d_fn(Z, R, N, dx_index))
    

    #grad is 2 or bigger, second derivative is calculated
    #print("calculating the second derivative of the representation")
    try: #  get derivation function
        dd_fn = ddfn_list[representation]
    except ValueError:
        dd_fn = ddfn_list['CM']
        print("your representation was not found. falling back to 'CM' for second derivative")

    try: #get second derivative index
        ddx_index = diff_list[ddx]
    except ValueError:
        ddx_index = diff_list['Z']
        print("your ddx value cannot be derived by. falling back to 'Z'")
    

    return(dd_fn(Z, R, N, dx_index, ddx_index))





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

        dZ = sort_derivative(repro, Z, R, N, 1, 'Z', M, order)
        dR = sort_derivative(repro, Z, R, N, 1, 'R', M, order)
        
        ddZ = sort_derivative(repro, Z, R, N, 2, 'Z', 'Z', M, order)
        dZdR = sort_derivative(repro, Z, R, N, 2, 'R', 'Z', M, order)
        ddR = sort_derivative(repro, Z, R, N, 2, 'R', 'R', M, order)
        
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
        dZdZ = sort_derivative(repro, Z, R, N, 2, 'Z', 'Z')
        for i in range(dim): #3 atoms in H2O
            for j in range(dim): #second dZ over 3 atoms in H2O
                print('dZ%idZ%i' %(i+1, j+1))
                print(dZdZ[i,j])


    if which_return[1]:
        dRdR = sort_derivative(repro, Z, R, N, 2, 'R', 'R')
        print("dRdR derivatives:")

        xyz = [[0,'x'],[1,'y'],[2,'z']]
        for i in range(dim): #3 atoms in H2O
            for x in xyz:
                for j in range(dim):
                    for y in xyz:
                        print('d%s%id%s%i' %(x[1], i+1, y[1], j+1))
                        print(dRdR[i,x[0], j, y[0]])
    
    if which_return[2]:
        dZdR = sort_derivative(repro, Z, R, N, 2, 'Z', 'R')

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

        fp = f(p[0], p[1], p[2], dim, True)[0]
        fm = f(m[0], m[1], m[2], dim, True)[0]

        return( (fp - fm)/(2*h))
    
    elif method == 'forward':
        p = update_index(ZRN, d, h)

        fp = f(p[0], p[1], p[2], dim, True)[0]
        fnormal = f(ZRN[0], ZRN[1], ZRN[2], dim, True)[0]

        return ((fp - fnormal)/h)
    
    elif method == 'backward':
        m = update_index(ZRN, d, -h)

        fnormal = f(ZRN[0], ZRN[1], ZRN[2], dim, True)[0]
        fm = f(m[0], m[1], m[2])[0]

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


'''Below follow function specific derivatives with corresponding sorting'''

def d_CM(Z, R, N, dx_index):
    '''this function calculates the derivatives of the sorted Coulomb matrix
    variables:
    ----------
    Z: jnp array, nuclear charges, unsorted
    R: jnp array of 3dim arrays, xyz coordinates, unsorted
    N: total electronic charges (irrelevant, for derivative necessary)
    dx_index: int, either 0, 1, or 2, if Z, R or by N should be derived

    returns:
    --------
    dCM*: jnp array, contains sorted derivatives.
        i.e. dZidZj can be retrieved by dCM[i,j]
        dxidyj by dCM[i,x,j,y]
        
    '''
    #print("calculating sorted second derivative of Coulomb Matrix")

    fM_sorted, order = jrep.CM_full_sorted(Z, R, N) #get order of sorted representation
    
    dim = len(order)
    #direct derivative as jacobian
    dCM = jacfwd(jrep.CM_full_sorted, dx_index)
    reference_dCM = dCM(Z, R, N)[0]
    
    '''reordering is correct, but signs unclear, check if values of EV are important
    '''
    if(dx_index == 0): #derivative by dZ
        #something does not work in this part
        #if the matrix was padded the derivative is weird
        #assign derivative to correct field in sorted matrix by reordering derZ_ij to derZ_kl
        dCMkl_dZkl = jnp.asarray([[[reference_dCM[l][k][m] for l in range(dim)] for k in range(dim)]for m in order])
        return(dCMkl_dZkl)
    elif(dx_index == 1):
        #unordered derivative taken from sorted matrix
        dCMkl_dRkl = jnp.asarray([[[[reference_dCM[l][k][m][x] for l in range(dim)] for k in range(dim)] for x in range(3)] for m in order])
        return(dCMkl_dRkl)
    else:
        return(reference_dCM)
    '''
    return(reference_dCM)
    '''

def d_CM_ev(Z, R, N, dx_index):
    '''Calculates first derivative of CM_ev w.r.t. dx_index
    sorts results (derivative is taken w.r.t. unsorted Z or R)
    Parameters
    ----------
    Z : 1 x n dimensional array
        contains nuclear charges
    R : 3 x n dimensional array
        contains nuclear positions
    N : float
        number of electrons in system
        here: meaningless, can remain empty
    dx_index : integer
        identifies by which variable to derive by
        (0 : Z, 1 : R, 2 : N)
    Return
    ------
    J : Jacobian of CM_ev
    '''

    print("calculating Jacobian of Coulomb Matrix eigenvalues")
    fM, order = jrep.CM_ev(Z, R, N) #get order of sorted representation
    dim = len(order)
    
    #direct derivative as jacobian
    dCM_ev = jacfwd(jrep.CM_ev, dx_index)
    ref_dCM_ev = dCM_ev(Z, R, N)[0]

    #ref_dCM_ev = dCM_ev(Z.astype(float), R.astype(float), float(N))[0]
    
    '''Not sure about reordering and values below. definitely need to recheck'''
    if(dx_index == 0):
        #assign derivative to correct field in sorted matrix by reordering derZ_ij to derZ_kl
        J_dZkl = jnp.asarray([[ref_dCM_ev[l][m] for l in range(dim)] for m in order])
        return(J_dZkl)
    elif(dx_index == 1):
        #unordered derivative taken from sorted matrix
        J_dRkl = jnp.asarray([[[ref_dCM_ev[l][m][x] for l in range(dim)] for x in range(3)] for m in order])
        return(J_dRkl)
    else:
        return(ref_dCM_ev)



def d_OM(Z, R, N, dx_index = 0):
    dim = jrep.OM_dimension(Z)
    Jraw = jacfwd(jrep.OM_full_sorted, dx_index)
    J = Jraw(Z, R, N)
    print('jraw', Jraw, 'J', J)
    return(J)

def dd_OM(Z, R, N, dx_index = 0, ddx_index = 0):
    dim = jrep.OM_dimension(Z)
    Hraw = hessian(jrep.OM_full_sorted, dx_index, ddx_index)(Z, R, N)[0]
    
    print('the matrix was not sorted according to derivatives')
    return(Hraw)

def d_OM_ev(Z, R, N, dx_index = 0):
    dim = jrep.OM_dimension(Z)


def dd_OM_ev(Z, R, N, dx_index = 0, ddx_index = 0):
    return()


def dd_CM(Z, R, N, dx_index = 0, ddx_index = 0, M = None, order = None, time_calculations = True):
    '''
    calculates and sorts second derivatives
    '''
    
    if M == None:
        fM_sorted, order = jrep.CM_full_sorted(Z, R, N)
    else:
        fM_sorted = M
    dim = len(order)
    
    #calculates dZdZ
    if (dx_index ==0):
        if (ddx_index == 0):
            HdZraw = hessian(jrep.CM_full_sorted, dx_index, ddx_index)(Z, R, N)[0]
            
            '''
            sorting function, performs the following rearrangement:
            [[[[HdZraw[k,m, n] for k in range(dim)] for m in order] for n in order])
            '''
                        
            dZdZ_ordered = np.transpose(HdZraw,(2, 0, 1))
            
            dZdZ_sorted = np.copy(dZdZ_ordered)
            for m in range(dZdZ_ordered.shape[0]):
                for n in range(dZdZ_ordered.shape[1]):
                    dZdZ_sorted[m, n] = dZdZ_ordered[order[m], order[n]]
            
            return(dZdZ_sorted)
    
    #calculates dRdR
    if (dx_index == 1):
        if (ddx_index == 1):
            
            HdRraw = hessian(jrep.CM_full_sorted, dx_index, ddx_index)(Z, R, N)[0]
            
            print("do dRdR sorting")
            
            '''
            sorting function, performs the following rearrangement:
            [[[[[[HdRraw[n, m, i, x, j, y] for n in range(dim)] for m in range(dim)] for y in range(3)] for j in order] for x in range(3)] for i in order])
            '''
            dRdR_ordered = np.transpose(HdRraw,(2, 3, 4, 5, 0, 1))
            
            dRdR_sorted = np.copy(dRdR_ordered)
            for i in range(dRdR_ordered.shape[0]):
                for x in range(3):
                    for j in range(dRdR_ordered.shape[2]):
                        for y in range(3):
                            dRdR_sorted[i, x, j, y] = dRdR_ordered[order[i], x, order[j], y]
            
            return(dRdR_sorted)

    if (dx_index == 0 and ddx_index == 1 ) or (dx_index == 1 and ddx_index == 0):
        print("you want to calculate dZdR or dRdZ, line 372")

        HdZdRraw = hessian(jrep.CM_full_sorted, 0, 1)(Z, R, N)[0]
        
        '''sorting function, performs the following reordering but in fast
        [[[[[HdZdRraw[n, m, i, j, x] for n in range(dim)] for m in range(dim)] for x in range(3)] for j in order] for i in order]
        '''
        #could be that I messed this up on 1.12.2020, check with push before
        dZdR_ordered = np.transpose(HdZdRraw,(2, 3, 4, 0, 1))
        dZdR_sorted = np.copy(dZdR_ordered)
 
        for i in range(dZdR_ordered.shape[0]):
            for j in range(dZdR_ordered.shape[1]):
                for x in range(3):
                    dZdR_sorted[i,j,x] = dZdR_ordered[order[i], order[j], x]
                    
        return(dZdR_sorted)

def dd_CM_ev(Z, R, N, dx_index = 0, ddx_index = 0):
   
    Z = Z.astype(float)
    R = R.astype(float)
    N = float(N)
    
    fM, order = jrep.CM_ev(Z, R, N)
    dim = len(Z)
    Hraw = hessian(jrep.CM_ev, dx_index, ddx_index)(Z, R, N)[0]

    '''
    calculates and sorts second derivatives
    '''
    fM_sorted, order = jrep.CM_full_sorted(Z, R,N)
    dim = len(order)

    #calculates dZdZ
    if (dx_index ==0):
        if (ddx_index == 0):
            '''
            sorting function, performs the following rearrangement:
            [[[[HdZraw[k, m, n] for k in range(dim)] for m in order] for n in order])
            '''
            dZdZ_ordered = np.transpose(Hraw,(2, 0, 1))

            dZdZ_sorted = np.copy(dZdZ_ordered)
            for i in range(dZdZ_ordered.shape[0]):
                for j in range(dZdZ_ordered.shape[1]):
                    dZdZ_sorted[i, j] = dZdZ_ordered[order[i], order[j]]

            return(dZdZ_sorted)
    
    #calculates dRdR
    if (dx_index == 1):
        if (ddx_index == 1):

            print("do dRdR sorting")
            '''
            sorting function, performs the following rearrangement:
            [[[[[[HdRraw[n, i, x, j, y] for n in range(dim)] for y in range(3)] for j in order] for x in range(3)] for i in order])
            '''
            dRdR_ordered = np.transpose(Hraw,(1, 2, 3, 4, 0))

            dRdR_sorted = np.copy(dRdR_ordered)
            for i in range(dRdR_ordered.shape[0]):
                for x in range(3):
                    for j in range(dRdR_ordered.shape[0]):#why does dim not work?? 
                        for y in range(3):
                            dRdR_sorted[i, x, j, y] = dRdR_ordered[order[i], x, order[j], y]

            return(dRdR_sorted)

    if (dx_index == 0 and ddx_index == 1 ) or (dx_index == 1 and ddx_index == 0):
        print("you want to calculate dZdR or dRdZ, line 443")

        '''sorting function, performs the following reordering but in fast
        [[[[[HdZdRraw[m, i, j, x] for m in range(dim)] for x in range(3)] for j in order] for i in order]
        '''
        dZdR_ordered = np.transpose(Hraw,(1, 3, 2, 0))
        dZdR_sorted = np.copy(dZdR_ordered)

        for i in range(dZdR_ordered.shape[0]):
            for j in range(dZdR_ordered.shape[1]):
                for x in range(3):
                    dZdR_sorted[i,j,x] = dZdR_ordered[order[i], order[j], x]

        return(dZdR_sorted)
    
    if (dx_index == 2 or ddx_index == 2):
        return(Hraw)




def hessian(f, dx, ddx):
    H = jacfwd(jacfwd(f, dx), ddx)
    return(H)

