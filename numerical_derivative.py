'''
@author: miriam
initiated: 18.12.2020

This file contains a numerical derivative function and a short workflow for getting derivatives
w.r.t.xi,yi,zi or Zi
'''
#from jax.config import config
#config.update("jax_enable_x64", True) #increase precision from float32 to float64
import time
import numpy as np
import database_preparation as datprep


dim = 23

def derivative(representation, ZRN, kind = 'numerical', order = 1, d1 = [0,0,0], d2 = [0,0,0], h = 0.00001):
    '''the numerical derivative can only deal with unsorted representations,
    while the analytical derivative can use both sorted and unsorted.
    
    Variables:
    ----------
    representation: some mapping function that takes as first three arguments Z, R and N
                    Examples are most of the jax.representation functions.
                    Derivatives d1=[0,...], d1 = [1, ...] and d2 = [0,...], d2 = [1,...]
                    can be taken of functions that have Z, R as first two arguments and
                    don't take any N value
    ZRN = [Z, R, N] : List of arrays, namely Z = nuclear charges, R = nuclear positions in x,y,z
                and N = total number of electrons of system (unloaded molecule: sum(Z)).
                Defines where we want to differentiate.

    kind : 'numerical', or 'analytical'
    order: 0 = no derivative, just repro, 1 = 1rst derivative, 2 = 2nd derivative
    d1, d2 : list of int, first and second derivative
            defines coordinate w.r.t. which function is derived
            [0,...] : dZ
            [1,...] : dR
            [2,...] : dN
            [j,i,...]: dji, e.g. [0,3,...] = dZ3
            [1,i,n] : which dx,dy,dz when deriving by dR
                        n = 0 : dxi
                        n = 1 : dyi
                        n = 2 : dzi

    h : float
        for numerical derivative, little change induced to variables
    '''

    if(kind == 'numerical'):
        #only works with unsorted representations
        #print("your representation is %s. If this is a sorted representation, change or do presorting\n \
        #        otherwise results may be false due to numerical differentiation" % (str(representation)))
        
        #return representation
        if order == 0:
            return( representation(ZRN[0], ZRN[1],ZRN[2]))
        
        #return 1st order numerical derivative
        elif order == 1:
            return( num_der1(representation, ZRN, d1, h))
        
        #return second order numerical derivative
        elif order == 2:
            return( num_der2(representation, ZRN, d1, d2, h))

        else: 
            print("you're order is out of bounds")



def num_der1(representation, ZRN, d1, h = 0.1, dimension = dim):
    '''computes central difference derivative
    of representatin w.r.t. d1
    some elements taken from @author nick
    Variables:
    ----------
    same definition as in 'derivative' function

    Returns:
    ---------
    difference : central difference derivative
    '''
    
    #print("checkpoint 3: numerical differentiation 1st grade started, numerical_differentiation line 84")
    #change variable by which to derive (d1) slightly by h
    #print("ZRN: ", ZRN)

    plus_ZRN = datprep.alter_coordinates(ZRN, d1, h)
    minus_ZRN = datprep.alter_coordinates(ZRN, d1, -h)
    
    #get representation with slightly changed input
    tic = time.perf_counter()
    repro_plus = representation(plus_ZRN[0], plus_ZRN[1], plus_ZRN[2])
    repro_minus = representation(minus_ZRN[0], minus_ZRN[1], minus_ZRN[2])
    toc = time.perf_counter()
    #print(f"2Repros in {toc - tic:0.4f} seconds")

    #print(repro_plus)
    #print("repro minus: \n", repro_minus)

    repro_pls = repro_plus.flatten()
    repro_mns = repro_minus.flatten()
    
    difference = (repro_pls - repro_mns)/ (2*h) #brackets around h are vital!

    return(np.asarray(difference))


def num_der2(representation, ZRN, d1, d2, h = 0.1, dim = 3):
    '''computes central difference derivative
    of representatin w.r.t. d1 and d2 (2nd order)
    Formula for 2nd order derivative:
        derivative = f(a+h_1, b+h_2) - f(a+h_1, b-h_2) - f(a-h_1, b+h_2) + f(a-h_1, b-h_2))/(4h_1*h_2)
    for d1 = d2:
        derivative = (f(a+h) - 2f(a) + f(a-h))/h²


    Variables:
    ----------
    same definition as in 'derivative' function

    Returns:
    ---------

    '''
    
    #change variable by which to derive (d1) slightly by h
    plus_ZRN = datprep.alter_coordinates(ZRN, d1, h)
    minus_ZRN = datprep.alter_coordinates(ZRN, d1, -h)

    if (d1 == d2):
        #calculate representation with slight changes
        repro_plus = representation(plus_ZRN[0], plus_ZRN[1], plus_ZRN[2])
        repro_minus = representation(minus_ZRN[0], minus_ZRN[1], minus_ZRN[2])
        repro_normal = representation(ZRN[0], ZRN[1], ZRN[2])

        #flatten representations
        repro_pls = repro_plus.flatten()
        repro_mns = repro_minus.flatten()
        repro_nml = repro_normal.flatten()

        difference = (repro_pls + repro_mns - 2*repro_nml) / (h**2)

    else:
        #change initial variable (d2) slightly by h
        plusplus_ZRN = datprep.alter_coordinates(plus_ZRN, d2, h)
        plusminus_ZRN = datprep.alter_coordinates(plus_ZRN, d2, -h)
        minusplus_ZRN = datprep.alter_coordinates(minus_ZRN, d2, h)
        minusminus_ZRN = datprep.alter_coordinates(minus_ZRN, d2, -h)

        #calculate representation
        repro_plusplus = representation(plusplus_ZRN[0], plusplus_ZRN[1], plusplus_ZRN[2])
        repro_plusminus = representation(plusminus_ZRN[0], plusminus_ZRN[1], plusminus_ZRN[2])
        repro_minusplus = representation(minusplus_ZRN[0], minusplus_ZRN[1], minusplus_ZRN[2])
        repro_minusminus = representation(minusminus_ZRN[0], minusminus_ZRN[1], minusminus_ZRN[2])
        
        #flatten results
        repro_pp = repro_plusplus.flatten()
        repro_pm = repro_plusminus.flatten()
        repro_mp = repro_minusplus.flatten()
        repro_mm = repro_minusminus.flatten()
        
        difference = (repro_pp + repro_mm - repro_mp - repro_pm) / (4*h**2)
    
    return(np.asarray(difference))
