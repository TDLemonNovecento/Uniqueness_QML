'''
@author: miriam
initiated: 18.12.2020

This file contains a numerical derivative function and a short workflow for getting derivatives
w.r.t. xi,yi,zi or Zi
'''
import numpy as np
import database_preparation as datprep

def derivative(representation, ZRN, kind = 'numerical', order = 1, d1 = [0,0,0], d2 = [0,0,0], dim = 3, h = 0.001):
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

    dim : int
            maximal dimenstion of representation
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



def num_der1(representation, ZRN, d1, h = 0.1, dim = 3):
    '''computes central difference derivative
    of representatin w.r.t. d1
    some elements taken from @author nick
    Variables:
    ----------
    same definition as in 'derivative' function

    Returns:
    ---------

    '''
    
    #change variable by which to derive (d1) slightly by h
    plus_ZRN = datprep.alter_coordinates(ZRN, d1, h)
    minus_ZRN = datprep.alter_coordinates(ZRN, d1, -h)
    
    #get representation with slightly changed input
    repro_plus = representation(plus_ZRN[0], plus_ZRN[1], plus_ZRN[2], dim)
    repro_minus = representation(minus_ZRN[0], minus_ZRN[1], minus_ZRN[2], dim)
    
    repro_pls = repro_plus.flatten()
    repro_mns = repro_minus.flatten()
    
    difference = (repro_pls - repro_mns)/ (2*h) #brackets around h are vital!

    return(difference)

