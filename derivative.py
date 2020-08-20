#here we calculate the first and second derivatives to the basis of chemical space depending on the chosen representation
import numpy as np
import sympy as sp

#variables by which to derive:

dx = sp.symbols('dx')


def firstd(f, dx):
    '''Compute the first derivative of a mapping function fM.
    Tries analytical derivation first.
    If it fails, numerical derivation is performed.

    Parameters
    ----------
    fM : function
        maps from xyz to space of representation M
    n : number of atoms

    Internal Variables
    ------------------
    r1,r2,r3 : variable
        Armstroeng, coordinates in xyz space
    Z : vector
        variable of Nuclear charge
    N : variable
        number of electrons

    Returns
    -------
    vector of 5 derivatives
    matching format to mapping function fM
    
    '''
    try:
        #print("analytical evaluation started")
        dev = sp.diff(f, dx)    
        
    except:
        #print("numerical evaluation had to be run")
        dev = numerical_dev(f, dx)
    else:
        pass
        #print("analytical evaluation excited successfully")
    finally:
        return(dev)
    
def secondd(rep):
    return("some hessian matrix is returned here")


def numerical_dev(fM, var, method = 'central'):
    '''Compute the difference formula for f'(a) with step size h.

    Parameters
    ----------
    f : function
        Vectorized function of one variable
    a : number
        Compute derivative at x = a
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
    
    h = 0.1 * ones(5, 1)

    if method == 'central':
        return (fM.sp.subs(var, var + h) - fM.sp.subs(var, var - h))/(2*h)
    elif method == 'forward':
        return (fM.sp.subs(var, var + h) - fM.sp.subs(var, var))/h
    elif method == 'backward':
        return (fM.sp.subs(var, var) - fM.sp.subs(var, var - h))/h
    else:
        raise ValueError("Something went wrong in numerical diff.")
