#In this program routines and subroutines are called depending on what shoud be done. Unused parts should be hashed and parts that conflict with each other marked with exclamation marks and special characters respectively twice that character. Written by Miriam Stuke.
import os
import qml
import repro
import derivative
import sympy as sp
import numpy as np
import sys
#Global variables
x1, x2, x3, y1, y2, y3, z1, z2, z3, Z1, Z2, Z3, N = sp.symbols('x1 x2 x3 y1 y2 y3 z1 z2 z3 Z1 Z2 Z3 N')
r1 = sp.Matrix([[x1, y1, z1]])
r2 = sp.Matrix([[x2, y2, z2]])
r3 = sp.Matrix([[x3, y3, z3]])

Z = sp.Matrix([[Z1, Z2, Z3]])
R =  sp.Matrix([r1, r2, r3])


#define path to folder containing xyz files. All files are considered.
database = "/home/stuke/Databases/XYZ_random/"

print("calculate Coulomb Matrix Representations")
f = repro.Coulomb_Matrix_FM(R, Z, N, 2 , 1, [0, 0, 0], [0, 0, 0])
print("Representation we're currently working with:")
print(f)

print("iterate over all molecules")
for xyzfile in os.listdir(database):
    xyz_fullpath = database + xyzfile #probably path can be gotten more directly
    compound = qml.Compound(xyz_fullpath)

    c_Z = compound.nuclear_charges
    cR = compound.coordinates
    cN = len(c_Z)
    

    zeros23 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    n = f.shape[0]
    
    cZ = np.append(c_Z, zeros23)

    def subs_CM(f, cZ, cR, cN):
        try:
            for i in range(n):
                for j in range(n):
                    f[i,j] = f[i,j].subs({sp.Symbol('x1'):cR[0, 0], sp.Symbol('y1'):cR[0,1], sp.Symbol('z1'):cR[0,2], sp.Symbol('x2'):cR[1,0], sp.Symbol('y2'): cR[1,1], sp.Symbol('z2'):cR[1,2], sp.Symbol('Z1'):cZ[0], sp.Symbol('Z2') : cZ[1], sp.Symbol('Z3'):cZ[2], sp.Symbol('N') : cN})
        except IndexError:
            pass
        return(f)
    eval_f = subs_CM(f, cZ, cR, cN)    
    print(eval_f)
        


    #eval_fM = f(c_R, c_Z, c_N)
    
             
