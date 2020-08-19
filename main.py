#In this program routines and subroutines are called depending on what shoud be done. Unused parts should be hashed and parts that conflict with each other marked with exclamation marks and special characters respectively twice that character. Written by Miriam Stuke.
import os
import qml
import repro
import derivative
import sympy as sp

#Global variables
r1, r2, r3, Z1, Z2, Z3, N = sp.symbols('r1 r2 r3 Z1 Z2 Z3 N')
Z = sp.Matrix([[Z1, Z2, Z3]])
R =  sp.Matrix([[r1, r2, r3]])


#define path to folder containing xyz files. All files are considered.
database = "/home/stuke/Databases/XYZ_random/"


'''
print("iterate over all molecules")
for xyzfile in os.listdir(database):
    xyz_fullpath = database + xyzfile #probably path can be gotten more directly
    compound = qml.Compound(xyz_fullpath)

    print(compound)

'''

f = repro.Coulomb_Matrix_FVec(R, Z, N, 3 , 1)
print(f)
