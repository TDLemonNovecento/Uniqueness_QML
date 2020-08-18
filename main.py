#In this program routines and subroutines are called depending on what shoud be done. Unused parts should be hashed and parts that conflict with each other marked with exclamation marks and special characters respectively twice that character. Written by Miriam Stuke.
import os
import qml
import repro
import derivative
import sympy as sp

#Global variables
r1, r2, r3, Z, N = sp.symbols('r1 r2 r3 Z N')


#define path to folder containing xyz files. All files are considered.
database = "/home/stuke/Databases/XYZ_random/"

#mapping function for respective representation from xyz to representation space
map_function = repro.Coulomb_Matrix_FVec(r1, r2, r3, Z, N, 2)
#"cm_ev"


print("iterate over all molecules")
for xyzfile in os.listdir(database):
    xyz_fullpath = database + xyzfile #probably path can be gotten more directly
    compound = qml.Compound(xyz_fullpath)

    print(compound)


    first_der = derivative.firstd(map_function, 2) #calculate first derivative of this repo
    #final_der = qml.representations.vector_to_matrix(first_der)
    print(first_der)
    second_der = derivative.secondd(first_der) #calculate second derivative of this repo


    print(map_function)
    print(first_der)
    print(second_der)

