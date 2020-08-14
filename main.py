#In this program routines and subroutines are called depending on what shoud be done. Unused parts should be hashed and parts that conflict with each other marked with exclamation marks and special characters respectively twice that character. Written by Miriam Stuke.
import os
import qml
import repro
import derivative

#Global variables

#define path to folder containing xyz files. All files are considered.
database = "/home/stuke/Databases/XYZ_random/"

#mapping function for respective representation from xyz to representation space
representation = repro.cm
#"cm_ev"


print("iterate over all molecules")
for xyzfile in os.listdir(database):
    xyz_fullpath = database + xyzfile #probably path can be gotten more directly
    rep = representation(xyz_fullpath) #create qml element
    first_der = derivative.firstd(rep) #calculate first derivative of this repo
    second_der = derivative.secondd(rep) #calculate second derivative of this repo


    print(rep)
    print(first_der)
    print(second_der)

