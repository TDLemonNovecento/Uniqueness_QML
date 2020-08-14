import numpy as np
from scipy.io import loadmat


#global variables
mat_data = "qm7.mat"



#code starts below
#load matlab dataset into python dictionary
dataset = loadmat(mat_data)
print(dataset.keys())
