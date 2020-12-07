'''this document contains all Coulomb matrix eigenvalue elements and first and second
derivatives for the H2O molecule described by

r1 = [-02, 1.353, -0.908]
r2 = [0.2, -1.353, -0.908]
r3 = [0, 0, 0.227]

the sorting of the matrix leads to new ordering from 1>3, 2>2 3>1, which means that dZ1 == derivative by Z=8
'''
M = [74.0702, 0.312046, 0.13442]


'''
From here the first derivatives by R follow
'''
dx1 = [[0, -0.28499, 0.28499],
        [-0.28499, 0, 0],
        [0.28499, 0, 0]]

dx2 =[[0, 0.28499, 0],
        [0.28499, 0, 0.0195432],
        [0, 0.0195432, 0]]

dx3 = [[0, 0, -0.28499],
        [0, 0, -0.0195432],
        [-0.28499, -0.0195432, 0]]


dy1 = [[0, 1.92796, -1.92796],
    [1.92796, 0, 0],
    [-1.92796, 0, 0]]

dy2 = [[0, -1.92796, 0],
    [-1.92796, 0, -0.13221],
    [0, -0.13221, 0]]

dy3 = [[0, 0, 1.92796],
    [0, 0, 0.13221],
    [1.92796, 0.13221, 0]]


dz1 = [[0, -1.61732, -1.61732],
        [-1.61732, 0, 0],
        [-1.61732, 0, 0]]

dz2 = [[0, 1.61732, 0],
    [1.61732, 0, 0],
    [0, 0, 0]]

dz3 =[[0, 0, 1.61732],
    [0, 0, 0],
    [1.61732, 0, 0]]


'''below are the three dZ derivatives'''
dZ1 = [[22.055, 0.562648, 0.562648],
        [0.562648, 0, 0],
        [0.562648, 0, 0]]

dZ2 = [[0, 4.50118, 0],
        [4.50118, 1.2, 0.365577],
        [0, 0.365577, 0]]

dZ3 = [[0, 0, 4.50118],
        [0, 0, 0.365577],
        [4.50118, 0.365577, 1.2]]

'''
From here on all possible combinations of two derivatives (Hessian) for the 
Molecule follow. Since they can be exchanged without effect (dxdy = dydx)
only one such combination is given.
'''

'''
Below are all dZdZ derivatives
'''
dZ1dZ1 = [[3.85963, 0, 0],
        [0, 0, 0],
        [0, 0, 0]]

dZ1dZ2 = [[0, 0.562648, 0],
        [0.562648, 0, 0],
        [0, 0, 0]]

dZ1dZ3 = [[0, 0, 0.562648],
        [0, 0, 0],
        [0.562648, 0, 0]]

dZ2dZ2 = [[0, 0, 0],
        [0, 1.68, 0],
        [0, 0, 0]]

dZ2dZ3 = [[0, 0, 0],
        [0, 0, 0.365577],
        [0, 0.365577, 0]]

dZ3dZ3 = [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 1.68]]

'''
Below are all dRdR derivatives
'''

dx1dx1 = [[0, -1.37082, -1.37082],
        [-1.37082, 0, 0],
        [-1.37082, 0, 0]]

dx1dx2 = [[0, 1.37082, 0],
        [1.37082, 0, 0],
        [0, 0, 0]]

dx1dx2 = [[0, 0, 1.37082],
        [0, 0, 0],
        [1.37082, 0, 0]]

dx1dy1 = [[0, -0.366203, -0.366203],
        [-0.366203, 0, 0],
        [-0.366203, 0, 0]]

dx1dy2 = [[0, 0.366203, 0],
        [0.366203, 0, 0],
        [0, 0, 0]]

dx1dy3 = [[0, 0, 0.366203],
        [0, 0, 0],
        [0.366203, 0, 0]]

dx1dz1 = [[0, 0.307199, -0.307199],
        [0.307199, 0, 0],
        [-0.207199, 0, 0]]

dx1dz2 = [[0, -0.307199, 0],
        [-0.307199, 0, 0],
        [0, 0, 0]]

dx1dz3 = [[0, 0, 0.307199],
        [0, 0, 0],
        [0.307199, 0, 0]]

'''now do all dx2 first and everything else second except for dx1'''

dx2dx2 = [[0, -1.37082, 0],
        [-1.37082, 0, -0.0457237],
        0, -0.0457237, 0]]

dx2dx3 = [[0, 0, 0],
        [0, 0, 0.0457237],
        [0, 0.0457237, 0]]

dx2dy1 = [[0, 0.366203, 0],
        [0.366203, 0, 0],
        [0, 0, 0]]

dx2dy2 = [[0, -0.366203, 0],
        [-0.366203, 0, -0.0212032],
        [0, -0.0212032,  0]]

dx2dy3 = [[0, 0, 0],
        [0, 0, 0.0212032],
        [0, 0.0212032, 0]]

dx2dz1 = [[0, -0.307199, 0],
        [-0.307199, 0, 0],
        [0, 0, 0]]

dx2dz2 = [[0, 0.307199, 0],
        [0.307199, 0, 0],
        [0, 0, 0]]

dx2dz3 = [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]]

'''it follows all dx3'''

dx3dx3 = [[0, 0, -1.37082],
        [0, 0, -0.0457237],
        [-1.37082, -0.0457237, 0]]

dx3dy1 = [[0, 0, 0.366203],
        [0, 0, 0],
        [0.366203, 0, 0]]

dx3dy2 = [[0, 0, 0],
        [0, 0, 0.0212032],
        [0, 0.0212032, 0]]

dx3dy3 = [[0, 0, -0.366203],
        [0, 0, -0.0212032],
        [-0.366203, -0.0212032, 0]]

dx3dz1 = [[0, 0, 0.307199],
        [0, 0, 0],
        [0.307199, 0, 0]]

dx3dz2 = [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]]

dx3dz3 = [[0, 0, -0.307199],
        [0, 0, 0],
        [-0.307199, 0, 0]]

'''onwards with y1'''

dy1dy1 = [[0, 1.05241, 1.05241],
        [1.05241, 0, 0],
        [1.05241, 0, 0]]

dy1dy2 = [[0, -1.05241, 0],
        [-1.05241, 0, 0],
        [0, 0, 0]]

dy1dy3 = [[0, 0, -1.05241],
        [0, 0, 0],
        [-1.05241, 0, 0]]

dy1dz1 = [[0, -2.0782, 0],
        [-2.0782, 0, 0],
        [0, 0, 0]]

dy1dz2 = [[0, 2.0782, 0],
        [2.0782, 0, 0],
        [0, 0, 0]]

dy1dz3 = [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]]

'''now y2'''

dy2dy2 = [[0, 1.05241, 0],
        [1.05241, 0, 0.0945814],
        [0, 0.0945814, 0]]

dy2dy3 = [[0, 0, 0],
        [0, 0, -0.0945814],
        [0, -0.0945814, 0]]

dy2dz1 = [[0, 2.0782, 0],
        [2.0782, 0, 0],
        [0, 0, 0]]

dy2dz2 = [[0, -2.0782, 0],
        [-2.0782, 0, 0],
        [0, 0, 0]]

dy2dz3 = [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]]

'''the last why: y3'''

dy3dy3 = [[0, 0, 1.05241],
        [0, 0, 0.0945817],
        [1.05241, 0.0945817, 0]]

dy3dz1 = [[0, 0, 2.0782],
        [0, 0, 0],
        [2.0782, 0, 0]]

dy3dz2 = [[0, 0, 0],
        [0, 0, 0]
        [0, 0, 0]]

dy3dz3 = [[0, 0, -2.0782],
        [0, 0, 0],
        [-2.0782, 0, 0]]

'''the first z! z1:'''

dz1dz1 = [[0, 0.318405, 0.318405],
        [0.318405, 0, 0],
        [0.318405, 0, 0]]

dz1dz2 = [[0, -0.318405, 0],
        [-0.318405, 0, 0],
        [0, 0, 0]]

dz1dz3 = [[0, 0, -0.318405],
        [0, 0, 0],
        [-0.318405, 0, 0]]

'''second last z2'''

dz2dz2 = [[0, 0.318405,0],
        [0.318405, 0, -0.048858],
        [0, -0.048858, 0]]

dz2dz3 = [[0, 0, 0],
        [0, 0, 0.048858],
        [0, 0.048858, 0]]

'''last one! z3!'''

dz3dz3 = [[0, 0, 0.318405],
        [0, 0, -0.048858],
        [0.318405, -0.048858, 0]]






'''
Below are all dZ1dR combinations
'''

dZ1dx1 = [[0, -0.0356238, 0.0356238],
        [-0.0356238, 0, 0],
        [0.0356238, 0, 0]]

dZ1dx2 = [[0, 0.0356238, 0],
        [0.0356238, 0, 0],
        [0, 0, 0]]

dZ1dx3 = [[0, 0, -0.0356238],
        [0, 0, 0],
        [-0.0356238, 0, 0]]

dZ1dy1 = [[0, 0.240995, -0.240995],
        [0.240995, 0, 0],
        [-0.240995, 0, 0]]

dZ1dy2 = [[0, -0.240995, 0],
        [-0.240995, 0, 0],
        [0, 0, 0]]

dZ1dy3 = [[0, 0, 0.240995],
        [0, 0, 0],
        [0.240995, 0, 0]]

dZ1dz1 = [[0, -0.202165, -0.202165],
        [-0.202165, 0, 0],
        [-0.202165, 0, 0]]

dZ1dz2 = [[0, 0.202165, 0],
        [0,202165, 0, 0],
        [0, 0, 0]]

dZ1dz3 = [[0, 0, 0.202165],
        [0, 0, 0],
        [0.202165, 0 ,0]]

'''Now all the dZ2dR derivatives'''

dZ2dx1 =[[0, -0.28499, 0],
        [-0.28499, 0, 0],
        [0, 0, 0]]

dZ2dx2 = [[0, 0.28499, 0],
        [0.28499, 0, 0.0195432],
        [0, 0.0195432, 0]]

dZ2dx3 = [[0, 0, 0],
        [0, 0, -0.0195432],
        [0, -0.0195432, 0]]

dZ2dy1 = [[0, 1.92796, 0],
        [1.92796, 0, 0],
        [0, 0, 0]]

dZ2dy2 = [[0, -1.92796, 0],
        [-1.92796, 0, -0.13221],
        [0, -0.13221, 0]]

dZ2dy3 = [[0, 0, 0],
        [0, 0, 0.13221],
        [0, 0.13221, 0]]

dZ2dz1 = [[0, -1.61732, 0],
    [-1.61732, 0, 0],
    [0, 0, 0]]

dZ2dz2 = [[0, 1.61732, 0],
    [1.61732, 0, 0],
    [0, 0, 0]]

dZ2dz3= [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]]


'''now all the dZ3dR derivatives'''

dZ3dx1 = [[0, 0, 0.28499],
        [0, 0, 0],
        [0.28499, 0, 0]]

dZ3dx2 = [[0, 0, 0],
        [0, 0, 0.0195432],
        [0, 0.0195432, 0]]

dZ3dx3 = [[0, 0, -0.28499],
        [0, 0, -0.0195432],
        [-0.28499, -0.0195432, 0]]


dZ3dy1 = [[0, 0, -1.92796],
        [0, 0, 0],
        [-1.92796, 0, 0]]

dZ3dy2 = [[0, 0, 0],
        [0, 0, -0.13221],
        [0, -013221, 0]]

dZ3dy3 = [[0, 0, 1.92796],
        [0, 0, 0.13221],
        [1.92796, 0.13221, 0]]

dZ3dz1 = [[0, 0, -1.61732],
        [0, 0, 0],
        [-1.61732, 0, 0]]

dZ3dz2 = [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]]

dZ3dz3 = [[0, 0, 1.61732],
        [0, 0, 0],
        [1.61732, 0, 0]]

