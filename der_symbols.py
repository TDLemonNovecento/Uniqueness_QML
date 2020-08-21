import sympy as sp

'''
This file contains symbols used for the derivative calculations
Sufficient for molecules containing up to 23 atoms
'''

N = sp.symbols('N')

Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8, Z9, Z10, Z11, Z12, Z13, Z14, Z15, Z16, Z17, Z18, Z19, Z20, Z21, Z22, Z23 = sp.symbols('Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Z13 Z14 Z15 Z16 Z17 Z18 Z19 Z20 Z21 Z22 Z23')

Z = sp.Matrix([[Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8, Z9, Z10, Z10, Z11, Z12, Z13, Z14, Z15, Z16, Z17, Z18, Z19, Z20, Z21, Z22, Z23]])

x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23 = sp.symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21 x22 x23')
y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23 = sp.symbols('y1 y2 y3 y4 y5 y6 y7 y8 y9 y10 y11 y12 y13 y14 y15 y16 y17 y18 y19 y20 y21 y22 y23')
z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12, z13, z14, z15, z16, z17, z18, z19, z20, z21, z22, z23 = sp.symbols('z1 z2 z3 z4 z5 z6 z7 z8 z9 z10 z11 z12 z13 z14 z15 z16 z17 z18 z19 z20 z21 z22 z23')

r1 = sp.Matrix([[x1, y1, z1]])
r2 = sp.Matrix([[x2, y2, z2]])
r3 = sp.Matrix([[x3, y3, z3]])
r4 = sp.Matrix([[x4, y4, z4]])
r5 = sp.Matrix([[x5, y5, z5]])
r6 = sp.Matrix([[x6, y6, z6]])
r7 = sp.Matrix([[x7, y7, z7]])
r8 = sp.Matrix([[x8, y8, z8]])
r9 = sp.Matrix([[x9, y9, z9]])
r10 = sp.Matrix([[x10, y10, z10]])
r11 = sp.Matrix([[x11, y11, z11]])
r12 = sp.Matrix([[x12, y12, z12]])
r13 = sp.Matrix([[x13, y13, z13]])
r14 = sp.Matrix([[x14, y14, z14]])
r15 = sp.Matrix([[x15, y15, z15]])
r16 = sp.Matrix([[x16, y16, z16]])
r17 = sp.Matrix([[x17, y17, z17]])
r18 = sp.Matrix([[x18, y18, z18]])
r19 = sp.Matrix([[x19, y19, z19]])
r20 = sp.Matrix([[x20, y20, z20]])
r21 = sp.Matrix([[x21, y21, z21]])
r22 = sp.Matrix([[x22, y22, z22]])
r23 = sp.Matrix([[x23, y23, z23]])


R =  sp.Matrix([r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23])
