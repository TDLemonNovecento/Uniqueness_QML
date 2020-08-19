import sympy as sp

x,y, z = sp.symbols('x y z')

f = (x**2+x*z +y)

k = sp.diff(f, x)
print(k)
