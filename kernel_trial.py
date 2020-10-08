from xtb.interface import Calculator
from xtb.utils import get_method
import numpy as np

Z = np.array([8, 1, 1])
R = np.array([[ 0.00000000000000, 0.00000000000000,-0.73578586109551],[ 1.44183152868459, 0.00000000000000, 0.36789293054775],[-1.44183152868459, 0.00000000000000, 0.36789293054775]])

calc = Calculator(get_method("GFN2-xTB"), Z, R)

res = calc.singlepoint()
energy = res.get_energy()



