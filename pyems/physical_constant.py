import numpy as np


C0 = 299792458
MUE0 = 4e-7 * np.pi
EPS0 = 1 / (MUE0 * (C0 ** 2))
# free space wave impedance
Z0 = np.sqrt(MUE0 / EPS0)  # Ohm
