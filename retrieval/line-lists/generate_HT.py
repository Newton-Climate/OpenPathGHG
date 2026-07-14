"""Generate a pressure-temperature lookup tensor for CH4 HT cross-sections.

Outputs a NumPy array with axes (wavenumber, pressure, temperature).
"""

import numpy as np
from hapi import *
# storage2cache('CH4_S')

# Pressure (atm), temperature (K), and spectral grid (cm^-1).
p_grid = np.arange(0.1, 1100, 10)
T_grid = np.arange(150, 330, 10)
d_nu = 0.001
nu = np.arange(6076, 6078, d_nu)
# fetch('CH4_S', 6,1,nu[0], nu[-1])

p_len, T_len = len(p_grid), len(T_grid)
nu_len = len(nu)

# Store results as sigma[nu, p, T].
sigma = np.zeros((nu_len, p_len, T_len))


for i in range(p_len):
    for j in range(T_len):
    # Evaluate HT absorption for each (p, T) pair.
        grid, sigma[:,i,j] = absorptionCoefficient_HT( SourceTables='merlin_12CH4', WavenumberRange=[nu[0],nu[-1]] ,WavenumberStep = d_nu ,Environment={'p':p_grid[i] ,'T':T_grid[j]},IntensityThreshold=1e-30)

# Persist lookup tensor for later interpolation/model building.
np.save('HT_12CH4', sigma)
