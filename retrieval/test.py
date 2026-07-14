"""Quick script to ingest a custom CH4 line list and compute an HT-profile cross-section.

This file is intended as an exploratory helper script, not a production module.
"""

# Import dependencies.
import numpy as np
import pandas as pd
from hapi2 import hapi
import matplotlib.pyplot as plt

# Step 1: Read the custom parameter file.
file_path = '~/Downloads/Merlin_Line List.par'


# Column order expected in the source .par file.
column_names = [
    'Mol', 'w0', 'S_296K', 'Elow', 'Gamma0_air', 'Gamma0_self', 'n_gamma0_air',
    'delta0_air', 'delta0_self', 'n_delta0_air', 'sigma0', 'n_sigma0',
    'gamma2', 'n_gamma2', 'd2', 'eta'
]

# Parse the fixed-format text file into a DataFrame.
df = pd.read_csv(
    file_path,
    sep=r'\s+',
    engine='python',
    comment='#',
    names=column_names,
    skiprows=24
)

# Step 2: Map source columns to HAPI2 field names.
hapi_df = pd.DataFrame({
    'molec_id': df['Mol'],
    'local_iso_id': 1,  # Adjust isotopologue number if necessary
    'nu_lines': df['w0'],
    'sw_lines': df['S_296K'],
    'gamma_air': df['Gamma0_air'],
    'gamma_self': df['Gamma0_self'],
    'elower': df['Elow'],
    'n_air': df['n_gamma0_air'],
    'delta_air': df['delta0_air'],
    'n_delta_air': df['n_delta0_air'],
    'sigma': df['sigma0'],
    'n_sigma': df['n_sigma0'],
    'gamma2': df['gamma2'],
    'n_gamma2': df['n_gamma2'],
    'eta': df['eta'],
})

hapi_df = hapi_df.fillna(0.0)
hapi_line_data = hapi_df.to_records(index=False)

# Step 3: Initialize HAPI2 database storage.
hapi.db_begin('data')
table_name = 'CH4_custom'

#hapi.save_to_file(
#    table_name,
#    hapi_line_data,
#    parameters=list(hapi_df.columns)
#)

# Step 4: Compute absorption cross-section using the Hartmann-Tran profile.
nu_start = 6076.0
nu_end = 6078.0
nu_step = 0.0001
nu_grid = np.arange(nu_start, nu_end, nu_step)

pressure = 1.0       # atm
temperature = 296.0  # K
mole_fraction = 1.0  # Adjust as needed

environment = {
    'p': pressure,
    'T': temperature,
    'MoleFractions': {'default': mole_fraction}
}

# Evaluate the HT profile on the configured grid.
cross_section = hapi.absorptionCoefficient_HT(
    nu_grid,
    table_name,
    environment,
    GammaL='gamma_air',
    VoigtType='HT',
    HITRAN_units=False
)
# Step 5: Plot the resulting cross-section.
plt.figure(figsize=(10, 6))
plt.plot(nu_grid, cross_section, label='Absorption Cross-Section')
plt.xlabel('Wavenumber (cm$^{-1}$)', fontsize=14)
plt.ylabel('Absorption Cross-Section (cm$^{-1}$)', fontsize=14)
plt.title('CH$_4$ Absorption Cross-Section using Hartmann-Tran Profile', fontsize=16)
plt.legend()
plt.grid(True)
plt.show()
