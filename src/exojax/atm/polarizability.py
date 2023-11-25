""" Gas Polarizability 

    Notes:
        Originally taken from PICASO/GPLv3 picaso/rayleigh.py
        polarizabilities are mainly taken from
        CRC handbook of chemistry and physics vol. 95 unit=cm3
        H3+ taken from Kawaoka & Borkman, 1971   
        Number density at reference conditions of refractive index measurements                                         
        i.e. number density of the ideal gas at T=273.15K (=0 C) and P=1atm [cm-2], as Patm*bar_cgs / (kB * 273.15) 
        http://refractiveindex.info   
        n_ref_refractive = 2.6867810458916872e+19
"""


polarizability = {
    'H2': 0.804e-24,
    'He': 0.21e-24,
    'N2': 1.74e-24,
    'O2': 1.57e-24,
    'O3': 3.21e-24,
    'H2O': 1.45e-24,
    'CH4': 2.593e-24,
    'C2H2': 3.33e-24,
    'CO': 1.95e-24,
    'CO2': 2.911e-24,
    'NH3': 2.26e-24,
    'HCN': 2.59e-24,
    'PH3': 4.84e-24,
    'SO2': 3.72e-24,
    'SO3': 4.84e-24,
    'C2H2': 3.33e-24,
    'H2S': 3.78e-24,
    'NO': 1.70e-24,
    'NO2': 3.02e-24,
    'H3+': 0.385e-24,
    'OH': 6.965e-24,
    'Na': 24.11e-24,
    'K': 42.9e-24,
    'Li': 24.33e-24,
    'Rb': 47.39e-24,
    'Cs': 59.42e-24,
    'TiO': 16.9e-24,
    'VO': 14.4e-24,
    'AlO': 8.22e-24,
    'SiO': 5.53e-24,
    'CaO': 23.8e-24,
    'TiH': 16.9e-24,
    'MgH': 10.5e-24,
    'NaH': 24.11e-24,
    'AlH': 8.22e-24,
    'CrH': 11.6e-24,
    'FeH': 9.47e-24,
    'CaH': 23.8e-24,
    'BeH': 5.60e-24,
    'ScH': 21.2e-24
}

king_correction_factor = {
    "O3": 1.060000,
    "CO": 1.016995,
    "C2H2": 1.064385,
    "C2H6": 1.006063,
    "OCS": 1.138786,
    "CH3Cl": 1.026042,
    "H2S": 1.001880,
    "SO2": 1.062638
}

