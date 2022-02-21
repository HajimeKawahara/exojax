"""Several functions to simplify the integration of FastChem2 to Exojax.

* To use this module, you need to install FastChem with Python extension, https://github.com/exoclime/FastChem

Note:
    To install pyfastchem, clone FastChem from https://github.com/exoclime/FastChem. Then::

        mkdir FastChem/build
        cd FastChem/build
        cmake -DUSE_PYTHON=ON ..
        make

    then set PYTHONPATH to FastChem/python.
"""
import pyfastchem
import numpy as np
import jax.numpy as jnp
from exojax.atm.idealgas import number_density
from jax.lax import scan


def set_C_to_O(fastchem, c_to_o):
    """Setting up C/O ratio by scaling the O abundance as a function of the C/O
    ratio (Molliére et al. 2015)

    Args:
        fastchem: fastchem object created via pyfastchem.FastChem( str(dir_fastchem)+"input/element_abundances_solar_ext.dat",str(dir_fastchem)+"input/logK_ext.dat", 1)
        c_to_o: C/O ratio

    Returns:
        setting the element abundances in the FastChem based on the inputted C/O ratio
    """
    # extracting the indices for O and C from FastChem
    index_C = fastchem.getSpeciesIndex('C')
    index_O = fastchem.getSpeciesIndex('O')

    # making a copy of the current abundances
    current_abundances = np.array(fastchem.getElementAbundances())
    element_abundances = np.copy(current_abundances)

    # setting the O abundance as a function of the C/O ratio (Molliére et al. 2015)
    element_abundances[index_O] = element_abundances[index_C]/c_to_o

    # setting the FastChem with the new element abundances
    fastchem.setElementAbundances(element_abundances)
    print('C/O is set to '+str(c_to_o))


def set_M_to_H(fastchem, M_to_H):
    """Setting up [M/H] by scaling all of the metallic abundances but H and He
    by 10**M_to_H.

    Args:
        fastchem: fastchem object created via pyfastchem.FastChem( str(dir_fastchem)+"input/element_abundances_solar_ext.dat",str(dir_fastchem)+"input/logK_ext.dat", 1)
        M_to_H: metallicity

    Returns:
        setting the element abundances in the FastChem based on the inputted [M/H]
    """

    metallicity = 10**M_to_H
    # making a copy of the current abundances
    current_abundances = np.array(fastchem.getElementAbundances())
    element_abundances = np.copy(current_abundances)

    # scaling the element abundances, except those of H and He
    for j in range(0, fastchem.getElementNumber()):
        if fastchem.getSpeciesSymbol(j) != 'H' and fastchem.getSpeciesSymbol(j) != 'He':
            element_abundances[j] *= metallicity

    # setting the FastChem with the new element abundances
    fastchem.setElementAbundances(element_abundances)
    print('[M/H] is set to '+str(M_to_H))


def set_Fe_to_H(fastchem, Fe_to_H):
    """Setting up [Fe/H] by scaling the Fe abundances by 10**Fe_to_H.

    Args:
        fastchem: fastchem object created via pyfastchem.FastChem( str(dir_fastchem)+"input/element_abundances_solar_ext.dat",str(dir_fastchem)+"input/logK_ext.dat", 1)
        Fe_to_H: metallicity

    Returns:
        setting the element abundances in the FastChem based on the inputted [Fe/H]
    """

    metallicity = 10**Fe_to_H
    # making a copy of the current abundances
    current_abundances = np.array(fastchem.getElementAbundances())
    element_abundances = np.copy(current_abundances)

    # scaling the element abundances, except those of H and He
    for j in range(0, fastchem.getElementNumber()):
        if fastchem.getSpeciesSymbol(j) == 'Fe':
            element_abundances[j] *= metallicity

    # setting the FastChem with the new element abundances
    fastchem.setElementAbundances(element_abundances)
    print('[Fe/H] is set to '+str(Fe_to_H))


def set_X_to_H(fastchem, X_to_H, X):
    """Setting up abundance ratio for any given element X ([X/H]) by scaling the X abundances by 10**X_to_H.

    Args:
        fastchem: fastchem object created via pyfastchem.FastChem( str(dir_fastchem)+"input/element_abundances_solar_ext.dat",str(dir_fastchem)+"input/logK_ext.dat", 1)
        X_to_H: [X/H]: solar-scaled logarithmic abundance ratio of the element "X" to H (float)
        X: chemical symbol of the element "X" (str) (e.g., 'Fe' or 'Na')

    Returns:
        setting the element abundances in the FastChem based on the inputted [X/H]
    """

    Xabundance = 10**X_to_H
    # making a copy of the current abundances
    current_abundances = np.array(fastchem.getElementAbundances())
    element_abundances = np.copy(current_abundances)

    # scaling the element abundances, except those of H and He
    for j in range(0, fastchem.getElementNumber()):
        if fastchem.getSpeciesSymbol(j) == X:
            element_abundances[j] *= Xabundance

    # setting the FastChem with the new element abundances
    fastchem.setElementAbundances(element_abundances)
    print('[X/H] is set to '+str(X_to_H))


def TP_profile_input(pressure, temperature):
    """Inputting Temperature-Pressure profile to FastChem.

    Args:
        pressure: pressure array in bar
        temperature: temperature array in Kelvin

    Returns:
        FastChem input and output structures
    """

    input_data = pyfastchem.FastChemInput()
    output_data = pyfastchem.FastChemOutput()

    input_data.pressure = pressure
    input_data.temperature = temperature
    return input_data, output_data


def run_fastchem(fastchem, input_data, output_data):
    """Calculating the number density of all gases using FastChem 2.0.

    Args:
        fastchem: fastchem object created via pyfastchem.FastChem(str(dir_fastchem)+"input/element_abundances_solar_ext.dat",str(dir_fastchem)+"input/logK_ext.dat", 1)
        input_data: FastChem input structures
        output_data: FastChem output structures

    Returns:
        volume mixing ratios of all available gases
    """

    # Calculating the total gas number density using ideal gas law (1/cm3)
    # plist is an array of pressure in bar, tlist is an array of temperature in Kelvin
    Total_gas_number_density = number_density(
        jnp.array(input_data.pressure), jnp.array(input_data.temperature))

    # Calculating number density of all gases
    fastchem_flag = fastchem.calcDensities(input_data, output_data)
    print('FastChem reports:', pyfastchem.FASTCHEM_MSG[fastchem_flag])

    # VMR of all gases
    mixing_ratios = jnp.array(output_data.number_densities) / \
        Total_gas_number_density[:, None]
    return mixing_ratios


def vmr_species_fc2(fastchem, mixing_ratios, name):
    """Extracting the volume mixing ratio (VMR) of specific species.

    Args:
        fastchem: fastchem object created via pyfastchem.FastChem( str(dir_fastchem)+"input/element_abundances_solar_ext.dat",str(dir_fastchem)+"input/logK_ext.dat", 1)
        mixing_ratios: volume mixing ratios of all available gases calculated using run_fastchem
        name: name of the specific species (e.g., H1O1 for OH, Fe, Fe1+ for singly ionised Fe)

    Returns:
        volume mixing ratios of specific species
    """
    n_el = fastchem.getSpeciesIndex(name)
    return mixing_ratios[:, n_el]


def continuum_vmr_fc2(fastchem, mixing_ratios):
    """Extracting the volume mixing ratio (VMR) of spectral continuum-related
    chemical elements.

    Args:
        fastchem: fastchem object created via pyfastchem.FastChem( str(dir_fastchem)+"input/element_abundances_solar_ext.dat",str(dir_fastchem)+"input/logK_ext.dat", 1)
        mixing_ratios: volume mixing ratios of all available gases calculated using run_fastchem

    Returns:
        volume mixing ratios of electron, H-, H, H2, and He
    """
    vmr_el = vmr_species_fc2(fastchem, mixing_ratios, 'e-').flatten()
    vmr_H_ = vmr_species_fc2(fastchem, mixing_ratios, 'H1-').flatten()
    vmr_H = vmr_species_fc2(fastchem, mixing_ratios, 'H').flatten()
    vmr_H2 = vmr_species_fc2(fastchem, mixing_ratios, 'H2').flatten()
    vmr_He = vmr_species_fc2(fastchem, mixing_ratios, 'He').flatten()
    return vmr_el, vmr_H_, vmr_H, vmr_H2, vmr_He


def mmr_species_fc2(SpeciesIndex, mixing_ratios, SpeciesMolecularWeight):
    """(Using scan) Extracting the mass mixing ratio (MMR) of specific species.

    Args:
        SpeciesIndex: index of the specific species (e.g., SpeciesIndex = fastchem.getSpeciesIndex('Fe')) (int)
        mixing_ratios: volume mixing ratios of all available gases calculated using run_fastchem [N_layer x N_species]
        SpeciesMolecularWeight: atomic/molecular weight of each species [N_species]

    Returns:
        mass mixing ratios of specific species [N_layer]
    """
    def floop(i_layer, SpeciesMMRarr_layer):
        MMRarr = mixing_ratios[i_layer] * SpeciesMolecularWeight
        SpeciesMMRarr_layer = MMRarr[SpeciesIndex] / np.sum(MMRarr)
        i_layer = i_layer + 1
        return i_layer, SpeciesMMRarr_layer
    i, SpeciesMMRarr = scan(floop, 0, np.zeros(len(mixing_ratios)))
    return SpeciesMMRarr


def get_H_He_HH_VMR(fastchem, mixing_ratios):
    """Extract the VMR list of neutral hydrogen, neutral helium, and H2 molecule.

    Args:
        Parr: total pressure array [N_layer]
        fastchem: fastchem object created via pyfastchem.FastChem( str(dir_fastchem)+"input/element_abundances_solar_ext.dat",str(dir_fastchem)+"input/logK_ext.dat", 1)
        mixing_ratios: volume mixing ratios of all available gases calculated using run_fastchem [N_layer x N_species]

    Returns:
        array of volume mixing ratios of electron, H, He, and H2 [N_layer x 3]
    """
    H_He_HH_VMR = np.stack([ \
          mixing_ratios[:, fastchem.getSpeciesIndex('H')], \
          mixing_ratios[:, fastchem.getSpeciesIndex('He')], \
          mixing_ratios[:, fastchem.getSpeciesIndex('H2')]]).T
    return H_He_HH_VMR
    

def get_H_He_HH_pressure(Parr, fastchem, mixing_ratios):
    """Extract the partial pressure arrays for three species that dominate the pressure broadening (neutral hydrogen, neutral helium, and H2 molecule).

    Args:
        Parr: total pressure array [N_layer]
        fastchem: fastchem object created via pyfastchem.FastChem( str(dir_fastchem)+"input/element_abundances_solar_ext.dat",str(dir_fastchem)+"input/logK_ext.dat", 1)
        mixing_ratios: volume mixing ratios of all available gases calculated using run_fastchem [N_layer x N_species]

    Returns:
        PH: Partial pressure array of neutral hydrogen (H) [N_layer]
        PHe: Partial pressure array of neutral helium (He) [N_layer]
        PHH: Partial pressure array of molecular hydrogen (H2) [N_layer]
    """
    PH = Parr * mixing_ratios[:, fastchem.getSpeciesIndex('H')]
    PHe = Parr * mixing_ratios[:, fastchem.getSpeciesIndex('He')]
    PHH = Parr * mixing_ratios[:, fastchem.getSpeciesIndex('H2')]
    return PH, PHe, PHH

