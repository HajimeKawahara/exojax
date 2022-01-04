#Several functions to simplify the integration of FastChem2 to Exojax, adopted from https://github.com/exoclime/FastChem/blob/master/python/
# Dictionary of the name of chemical species available in the FastChem2
# fc_dict=np.array(["P(bar)", "T(k)", "n_<tot>(cm-3)","n_g(cm-3)", "m(u)", "e-", "Al", "Ar", "C", "Ca", "Cl",
#                    "Co", "Cr", "Cu", "F", "Fe", "Ge", "H", "He", "K", "Mg", "Mn", "N", "Na", "Ne", "Ni", "O",
#                    "P", "S", "Si", "Ti", "V", "Zn", "Li", "Be", "B", "Sc", "Ga", "As", "Se", "Rb", "Sr", "Y",
#                    "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "Cs", "Ba", "La",
#                    "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta",
#                    "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Th", "U", "Al1Cl1", "Al1Cl1F1",
#                    "Al1Cl1F2", "Al1Cl1O1", "Al1Cl2", "Al1Cl2F1", "Al1Cl3", "Al1F1", "Al1F1O1", "Al1F2",
#                    "Al1F2O1", "Al1F3", "Al1F4Na1", "Al1H1", "Al1H1O1_1", "Al1H1O1_2", "Al1H1O2", "Al1N1",
#                    "Al1O1", "Al1O2", "Al1S1", "Al2", "Al2Cl6", "Al2F6", "Al2O1", "Al2O2", "C1Al1", "C1Cl1",
#                    "C1Cl1F1O1", "C1Cl1F3", "C1Cl1N1", "C1Cl1O1", "C1Cl2", "C1Cl2F2", "C1Cl2O1", "C1Cl3",
#                    "C1Cl3F1", "C1Cl4", "C1F1", "C1F1N1", "C1F1O1", "C1F2", "C1F2O1", "C1F3", "C1F4", "C1F4O1",
#                    "C1F8S1", "C1H1", "C1H1Cl1", "C1H1Cl3", "C1H1F1", "C1H1F1O1", "C1H1F3", "C1H1N1_1",
#                    "C1H1N1_2", "C1H1N1O1", "C1H1O1", "C1H1P1", "C1H2", "C1H2Cl2", "C1H2Cl1F1", "C1H2F2",
#                    "C1H2O1", "C1H3", "C1H3Cl1", "C1H3F1", "C1H4", "C1H4O2", "C1K1N1", "C1N1", "C1N1Na1",
#                    "C1N1O1", "C1N2_cnn", "C1N2_ncn", "C1O1", "C1O1S1", "C1O2", "C1P1", "C1S1", "C1S2", "C1Si1",
#                    "C1Si2", "C2", "C2Cl2", "C2Cl4", "C2Cl6", "C2Cr1", "C2F2", "C2F3N1", "C2F4", "C2F6", "C2H1",
#                    "C2H1Cl1", "C2H1F1", "C2H2", "C2H2O2", "C2H2O4", "C2H3Cl1O2", "C2H4", "C2H4O1", "C2H4O3",
#                    "C2H6O2", "C2K2N2", "C2N1", "C2N1O1", "C2N2", "C2N2Na2", "C2Si1", "C2Si2", "C2O1", "C2Ti1",
#                    "C2V1", "C3", "C3H1", "C3N2O1", "C3O2", "C4", "C4H6O4", "C4N2", "C4Ni1O4", "C4Ti1", "C4V1",
#                    "C5", "C5Fe1O5", "Ca1Cl1", "Ca1Cl2", "Ca1F1", "Ca1F2", "Ca1H1", "Ca1H1O1", "Ca1H2O2", "Ca1O1",
#                    "Ca1S1", "Ca2", "Cl1Co1", "Cl1Cu1", "Cl1F1", "Cl1F1Mg1", "Cl1F1O2S1", "Cl1F1O3", "Cl1F2O1P1",
#                    "Cl1F3", "Cl1F3Si1", "Cl1F5", "Cl1F5S1", "Cl1Fe1", "Cl1H1", "C1H1Cl1F2", "C1H1Cl2F1", 
#                    "Cl1H1O1", "Cl1H3Si1", "Cl1K1", "Cl1Mg1", "Cl1N1O1", "Cl1N1O2", "Cl1Na1", "Cl1Ni1", "Cl1O1",
#                    "Cl1O1Ti1", "Cl1O2", "Cl1O3", "Cl1P1", "Cl1S1", "Cl1S2", "Cl1Si1", "Cl1Ti1", "Cl2", "Cl2Co1",
#                    "Cl2F1O1P1", "Cl2Fe1", "Cl2H2Si1", "Cl2K2", "Cl2Mg1", "Cl2Na2", "Cl2Ni1", "Cl2O1_clocl",
#                    "Cl2O1_clclo", "Cl2O1Ti1", "Cl2O2_clo2cl", "Cl2O2_cloclo","Cl2O2S1", "Cl2S1", "Cl2Si1",
#                    "Cl2Ti1", "Cl3Co1", "Cl3Cu3", "Cl3F1Si1", "Cl3Fe1", "Cl3H1Si1", "Cl3O1P1", "Cl3P1",
#                    "Cl3P1S1", "Cl3Si1", "Cl3Ti1", "Cl4Co2", "Cl4Fe2", "Cl4Mg2", "Cl4Si1", "Cl4Ti1", "Cl5P1",
#                    "Cl6Fe2", "Co1F2", "Cr1H1", "Cr1N1", "Cr1O1", "Cr1O2", "Cr1O3", "Cu1F1", "Cu1F2", "Cu1H1",
#                    "Cu1O1", "Cu1S1", "Cu2", "F1Fe1", "F1H1", "F1H1O1", "F1H1O3S1", "F1H3Si1", "F1K1", "F1Mg1",
#                    "F1N1", "F1N1O1", "F1N1O2", "F1N1O3", "F1Na1", "F1O1", "F1O1Ti1", "F1O2_ofo", "F1O2_foo",
#                    "F1P1", "F1P1S1", "F1S1", "F1Si1", "F1Ti1", "F2", "F2Fe1", "F2H2", "F2H2Si1", "F2K2",
#                    "F2Mg1", "F2N1", "F2N2cis", "F2N2trans", "F2Na2", "F2O1", "F2O1S1", "F2O1Si1", "F2O1Ti1",
#                    "F2O2", "F2O2S1", "F2P1", "F2S1", "F2S2_1", "F2S2_2", "F2Si1", "F2Ti1", "F3Fe1", "F3H1Si1",
#                    "F3H3", "F3N1", "F3N1O1", "F3O1P1", "F3P1", "F3P1S1", "F3S1", "F3Si1", "F3Ti1", "F4H4",
#                    "F4Mg2", "F4N2", "F4S1", "F4Si1", "F4Ti1", "F5H5", "F5P1", "F5S1", "F6H6", "F6S1", "F7H7",
#                    "F10S2", "Fe1H1", "Fe1H2O2", "Fe1O1", "Fe1S1", "H1K1", "H1K1O1", "H1Mg1", "H1Mg1O1",
#                    "H1Mn1", "H1N1", "H1N1O1", "H1N1O2cis", "H1N1O2trans", "H1N1O3", "H1Na1", "H1Na1O1",
#                    "H1Ni1", "H1O1", "H1O2", "H1P1", "H1S1", "H1Si1", "H1Ti1", "H2", "H2K2O2", "H2Mg1O2",
#                    "H2N1", "H2N2", "H2Na2O2", "H2O1", "H2O2", "H2O4S1", "H2P1", "H2S1", "H2Si1", "H3N1",
#                    "H3P1", "H3Si1", "H4N2", "H4Si1", "K1O1", "K2", "K2O4S1", "Mg1N1", "Mg1O1", "Mg1S1",
#                    "Mg2", "Mn1O1", "Mn1S1", "N1O1", "N1O2", "N1O3", "N1P1", "N1S1", "N1Si1", "N1Si2",
#                    "N1Ti1", "N1V1", "N2", "N2O1", "N2O3", "N2O4", "N2O5", "N3", "Na1O1", "Na2", "Na2O4S1",
#                    "Ni1O1", "Ni1S1", "O1P1", "O1S1", "O1S2", "O1Si1", "O1Ti1", "O1V1", "O2", "O2P1", "O2S1",
#                    "O2Si1", "O2Ti1", "O2V1", "O3", "O3S1", "O6P4", "O10P4", "P1S1", "P2", "P4", "P4S3",
#                    "S1Si1", "S1Ti1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "Si2", "Si3", "Al1+", "Al1-",
#                    "Al1Cl1+", "Al1Cl1F1+", "Al1Cl2+", "Al1Cl2-", "Al1F1+", "Al1F2+", "Al1F2-", "Al1F2O1-",
#                    "Al1F4-", "Al1H1O1+", "Al1H1O1-", "Al1O1+", "Al1O1-", "Al1O2-", "Al2O1+", "Al2O2+", 
#                    "Ar1+", "C1+", "C1-", "C1F1+", "C1F2+", "C1F3+", "C1H1+", "C1H1-", "C1H1O1+", "C1N1+",
#                    "C1N1-", "C1O2-", "C2-", "Ca1+", "Ca1H1O1+", "Cl1+", "Cl1Mg1+", "Cl1S1+", "Cl1-", "Cl2S1+",
#                    "Co1+", "Co1-", "Cr1+", "Cr1-", "Cu1+", "Cu1-", "F1+", "F1-", "F1Mg1+", "F1P1+", "F1P1-", 
#                    "F1S1+", "F1S1-", "F2K1-", "F2Mg1+", "F2Na1-", "F2P1+", "F2P1-", "F2S1+", "F2S1-", "F3S1+",
#                    "F3S1-", "F4S1+", "F4S1-", "F5S1+", "F5S1-", "F6S1-", "Fe1+", "Fe1-", "H1+", "H1-",
#                    "H1K1O1+", "H1Mg1O1+", "H1Na1O1+", "H1O1+", "H1O1-", "H1S1-", "H1Si1+", "H2+", "H2-",
#                    "H3O1+", "He1+", "K1+", "K1-", "K1O1-", "Mg1+", "Mn1+", "N1+", "N1-", "N1O1+", "N1O2-",
#                    "N2+", "N2-", "N2O1+", "Na1+", "Na1-", "Na1O1-", "Ne1+", "Ni1+", "Ni1-", "O1+", "O1-",
#                    "O2+", "O2-", "P1+", "P1-", "S1+", "S1-", "Si1+", "Si1-", "Ti1+", "Ti1-", "V1+", "V1-",
#                    "Zn1+", "Zn1-", "Li1+", "Li1-", "Li1++", "Be1+", "Be1++", "B1+", "B1-", "B1++", "C1++",
#                    "N1++", "O1++", "F1++", "Ne1++", "Na1++", "Mg1++", "Al1++", "Si1++", "P1++", "S1++",
#                    "Cl1++", "Ar1++", "K1++", "Ca1-", "Ca1++", "Sc1+", "Sc1-", "Sc1++", "Ti1++", "V1++",
#                    "Cr1++", "Mn1++", "Fe1++", "Co1++", "Ni1++", "Cu1++", "Zn1++", "Ga1+", "Ga1-", "Ga1++",
#                    "Ge1+", "Ge1-", "Ge1++", "As1+", "As1-", "As1++", "Se1+", "Se1-", "Se1++", "Rb1+",
#                    "Rb1-", "Rb1++", "Sr1+", "Sr1-", "Sr1++", "Y1+", "Y1-", "Y1++", "Zr1+", "Zr1-", "Zr1++",
#                    "Nb1+", "Nb1-", "Nb1++", "Mo1+", "Mo1-", "Mo1++", "Ru1+", "Ru1-", "Ru1++", "Rh1+", "Rh1-",
#                    "Rh1++", "Pd1+", "Pd1-", "Pd1++", "Ag1+", "Ag1-", "Ag1++", "Cd1+", "Cd1++", "In1+",
#                    "In1-", "In1++", "Sn1+", "Sn1-", "Sn1++", "Sb1+", "Sb1-", "Sb1++", "Te1+", "Te1-",
#                    "Te1++", "Cs1+", "Cs1-", "Cs1++", "Ba1+", "Ba1-", "Ba1++", "La1+", "La1-", "La1++",
#                    "Ce1+", "Ce1-", "Ce1++", "Pr1+", "Pr1-", "Pr1++", "Nd1+", "Nd1-", "Nd1++", "Sm1+",
#                    "Sm1-", "Sm1++", "Eu1+", "Eu1-", "Eu1++", "Gd1+", "Gd1-", "Gd1++", "Tb1+", "Tb1-",
#                    "Tb1++", "Dy1+", "Dy1-", "Dy1++", "Ho1+", "Ho1-", "Ho1++", "Er1+", "Er1-", "Er1++", 
#                    "Tm1+", "Tm1-", "Tm1++", "Yb1+", "Yb1++", "Lu1+", "Lu1-", "Lu1++", "Hf1+", "Hf1-", 
#                    "Hf1++", "Ta1+", "Ta1-", "Ta1++", "W1+", "W1-", "W1++", "Re1+", "Re1-", "Re1++",
#                    "Os1+", "Os1-", "Os1++", "Ir1+", "Ir1-", "Ir1++", "Pt1+", "Pt1-", "Pt1++", "Au1+",
#                    "Au1-", "Au1++", "Hg1+", "Hg1++", "Tl1+", "Tl1-", "Tl1++", "Pb1+", "Pb1-", "Pb1++",
#                    "Bi1+", "Bi1-", "Bi1++", "Th1+", "Th1++", "U1+", "U1++"])

import pyfastchem
from exojax.atm.idealgas import number_density

#Getting the volume mixing ratio (VMR) of specific chemical element
def vmr_species_fc2(name):
    n_el=fastchem.getSpeciesIndex(name)
    return mixing_ratios[:,n_el]

#Getting the VMR of spectral continuum-related chemical element
def continuum_vmr_fc2():
    vmr_el= vmr_species_fc2('e-').flatten()
    vmr_H_= vmr_species_fc2('H1-').flatten()
    vmr_H = vmr_species_fc2('H').flatten()
    vmr_H2= vmr_species_fc2('H2').flatten()
    vmr_He= vmr_species_fc2('He').flatten()
    return vmr_el, vmr_H_, vmr_H, vmr_H2, vmr_He

#Setting up C/O value
def set_C_to_O(fastchem,c_to_o):
    #we need to know the indices for O and C from FastChem
    index_C = fastchem.getSpeciesIndex('C')
    index_O = fastchem.getSpeciesIndex('O')

    #make a copy of the solar abundances from FastChem
    solar_abundances = np.array(fastchem.getElementAbundances())
    element_abundances = np.copy(solar_abundances)

    #set the O abundance as a function of the C/O (Molliére et al. 2015)
    element_abundances[index_O] = element_abundances[index_C]/c_to_o

    fastchem.setElementAbundances(element_abundances)
    print ("C/O is set to "+str(c_to_o))
    
#Inputting Temperature-Pressure profile to FastChem, from top to bottom layer    
def TP_profile_input(pressure,temperature):
    input_data = pyfastchem.FastChemInput()
    output_data = pyfastchem.FastChemOutput()

    input_data.pressure= pressure
    input_data.temperature= temperature
    return input_data,output_data
  
#Run fastchem  
def run_fastchem():
    #Calculating the total gas number density using ideal gas law (1/cm3)
    #plist is an array of pressure in bar, tlist is an array of temperature in Kelvin
    Total_gas_number_density=  number_density(jnp.array(input_data.pressure),jnp.array(input_data.temperature))

    #Calculating number density of all gases
    fastchem_flag = fastchem.calcDensities(input_data, output_data)
    print("FastChem reports:", pyfastchem.FASTCHEM_MSG[fastchem_flag])
    
    #VMR of all gases
    mixing_ratios = jnp.array(output_data.number_densities)/Total_gas_number_density [:,None]
    return mixing_ratios
