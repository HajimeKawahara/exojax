"""filenames of test data used in unit tests

"""
# exomol moldb template used in unit tests
# exojax.src.test.generate should make this file
TESTDATA_moldb_CO_HITEMP = "moldb_co_hitemp.pickle"
TESTDATA_moldb_CO_HITEMP_SINGLE_ISOTOPE = "moldb_co_hitemp_single_isotope.pickle"
TESTDATA_moldb_VALD = "moldb_vald.pickle"

# cross section references
TESTDATA_CO_EXOMOL_LPF_XS_REF = "lpf_test_ref.txt"
TESTDATA_CO_EXOMOL_MODIT_XS_REF = "modit_test_ref.txt"
TESTDATA_CO_EXOMOL_PREMODIT_XS_REF = "premodit_test_ref.txt"

# cross section references
TESTDATA_CO_HITEMP_LPF_XS_REF = "lpf_test_hitemp_ref.txt"
TESTDATA_CO_HITEMP_MODIT_XS_REF = "modit_test_hitemp_ref.txt" #Pself=P
TESTDATA_CO_HITEMP_MODIT_XS_REF_AIR = "modit_test_hitemp_ref_air.txt" #Pself=0.0
TESTDATA_CO_HITEMP_PREMODIT_XS_REF = "premodit_test_hitemp_ref.txt"

# emission spectra references
TESTDATA_CO_EXOMOL_LPF_EMISSION_REF = "lpf_rt_test_ref.txt"
TESTDATA_CO_HITEMP_LPF_EMISSION_REF = "lpf_rt_test_hitemp_ref.txt"
TESTDATA_CO_EXOMOL_MODIT_EMISSION_REF = "modit_rt_test_ref.txt"
TESTDATA_CO_HITEMP_MODIT_EMISSION_REF = "modit_rt_test_hitemp_ref.txt"
TESTDATA_CO_EXOMOL_PREMODIT_EMISSION_REF = "premodit_rt_test_ref.txt"
TESTDATA_CO_HITEMP_PREMODIT_EMISSION_REF = "premodit_rt_test_hitemp_ref.txt"
TESTDATA_VALD_MODIT_EMISSION_REF = "modit_rt_test_vald_ref.txt"

#sample spectra
SAMPLE_SPECTRA_CO = "spectrum_co.txt"
SAMPLE_SPECTRA_CH4 = "spectrum_ch4.txt" #exojax version 1.0
SAMPLE_SPECTRA_CH4_NEW = "spectrum_ch4_new.txt" #generate_methane_spectrum.py

#sample transmission spectra
SAMPLE_TRANSMISSION_CH4 = "transmission_ch4.txt"

#test data par file
TESTDATA_CO_HITEMP_PARFILE = "05_HITEMP_SAMPLE.par"

#test data CIA H2-H2 (4300-4400 cm-1)
TESTDATA_H2_H2_CIA = "H2-H2_TEST.cia" 