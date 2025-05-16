from exojax.spec.exomolhr import XdbExomolHR

def test_mdbexomolhr_online():
    temperature = 1000.0
    mdb = XdbExomolHR("12C-16O2", [0.0,2000.0], temperature)


def test_mdbexomolhr_exomol_comparison():
    from exojax.test.emulate_mdb import mock_wavenumber_grid
    from exojax.test.emulate_mdb import mock_mdbExomol
    import matplotlib.pyplot as plt
    nus, _, _ = mock_wavenumber_grid()
    temperature = 296.0
    
    mdb = XdbExomolHR("1H2-16O", nus, temperature)
    mdb_orig = mock_mdbExomol("H2O")
    
    plt.plot(mdb.nu_lines, mdb.line_strength, ".", label="ExoMolHR")
    plt.plot(mdb_orig.nu_lines, mdb_orig.line_strength(temperature), "+", alpha=0.5, label="ExoMol")
    plt.yscale("log")
    plt.xlabel("Wavenumber (cm-1)")
    plt.ylabel("Line Strength (cm/molecule)")
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    #test_mdbexomolhr_online()
    test_mdbexomolhr_exomol_comparison()
    
    print("Test passed.")