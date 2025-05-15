from exojax.spec.exomolhr import MdbExomolHR

def test_mdbexomolhr_online():
    mdb = MdbExomolHR("12C-16O2", [0.0,2000.0])


def test_mdbexomolhr_exomol_comparison():
    from exojax.test.emulate_mdb import mock_wavenumber_grid
    from exojax.test.emulate_mdb import mock_mdbExomol
    import matplotlib.pyplot as plt
    nus, wav, res = mock_wavenumber_grid()

    mdb = MdbExomolHR("1H2-16O", nus)
    mdb_orig = mock_mdbExomol("H2O")
    
    plt.plot(mdb.nu_lines, mdb.line_strength_ref_original, ".", label="ExoMolHR")
    plt.plot(mdb_orig.nu_lines, mdb_orig.line_strength(1000.0), "+", alpha=0.5, label="ExoMol")
    plt.yscale("log")
    plt.xlabel("Wavenumber (cm-1)")
    plt.ylabel("Line Strength (cm/molecule)")
    plt.legend()
    plt.show()
    print(mdb.line_strength_ref_original)
    print(mdb_orig.line_strength(1000.0))


if __name__ == "__main__":
    #test_mdbexomolhr_online()
    test_mdbexomolhr_exomol_comparison()
    
    print("Test passed.")