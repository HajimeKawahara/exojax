from exojax.spec.exomolhr import MdbExomolHR

def test_mdbexomolhr_online():
    mdb = MdbExomolHR("12C-16O2", [0.0,2000.0])

if __name__ == "__main__":
    test_mdbexomolhr_online()
    print("Test passed.")