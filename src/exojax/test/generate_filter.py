
def generate_filter_sdss_g():
    import pandas as pd
    from exojax.utils.photometry import download_filter_from_svo
    from exojax.utils.photometry import download_zero_magnitude_flux_from_svo
    from exojax.test.data import TESTDATA_FILTER_SDSS_G

    filter_name = "SLOAN/SDSS.g"
    nu_ref, transmission_ref = download_filter_from_svo(filter_name)
    pd.DataFrame({"nu": nu_ref, "transmission": transmission_ref}).to_csv(TESTDATA_FILTER_SDSS_G)
    nu0, f0 = download_zero_magnitude_flux_from_svo(filter_name, unit="cm-1")
    return nu0, f0

if __name__ == "__main__":
    nu0, f0 = generate_filter_sdss_g()
    print(nu0, f0)
    #nu0 = 21258.71108019469
    #f0 = 1.20623691665283e-09
    