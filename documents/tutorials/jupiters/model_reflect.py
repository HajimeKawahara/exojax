import jax.numpy as jnp

def unpack_params(params):
        multiple_factor = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 10000.0, 0.01, 1.0])
        par = params * multiple_factor
        log_fsed = par[0]
        sigmag = par[1]
        log_Kzz = par[2]
        vrv = par[3]
        vv = par[4]
        _broadening = par[5]
        const_mmr_ch4 = par[6]
        factor = par[7]
        fsed = 10**log_fsed
        Kzz = 10**log_Kzz

        return fsed, sigmag, Kzz, vrv, vv, _broadening, const_mmr_ch4, factor

def spectral_model(params):
    vv, factor, broadening, asymmetric_parameter, single_scattering_albedo, dtau = (
        atmospheric_model(params)
    )
    # velocity
    vpercp = (vrv_fixed + vv) / c
    incoming_flux = jnp.interp(nusjax, nusjax_solar * (1.0 + vpercp), solspecjax)

    Fr = art.run(
        dtau,
        single_scattering_albedo,
        asymmetric_parameter,
        reflectivity_surface,
        incoming_flux,
    )

    std = resolution_to_gaussian_std(broadening)
    Fr_inst = sop.ipgauss(Fr, std)
    Fr_samp = sop.sampling(Fr_inst, vv, nus_obs)
    return factor * Fr_samp

