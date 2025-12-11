import numpy as np
import jax.numpy as jnp


def test_from_external_exomolop_basic(monkeypatch):
    # Synthetic CKD grid dimensions
    nT, nP, Ng, nBands = 2, 2, 3, 2

    # Build synthetic data returned by provider.exomolop.load_ckd
    temperatures = np.array([500.0, 1000.0], dtype=np.float32)
    pressures = np.array([0.1, 1.0], dtype=np.float32)
    samples = np.array([0.2, 0.5, 0.8], dtype=np.float32)  # g-grid
    weights = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
    nu_centers = np.array([1000.0, 1005.0], dtype=np.float32)

    xsgrid = np.zeros((nT, nP, Ng, nBands), dtype=np.float32)
    for it in range(nT):
        for ip in range(nP):
            for ig in range(Ng):
                for ib in range(nBands):
                    # Unique, positive values per index
                    xsgrid[it, ip, ig, ib] = (
                        (it + 1) * 10.0 + (ip + 1) + ig * 0.1 + ib * 0.01
                    )

    def fake_load_ckd(_):
        # Return values follow provider.exomolop.load_ckd contract
        molecule, mol_mass = "H2O", 18.0
        return (
            xsgrid.copy(),
            samples.copy(),
            weights.copy(),
            temperatures.copy(),
            pressures.copy(),
            nu_centers.copy(),
            molecule,
            mol_mass,
        )

    # Monkeypatch before importing the API path that uses it
    import exojax.provider.exomolop as exomolop_provider

    monkeypatch.setattr(exomolop_provider, "load_ckd", fake_load_ckd)

    from exojax.opacity.ckd.api import OpaCKD

    ckd = OpaCKD.from_external("exomolop", "dummy.h5")

    # Basic shape checks
    assert ckd.ready is True
    assert ckd.Ng == Ng
    assert ckd.ckd_info.log_kggrid.shape == (nT, nP, Ng, nBands)
    assert ckd.ckd_info.ggrid.shape == (Ng,)
    assert ckd.ckd_info.nu_bands.shape == (nBands,)

    # Interpolation at exact grid points should equal the original slice
    out = ckd.xsarray_ckd(temperatures[0], pressures[0])  # shape (Ng, nBands)
    np.testing.assert_allclose(np.asarray(out), xsgrid[0, 0, :, :], rtol=1e-6, atol=0.0)

    # Tensor interpolation across layers
    T_arr = jnp.array(temperatures)
    P_arr = jnp.array(pressures)
    xst = ckd.xstensor_ckd(T_arr, P_arr)  # (2, Ng, nBands)
    np.testing.assert_allclose(np.asarray(xst[0]), xsgrid[0, 0], rtol=1e-6, atol=0.0)
    np.testing.assert_allclose(np.asarray(xst[1]), xsgrid[1, 1], rtol=1e-6, atol=0.0)

