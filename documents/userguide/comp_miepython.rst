Compare geometric cloud opacity with miepython
==============================================

We use the MgSiO3 refractive index from the particulate database to
compare Mie scattering with the large-particle geometric approximation.
The refractive index in ExoJAX is ``m = n + ik``, where positive ``k``
represents absorption.

.. code:: ipython3

    import os

    os.environ["MIEPYTHON_USE_JIT"] = "1"  # set before importing miepython

    import numpy as np
    import miepython

    from exojax.database.pardb import PdbCloud

    pdb = PdbCloud("MgSiO3", path=".database/particulates/virga")
    m = pdb.refraction_index
    wavelength_nm = pdb.refraction_index_wavelength_nm


.. parsed-literal::

    .database/particulates/virga/virga.zip  exists. Remove it if you wanna re-download and unzip.
    Refractive index file found:  .database/particulates/virga/MgSiO3.refrind
    Miegrid file does not exist at  .database/particulates/virga/miegrid_lognorm_MgSiO3.mg.npz
    Generate miegrid file using pdb.generate_miegrid if you use Mie scattering


Select the tabulated wavelength nearest 2 micrometers. Use a geometric
mean radius of 10 micrometers and a geometric standard deviation of 1.5.

.. code:: ipython3

    iwav = np.argmin(np.abs(wavelength_nm - 2000.0))
    wavelength = wavelength_nm[iwav]
    rg = 1.0e-3  # radius in cm: 10 micrometers
    rg_nm = rg * 1.0e7
    sigmag = 1.5
    N0 = 1.0  # particle number density in cm^-3
    print("Wavelength (nm):", wavelength)


.. parsed-literal::

    Wavelength (nm): 2009.9999999999998


First compute the dimensionless efficiencies for a single sphere. When
calling miepython directly, use its ``n - ik`` convention and pass the
diameter and wavelength in the same units. Here both lengths are in nm.

.. code:: ipython3

    qext, qsca, qback, g = miepython.efficiencies(
        np.conj(m[iwav]), d=2.0 * rg_nm, lambda0=wavelength
    )
    print("Qext, Qsca, Qback, g:", qext, qsca, qback, g)


.. parsed-literal::

    Qext, Qsca, Qback, g: 2.0994639257611847 2.0105214275159686 1.2161625227051667 0.7714452710932014


For a lognormal distribution, ExoJAX integrates the single-particle
results on a radius grid. ``mie_lognormal`` accepts the ExoJAX
``n + ik`` convention and converts it internally. Its wavelength,
geometric mean radius, and integration radii are all in nm.
``auto_rgrid`` selects the integration range; check convergence when
using unusually broad distributions.

.. code:: ipython3

    from exojax.database.mie import auto_rgrid, mie_lognormal

    rgrid = auto_rgrid(rg_nm, sigmag)
    coeff = mie_lognormal(m[iwav], wavelength, sigmag, rg_nm, N0, rgrid)
    Bext, Bsca, Babs, G, Bpr, Bback, Bratio = coeff

The seven stored fields retain the existing miegrid convention.
``Bext``, ``Bsca``, ``Babs``, ``Bpr``, and ``Bback`` are coefficients in
inverse megameters (Mm^-1); ``G`` is the dimensionless
scattering-weighted asymmetry factor. ``Bratio`` is a legacy
size-integrated backscatter-ratio field. For comparison with a
per-particle cross section, convert ``Bext`` to cm^-1 and divide by
``N0``.

.. code:: ipython3

    beta_ext = Bext * 1.0e-8  # extinction coefficient in cm^-1
    sigma_ext = beta_ext / N0  # cross section in cm^2 per particle

The geometric approximation assumes an extinction efficiency of 2 and
averages the projected area over the same distribution. It is
appropriate when the particles contributing most of the opacity are much
larger than the wavelength. It need not agree with Mie scattering for
small particles or near spectral resonances.

.. code:: ipython3

    from exojax.atm.amclouds import geometric_radius

    rgeo = float(geometric_radius(rg, sigmag))
    sigma_geo = 2.0 * np.pi * rgeo**2
    print("Mie cross section (cm^2):", sigma_ext)
    print("Geometric cross section (cm^2):", sigma_geo)
    print("Mie / geometric:", sigma_ext / sigma_geo)


.. parsed-literal::

    Mie cross section (cm^2): 9.449965265215964e-06
    Geometric cross section (cm^2): 8.729263257013955e-06
    Mie / geometric: 1.0825616076617834
