Ackerman and Marley Cloud Model
===============================

Here, we try to compute a cloud opacity using Ackerman and Marley Model.
We consider enstatite (MgSiO3) and Fe clouds.

.. code:: ipython3

    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

Setting a simple atmopheric model. We need the density of atmosphere.

.. code:: ipython3

    from exojax.atm.atmprof import pressure_layer_logspace
    from exojax.utils.constants import kB, m_u
    
    NP = 100
    Parr, dParr, k = pressure_layer_logspace(log_pressure_top=-8, log_pressure_btm=4.0, nlayer=NP)
    alpha = 0.097
    T0 = 1200.
    Tarr = T0 * (Parr)**alpha
    
    mu = 2.0  # mean molecular weight
    R = kB / (mu * m_u)
    rho = Parr / (R * Tarr)


The solar abundance can be obtained using utils.zsol.nsol. Here, we
assume a maximum VMR for MgSiO3 and Fe from solar abundance.

.. code:: ipython3

    from exojax.utils.zsol import nsol
    
    n = nsol()  #solar abundance
    VMR_enstatite = np.min([n["Mg"], n["Si"], n["O"] / 3])
    VMR_Fe = n["Fe"]


Vapor saturation pressures can be obtained using atm.psat

.. code:: ipython3

    from exojax.atm.psat import Psat_enstatite_AM01, Psat_Fe_solid
    
    P_enstatite = Psat_enstatite_AM01(Tarr)
    P_fe_sol = Psat_Fe_solid(Tarr)


Compute a cloud base pressure.

.. code:: ipython3

    from exojax.atm.amclouds import get_Pbase
    
    Pbase_enstatite = get_Pbase(Parr, P_enstatite, VMR_enstatite)
    Pbase_Fe_sol = get_Pbase(Parr, P_fe_sol, VMR_Fe)


The cloud base is located at the intersection of a TP profile and the
vapor saturation puressure devided by VMR.

.. code:: ipython3

    plt.plot(Tarr, Parr, color="black", ls="dashed", label="T - P profile")
    plt.plot(Tarr,
             P_enstatite / VMR_enstatite,
             label="$P_{sat}/\\xi$ (enstatite)",
             color="gray")
    plt.axhline(Pbase_enstatite, color="gray", alpha=0.7, ls="dotted")
    plt.text(500, Pbase_enstatite * 0.8, "cloud base (enstatite)", color="gray")
    
    plt.plot(Tarr, P_fe_sol / VMR_Fe, label="$P_{sat}/\\xi$ (Fe)", color="black")
    plt.axhline(Pbase_Fe_sol, color="black", alpha=0.7, ls="dotted")
    plt.text(500, Pbase_Fe_sol * 0.8, "cloud base (Fe)", color="black")
    
    plt.yscale("log")
    plt.ylim(1.e-7, 10000)
    plt.xlim(0, 2700)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.xlabel("Temperature (K)")
    plt.ylabel("Pressure (bar)")
    plt.savefig("pbase.pdf", bbox_inches="tight", pad_inches=0.0)
    plt.savefig("pbase.png", bbox_inches="tight", pad_inches=0.0)
    plt.show()



.. image:: Ackerman_and_Marley_cloud_model_files/Ackerman_and_Marley_cloud_model_12_0.png


Compute VMRs of clouds. Because Parr is an array, we apply jax.vmap to
atm.amclouds.VMRclouds.

.. code:: ipython3

    from jax import vmap
    from exojax.atm.amclouds import VMRcloud
    
    get_VMRc = vmap(VMRcloud, (0, None, None, None), 0)
    
    fsed = 3
    VMRbase_enstatite = VMR_enstatite
    VMRc_enstatite = get_VMRc(Parr, Pbase_enstatite, fsed, VMR_enstatite)
    
    VMRbase_Fe = VMR_Fe
    VMRc_Fe = get_VMRc(Parr, Pbase_Fe_sol, fsed, VMR_Fe)


Here is the VMR distribution.

.. code:: ipython3

    plt.figure()
    plt.gca().get_xaxis().get_major_formatter().set_powerlimits([-3, 3])
    plt.plot(VMRc_enstatite, Parr, color="gray", label="VMR (enstatite)")
    plt.plot(VMRc_Fe, Parr, color="black", ls="dashed", label="VMR (Fe)")
    
    plt.yscale("log")
    plt.ylim(1.e-7, 10000)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.xlabel("VMR (clouds)")
    plt.ylabel("Pressure (bar)")
    plt.savefig("vmrcloud.pdf", bbox_inches="tight", pad_inches=0.0)
    plt.savefig("vmrcloud.png", bbox_inches="tight", pad_inches=0.0)
    plt.show()



.. image:: Ackerman_and_Marley_cloud_model_files/Ackerman_and_Marley_cloud_model_16_0.png


Compute dynamic viscosity in H2 atmosphere (cm/g/s)

.. code:: ipython3

    from exojax.atm.viscosity import eta_Rosner, calc_vfactor
    
    T = np.logspace(np.log10(1000), np.log10(2000))
    vfactor, Tr = calc_vfactor("H2")
    eta = eta_Rosner(T, vfactor)


.. code:: ipython3

    plt.plot(T, eta)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Dynamic viscosity (cm/g/s)")
    plt.show()



.. image:: Ackerman_and_Marley_cloud_model_files/Ackerman_and_Marley_cloud_model_19_0.png


The pressure scale height can be computed using atm.atmprof.Hatm.

.. code:: ipython3

    from exojax.atm.atmprof import pressure_scale_height
    
    T = 1000  #K
    mu = 2  #mean molecular weight
    print("scale height=", pressure_scale_height(1.e5, T, mu), "cm")



.. parsed-literal::

    scale height= 415722.99317937146 cm


We need a density of condensates.

.. code:: ipython3

    rhoc_enstatite = 3.192  #g/cm3 Lodders and Fegley (1998)
    rhoc_Fe = 7.875
    
    from exojax.spec.molinfo import molmass
    
    mu = molmass("H2")
    muc_enstatite = molmass("MgSiO3")
    muc_Fe = molmass("Fe")



.. parsed-literal::

    ['H2O', 'CO2', 'O3', 'N2O', 'CO', 'CH4', 'O2', 'NO', 'SO2', 'NO2', 'NH3', 'HNO3', 'OH', 'HF', 'HCl', 'HBr', 'HI', 'ClO', 'OCS', 'H2CO', 'HOCl', 'N2', 'HCN', 'CH3Cl', 'H2O2', 'C2H2', 'C2H6', 'PH3', 'COF2', 'SF6', 'H2S', 'HCOOH', 'HO2', 'O', 'ClONO2', 'NO+', 'HOBr', 'C2H4', 'CH3OH', 'CH3Br', 'CH3CN', 'CF4', 'C4H2', 'HC3N', 'H2', 'CS', 'SO3', 'C2N2', 'COCl2', 'SO', 'CH3F', 'GeH4', 'CS2', 'CH3I', 'NF3']
    ['H2O', 'CO2', 'O3', 'N2O', 'CO', 'CH4', 'O2', 'NO', 'SO2', 'NO2', 'NH3', 'HNO3', 'OH', 'HF', 'HCl', 'HBr', 'HI', 'ClO', 'OCS', 'H2CO', 'HOCl', 'N2', 'HCN', 'CH3Cl', 'H2O2', 'C2H2', 'C2H6', 'PH3', 'COF2', 'SF6', 'H2S', 'HCOOH', 'HO2', 'O', 'ClONO2', 'NO+', 'HOBr', 'C2H4', 'CH3OH', 'CH3Br', 'CH3CN', 'CF4', 'C4H2', 'HC3N', 'H2', 'CS', 'SO3', 'C2N2', 'COCl2', 'SO', 'CH3F', 'GeH4', 'CS2', 'CH3I', 'NF3']


.. parsed-literal::

    /home/kawahara/exojax/src/exojax/spec/molinfo.py:64: UserWarning: db_HIT is set as True, but the molecular name 'MgSiO3' does not exist in the HITRAN database. So set db_HIT as False. For reference, all the available molecules in the HITRAN database are as follows:
      warnings.warn(warn_msg, UserWarning)
    /home/kawahara/exojax/src/exojax/spec/molinfo.py:64: UserWarning: db_HIT is set as True, but the molecular name 'Fe' does not exist in the HITRAN database. So set db_HIT as False. For reference, all the available molecules in the HITRAN database are as follows:
      warnings.warn(warn_msg, UserWarning)


Let’s compute the terminal velocity. We can compute the terminal
velocity of cloud particle using atm.vterm.vf. vmap is again applied to
vf.

.. code:: ipython3

    from exojax.atm.viscosity import calc_vfactor, eta_Rosner
    from exojax.atm.vterm import vf
    from jax import vmap
    
    vfactor, trange = calc_vfactor(atm="H2")
    rarr = jnp.logspace(-6, -4, 2000)  #cm
    drho = rhoc_enstatite - rho
    eta_fid = eta_Rosner(Tarr, vfactor)
    
    g = 1.e5
    vf_vmap = vmap(vf, (None, None, 0, 0, 0))
    vfs = vf_vmap(rarr, g, eta_fid, drho, rho)


Kzz/L will be used to calibrate :math:`r_w`. following Ackerman and
Marley 2001

.. code:: ipython3

    Kzz = 1.e5  #cm2/s
    sigmag = 2.0
    alphav = 1.3
    L = pressure_scale_height(g, 1500, mu)


.. code:: ipython3

    Kzz/L




.. parsed-literal::

    0.16161803517166456



.. code:: ipython3

    for i in range(0, len(Tarr)):
        plt.plot(rarr, vfs[i, :], alpha=0.2, color="gray")
    plt.xscale("log")
    plt.yscale("log")
    plt.axhline(Kzz / L, label="Kzz/H", color="C2", ls="dotted")
    plt.ylabel("stokes terminal velocity (cm/s)")
    plt.xlabel("condensate size (cm)")




.. parsed-literal::

    Text(0.5, 0, 'condensate size (cm)')




.. image:: Ackerman_and_Marley_cloud_model_files/Ackerman_and_Marley_cloud_model_29_1.png


Find the intersection.

.. code:: ipython3

    from exojax.atm.amclouds import find_rw
    
    vfind_rw = vmap(find_rw, (None, 0, None), 0)
    rw = vfind_rw(rarr, vfs, Kzz / L)


Then, :math:`r_g` can be computed from :math:`r_w` and other quantities.

.. code:: ipython3

    from exojax.atm.amclouds import get_rg
    
    rg = get_rg(rw, fsed, alphav, sigmag)


.. code:: ipython3

    plt.plot(rg * 1.e4, Parr, label="$r=r_g$", color="black")
    plt.plot(rw * 1.e4, Parr, ls="dashed", label="$r=r_w$", color="black")
    plt.ylim(1.e-7, 10000)
    plt.xlabel("$r$ (micron)")
    plt.ylabel("Pressure (bar)")
    plt.yscale("log")
    plt.savefig("rgrw.png")
    plt.legend()




.. parsed-literal::

    <matplotlib.legend.Legend at 0x7f55651dcc10>




.. image:: Ackerman_and_Marley_cloud_model_files/Ackerman_and_Marley_cloud_model_34_1.png


We found here the particle size is basically sub-micron. So, we should
use the Rayleigh scattering. But, here, we try to use the geometric
cross section instead though this is wrong.

.. code:: ipython3

    from exojax.atm.amclouds import dtau_cloudgeo
    
    dtau_enstatite = dtau_cloudgeo(Parr, muc_enstatite, rhoc_enstatite, mu,
                                   VMRc_enstatite, rg, sigmag, g)
    dtau_Fe = dtau_cloudgeo(Parr, muc_Fe, rhoc_Fe, mu, VMRc_Fe, rg, sigmag, g)


.. code:: ipython3

    plt.plot(dtau_enstatite, Parr, color="C1")
    plt.plot(dtau_Fe, Parr, color="C2", ls="dashed")
    plt.yscale("log")
    plt.ylim(1.e-7, 10000)
    plt.xlabel("$d\\tau$")
    plt.ylabel("Pressure (bar)")
    #plt.xscale("log")
    plt.gca().invert_yaxis()



.. image:: Ackerman_and_Marley_cloud_model_files/Ackerman_and_Marley_cloud_model_37_0.png


Let’s compare with CIA

.. code:: ipython3

    #CIA
    from exojax.utils.grids import wavenumber_grid
    nus, wav, res = wavenumber_grid(9500, 30000, 1000, unit="AA", xsmode="premodit")
    
    from exojax.spec import contdb
    cdbH2H2 = contdb.CdbCIA('.database/H2-H2_2011.cia', nus)


.. parsed-literal::

    xsmode =  premodit
    xsmode assumes ESLOG in wavenumber space: mode=premodit
    ======================================================================
    We changed the policy of the order of wavenumber/wavelength grids
    wavenumber grid should be in ascending order and now 
    users can specify the order of the wavelength grid by themselves.
    Your wavelength grid is in ***  descending  *** order
    This might causes the bug if you update ExoJAX. 
    Note that the older ExoJAX assumes ascending order as wavelength grid.
    ======================================================================
    H2-H2


.. parsed-literal::

    /home/kawahara/exojax/src/exojax/utils/grids.py:145: UserWarning: Resolution may be too small. R=868.7669794117727
      warnings.warn('Resolution may be too small. R=' + str(resolution),


.. code:: ipython3

    from exojax.spec.layeropacity import layer_optical_depth_CIA
    
    mmw = 2.33  # mean molecular weight
    mmrH2 = 0.74
    molmassH2 = molmass("H2")
    vmrH2 = mmrH2 * mmw / molmassH2  # VMR
    dtaucH2H2 = layer_optical_depth_CIA(
        nus,
        Tarr,
        Parr,
        dParr,
        vmrH2,
        vmrH2,
        mmw,
        g,
        cdbH2H2.nucia,
        cdbH2H2.tcia,
        cdbH2H2.logac,
    )


.. parsed-literal::

    /home/kawahara/exojax/src/exojax/spec/dtau_mmwl.py:14: FutureWarning: dtau_mmwl might be removed in future.
      warnings.warn("dtau_mmwl might be removed in future.", FutureWarning)


.. code:: ipython3

    dtau = dtaucH2H2 + dtau_enstatite[:, None] + dtau_Fe[:, None]


.. code:: ipython3

    from exojax.plot.atmplot import plotcf
    
    plotcf(nus, dtau, Tarr, Parr, dParr, unit="nm")
    plt.show()



.. image:: Ackerman_and_Marley_cloud_model_files/Ackerman_and_Marley_cloud_model_42_0.png


.. code:: ipython3

    from exojax.plot.atmplot import plotcf
    
    plotcf(nus, dtaucH2H2, Tarr, Parr, dParr, unit="AA")
    plt.show()



.. image:: Ackerman_and_Marley_cloud_model_files/Ackerman_and_Marley_cloud_model_43_0.png


.. code:: ipython3

    from exojax.plot.atmplot import plotcf
    
    plotcf(nus,
           dtau_enstatite[:, None] + np.zeros_like(dtaucH2H2),
           Tarr,
           Parr,
           dParr,
           unit="AA")
    plt.show()



.. image:: Ackerman_and_Marley_cloud_model_files/Ackerman_and_Marley_cloud_model_44_0.png


.. code:: ipython3

    from exojax.spec import planck
    from exojax.spec.rtransfer import rtrun_emis_pureabs_fbased2st as rtrun
    
    sourcef = planck.piBarr(Tarr, nus)
    F0 = rtrun(dtau, sourcef)
    F0CIA = rtrun(dtaucH2H2, sourcef)
    F0cl = rtrun(dtau_enstatite[:, None] + np.zeros_like(dtaucH2H2), sourcef)


.. code:: ipython3

    plt.plot(wav[::-1], F0, label="Clouds+CIA")
    plt.plot(wav[::-1], F0CIA, label="CIA only", ls="dashed")
    plt.plot(wav[::-1], F0cl, label="Clouds only", ls="dotted")
    plt.xlabel("wavelenght AA")
    plt.legend()
    plt.show()



.. image:: Ackerman_and_Marley_cloud_model_files/Ackerman_and_Marley_cloud_model_46_0.png


