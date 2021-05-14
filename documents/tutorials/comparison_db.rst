
Comparing HITEMP and ExoMol
---------------------------

*Update: May 14/2021, Hajime Kawahara*

.. code:: ipython3

    from exojax.spec import xsection
    from exojax.spec.hitran import SijT, doppler_sigma, gamma_hitran, gamma_natural
    from exojax.spec.exomol import gamma_exomol
    from exojax.spec import moldb
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.style.use('bmh')


We call moldb instance with the path of par file. If the par file does
not exist, moldb will try to download it from HITEMP website.

.. code:: ipython3

    # Setting wavenumber bins and loading HITEMP database
    wav=np.linspace(22930.0,23000.0,4000,dtype=np.float64) #AA
    nus=1.e8/wav[::-1] #cm-1
    mdbCO_HITEMP=moldb.MdbHit('/home/kawahara/exojax/data/CO/05_HITEMP2019.par.bz2',nus)


.. parsed-literal::

    https://www.cfa.harvard.edu/HITRAN/HITRAN2012/HITRAN2012/By-Molecule/Uncompressed-files/05_HITEMP2019.par.bz2
    HITRAN download failed
    https://hitran.org/hitemp/data/bzip2format/05_HITEMP2019.par.bz2
    bunziping

Also, we use ExoMol/Li2015.

    
.. code:: ipython3

    emf='/home/kawahara/exojax/data/CO/12C-16O/Li2015'
    mdbCO_Li2015=moldb.MdbExomol(emf,nus)


.. parsed-literal::

    Downloading http://www.exomol.com/db/CO/12C-16O/Li2015/12C-16O__Li2015.def
    Downloading http://www.exomol.com/db/CO/12C-16O/Li2015/12C-16O__Li2015.pf
    Downloading http://www.exomol.com/db/CO/12C-16O/Li2015/12C-16O__Li2015.states.bz2
    Mol mass= 28.0101
    gamma width= 0.07
    T exponent= 0.5
    Note: Couldn't find the feather format. We convert data to the feather format. After the second time, it will become much faster.
    Reading transition file
    Downloading http://www.exomol.com/db/CO/12C-16O/Li2015/12C-16O__Li2015.trans.bz2
    Note: Couldn't find the feather format. We convert data to the feather format. After the second time, it will become much faster.


Define molecular weight of CO (:math:`\sim 12+16=28`), temperature (K),
and pressure (bar). Also, we here assume the 1 % CO partial pressure,
i.e. the partial pressure = pressure.

.. code:: ipython3

    from exojax.spec import molinfo
    Mmol=molinfo.molmass("CO") # molecular weight
    Tfix=1300.0 # we assume T=1300K
    Pfix=0.99 # we compute P=1 bar=0.99+0.1
    Ppart=0.01 #partial pressure of CO. here we assume a 1% CO atmosphere (very few). 

partition function ratio :math:`q(T)` is defined by

:math:`q(T) = Q(T)/Q(T_{ref})`; :math:`T_{ref}`\ =296 K

Here, we use the partition function from ExoMol for both databases.

.. code:: ipython3

    mdbCO_HITEMP.ExomolQT(emf) #use Q(T) from Exomol/Li2015
    qt_HITEMP=mdbCO_HITEMP.qr_interp(Tfix)
    qt_Li2015=mdbCO_Li2015.qr_interp(Tfix)

Let us compute the line strength S(T) at temperature of Tfix.

:math:`S (T;s_0,\nu_0,E_l,q(T)) = S_0 \frac{Q(T_{ref})}{Q(T)} \frac{e^{- h c E_l /k_B T}}{e^{- h c E_l /k_B T_{ref}}} \frac{1- e^{- h c \nu /k_B T}}{1-e^{- h c \nu /k_B T_{ref}}}= q_r(T)^{-1} e^{ s_0 - c_2 E_l (T^{-1} - T_{ref}^{-1})} \frac{1- e^{- c_2 \nu_0/ T}}{1-e^{- c_2 \nu_0/T_{ref}}}`

:math:`s_0=\log_{e} S_0` : logsij0

:math:`\nu_0`: nu_lines

:math:`E_l` : elower

Computing line strength...

.. code:: ipython3

    Sij_HITEMP=SijT(Tfix,mdbCO_HITEMP.logsij0,mdbCO_HITEMP.nu_lines,\
             mdbCO_HITEMP.elower,qt_HITEMP)
    Sij_Li2015=SijT(Tfix,mdbCO_Li2015.logsij0,mdbCO_Li2015.nu_lines,\
                    mdbCO_Li2015.elower,qt_Li2015)

Then, compute the Lorentz gamma factor (pressure+natural broadening). This causes a main difference between tow databases as we will see later.

:math:`\gamma_L = \gamma^p_L + \gamma^n_L`

where the pressure broadning for HITEMP

:math:`\gamma^p_L = (T/296K)^{-n_{air}} [ \alpha_{air} ( P - P_{part})/P_{atm} + \alpha_{self} P_{part}/P_{atm}]`

:math:`P_{atm}`: 1 atm in the unit of bar (i.e. = 1.01325)

or

the pressure broadning for ExoMol

$:raw-latex:`\gamma`^p_L = :raw-latex:`\alpha`\ *{ref} ( T/T*\ {ref}
)^{-n_{texp}} ( P/P_{ref}), $

and the natural broadening

:math:`\gamma^n_L = \frac{A}{4 \pi c}`

.. code:: ipython3

    gammaL_HITEMP = gamma_hitran(Pfix,Tfix, Ppart, mdbCO_HITEMP.n_air, \
                          mdbCO_HITEMP.gamma_air, mdbCO_HITEMP.gamma_self) \
    + gamma_natural(mdbCO_HITEMP.A) 
    
    gammaL_Li2015 = gamma_exomol(Pfix,Tfix,mdbCO_Li2015.n_Texp,mdbCO_Li2015.alpha_ref)\
    + gamma_natural(mdbCO_Li2015.A) 

Thermal broadening

:math:`\sigma_D^{t} = \sqrt{\frac{k_B T}{M m_u}} \frac{\nu_0}{c}`

.. code:: ipython3

    # thermal doppler sigma
    sigmaD_HITEMP=doppler_sigma(mdbCO_HITEMP.nu_lines,Tfix,Mmol)
    sigmaD_Li2015=doppler_sigma(mdbCO_Li2015.nu_lines,Tfix,Mmol)

Then, the line center…

In HITRAN database, a slight pressure shift can be included using
:math:`\delta_{air}`: :math:`\nu_0(P) = \nu_0 + \delta_{air} P`. But
this shift is quite a bit.

.. code:: ipython3

    #line center
    nu0_HITEMP=mdbCO_HITEMP.nu_lines
    nu0_Li2015=mdbCO_Li2015.nu_lines


Let us compute the cross sections!
    
.. code:: ipython3

    xsv_HITEMP=xsection(nus,nu0_HITEMP,sigmaD_HITEMP,gammaL_HITEMP\
                         ,Sij_HITEMP,memory_size=30) #use 30MB GPU MEMORY for numax
    xsv_Li2015=xsection(nus,nu0_Li2015,sigmaD_Li2015,gammaL_Li2015\
                         ,Sij_Li2015,memory_size=30) #use 30MB GPU MEMORY for numax


.. parsed-literal::

    100%|██████████| 12/12 [00:00<00:00, 15.44it/s]
    100%|██████████| 1/1 [00:00<00:00,  1.92it/s]


Plot them.

.. code:: ipython3

    fig=plt.figure(figsize=(10,3))
    ax=fig.add_subplot(111)
    plt.plot(wav[::-1],xsv_HITEMP,lw=2,label="HITEMP2019")
    plt.plot(wav[::-1],xsv_Li2015,lw=2,ls="dashed",label="Exomol $(\\alpha_\\mathrm{ref}=0.07,\
             n_\\mathrm{texp}=0.5$)")
    plt.xlim(22970,22988)
    plt.xlabel("wavelength ($\AA$)",fontsize=14)
    plt.ylabel("cross section ($cm^{2}$)",fontsize=14)
    plt.legend(loc="upper left",fontsize=14)
    plt.tick_params(labelsize=12)
    plt.savefig("co_comparison.pdf", bbox_inches="tight", pad_inches=0.0)
    plt.savefig("co_comparison.png", bbox_inches="tight", pad_inches=0.0)
    plt.title("T=1300K,P=1bar")
    plt.show()



.. image:: comparison_db/output_20_0.png


Oh, so different! This is beacuse of the different gamma factors.
	   
.. code:: ipython3

    np.min(gammaL_HITEMP),np.max(gammaL_HITEMP)




.. parsed-literal::

    (DeviceArray(0.01456154, dtype=float32),
     DeviceArray(0.02531094, dtype=float32))



.. code:: ipython3

    np.min(gammaL_Li2015),np.max(gammaL_Li2015)




.. parsed-literal::

    (DeviceArray(0.03263554, dtype=float32),
     DeviceArray(0.03263554, dtype=float32))



