Compatibility to External Packages
=====================================


PetitRadtrans
-------------------

One can read the high-R opacity data provided by petitRadtrans as

.. code:: ipython
       
       >>> petitdir="~/petitRADTRANS/petitRADTRANS/input_data/opacities/lines/line_by_line/CO_all_iso/"
       >>> with open(petitdir+"wlen.dat", 'rb') as w:
       >>>     contentw = np.fromfile(w, dtype=np.float64)
       >>> 
       >>> with open(petitdir+"sigma_05_900.K_0.100000bar.dat", 'rb') as f:
       >>>     contentf = np.fromfile(f, dtype=np.float64)

Also, one can export the opacity computed by ExoJAX to petitRadtrans high-R form, such as

.. code:: ipython
       
       >>> outdir="~/petitRADTRANS/petitRADTRANS/input_data/opacities/lines/line_by_line/CO_exojax/"
       >>> np.array(1.0/nus[::-1],dtype=np.float64).tofile(outdir+"wlen.dat")
       >>> Mmol=28.010446441149536 # molecular weight
       >>> 
       >>> Tarr=np.logspace(2,3.5,10)
       >>> Parr=np.logspace(-10,2,13)
       >>> nu0=mdbCO.nu_lines
       >>> 
       >>> f=open(outdir+"PTpaths.ls","w")
       >>> for Tfix in Tarr:
       >>>     qt=mdbCO.Qr_line_HAPI(Tfix)
       >>>     sigmaD=doppler_sigma(mdbCO.nu_lines,Tfix,Mmol)
       >>>     for Pfix in Parr:
       >>>         Ppart=Pfix #partial pressure of CO. here we assume a 100% CO atmosphere. 
       >>> 
       >>>         Sij=SijT(Tfix,mdbCO.logsij0,mdbCO.nu_lines,mdbCO.elower,qt)
       >>>         gammaL = gamma_hitran(Pfix,Tfix, Ppart, mdbCO.n_air, \
       >>>                       mdbCO.gamma_air, mdbCO.gamma_self) + gamma_natural(mdbCO.A) 
       >>>         # thermal doppler sigma
       >>>         xsv=auto_xsection(nus,nu0,sigmaD,gammaL,Sij,memory_size=30)
       >>>         Pval="{:.6f}".format(Pfix)
       >>>         P=str(Pval)+"bar"
       >>>         out="sigma_05_"+str(Tfix)+".K_"+P+".dat"
       >>>         f.write(str(Pval)+" "+str(Tfix)+" "+out+"\n")    
       >>>         op=np.array(xsv[::-1],dtype=np.float64)/(Mmol*1.66053892e-24)
       >>>         op.tofile(outdir+out)
       >>> f.close()

