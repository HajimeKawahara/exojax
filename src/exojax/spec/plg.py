"""Pseudo line generator

"""
import numpy as np
from exojax.spec.lsd import npgetix
import jax.numpy as jnp
import warnings
from exojax.utils.constants import hcperk
from exojax.spec.hitran import SijT


def plg_elower_addcon(indexa,Na,cnu,indexnu,nu_grid,mdb,Tgue, errTgue=500., elower_grid=None,Nelower=10,Ncrit=0,reshape=False, weedout=False, Tmargin=0., coefTgue=1., verbose=True, threshold_persist_freezing=10000.):
    """Pseudo Line Grid for elower w/ an additional condition
    
    Args:
       indexa: the indexing of the additional condition
       Na: the number of the additional condition grid
       cnu: contribution of wavenumber for LSD
       indexnu: nu index
       wavenumber_grid: nu grid
       mdb: molecular database (instance made by the MdbExomol/MdbHit class in moldb.py)
       Tgue: rough guess on the typical atmospheric temperature
       errTgue: expected one-sided error of Tgue
       elower_grid: elower_grid (optional)
       Nelower: # of division of elower between min to max values (when elower_grid is not given)
       Ncrit: frozen line number per bin
       reshape: reshaping output arrays
       weedout: Is it ok to remove weakest lines or not?
       Tmargin: margin even above Tuplim to prevent freeze of lines that  become strong enough as temperature rises slightly. (more needed for cooler objects)
       coefTgue: coefficient for Tgue to optimize elower_grid
       verbose: vervose printing or not
       threshold_persist_freezing: How weak lines compared to the deepest one will you persist freezing (not retaken)?

    Returns:
       qlogsij0: pseudo logsij0
       qcnu: pseudo cnu
       num_unique: number of the lines in each bin of grid
       elower_grid: elower of pl
       frozen_mask: mask for frozen lines into pseudo lines 
       nonzeropl_mask: mask for pseudo-lines w/ non-zero

    """
    elower = mdb.elower
    Nnugrid=len(nu_grid)
    Tref = 296.0
    Tuplim = Tgue + errTgue
    Tuplimc = Tuplim * coefTgue
    float32_almost_max = 2.**(2**7-1)
    preov = np.max(- hcperk*(elower/Tuplimc - elower/Tref)) - np.log(float32_almost_max) #ad hoc parameter to prevent overflow
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        try:
            expme = np.exp(- hcperk*(elower/Tuplimc - elower/Tref) - preov)
        except RuntimeWarning as e:
            raise Exception(str(e)+' :\t expme does not fall within the representable range of single-precision floating-point number (float32) ...')
        if elower_grid is None:
            margin = 1.0
            min_expme = np.exp(- hcperk*((np.min(elower)-margin)/Tuplimc - (np.min(elower)-margin)/Tref) - preov)
            max_expme = np.exp(- hcperk*((np.max(elower)+margin)/Tuplimc - (np.max(elower)+margin)/Tref) - preov)
            expme_grid = np.linspace(min_expme, max_expme, Nelower)
            elower_grid = (np.log(expme_grid) + preov) / (-hcperk) / (1/Tuplimc - 1/Tref)
        else:
            expme_grid = np.exp(elower_grid*(1/Tuplimc - 1/Tref)*(-hcperk) - preov)
            Nelower=len(expme_grid)

    qlogsij0, qcnu, num_unique, frozen_mask = get_qlogsij0_addcon(indexa, Na, cnu, indexnu, Nnugrid, mdb, expme, expme_grid, Ncrit=Ncrit, Tuplim=Tuplim, Tmargin=Tmargin, threshold_persist_freezing=threshold_persist_freezing)
    
    nonzeropl_mask = qlogsij0 > -np.inf
    Nline = len(elower)
    Nunf = np.sum(~frozen_mask)
    Npl = len(qlogsij0[nonzeropl_mask])
    if verbose:
        print("# of original lines:", Nline)
        print("# of unfrozen lines:", Nunf)
        print("# of frozen lines:", np.sum(frozen_mask))
        print("# of pseudo lines:", Npl)
        print("# of total lines:", (Npl+Nunf))
        print("# compression:", (Npl+Nunf)/Nline)

    if reshape==True:
        qlogsij0=qlogsij0.reshape(Na,Nnugrid,Nelower)
        qcnu=qcnu.reshape(Na,Nnugrid,Nelower)
        num_unique=num_unique.reshape(Na,Nnugrid,Nelower)
        
    return qlogsij0, qcnu, num_unique, elower_grid, frozen_mask, nonzeropl_mask


def get_qlogsij0_addcon(indexa, Na, cnu, indexnu, Nnugrid, mdb, expme, expme_grid, Ncrit=0, Tuplim=1000., Tmargin=0., threshold_persist_freezing=10000.):
    """gather (freeze) lines w/ additional indexing

    Args:
       indexa: the indexing of the additional condition
       Na: the number of the additional condition grid
       cnu: contribution of wavenumber for LSD
       indexnu: index of wavenumber
       Nnugrid: number of nu grid
       mdb: molecular database (instance made by the MdbExomol/MdbHit class in moldb.py)
       expme: array of exponentials of functions of elower: exp(- hcperk*(elower/Tuplimc - elower/Tref) - preov)
       expme_grid: grid of expme
       Ncrit: frozen line number per bin
       Tuplim: rough guess on the typical atmospheric temperature
       Tmargin: margin even above Tuplim to prevent freeze of lines that  become strong enough as temperature rises slightly. (more needed for cooler objects)
       threshold_persist_freezing: How weak lines compared to the deepest one will you persist freezing (not retaken)?

    Returns:
        qlogsij0: log line strength
        qcnu: cnu for pseudo line (contribution of wavenumber for LSD)
        num_unique: number of the lines in each bin of grid
        frozen_mask: mask for frozen lines into pseudo lines

    """
    m = len(expme_grid)
    n = m * Nnugrid
    Ng = n * Na
    
    cont, index = npgetix(expme, expme_grid) #elower contribution and elower index of lines
    eindex = index + m * indexnu + n * indexa #extended index elower,nu,a
    
    #frozen criterion
    num_unique = np.bincount(eindex, minlength=Ng) # number of the lines in a bin
    
    lmask = (num_unique >= Ncrit)
    erange = range(0, Ng)
    frozen_eindex = np.array(erange)[lmask]
    frozen_mask = np.isin(eindex, frozen_eindex)
    
    #Retake relatively strong lines.
    SijTgue_frozen = SijT(Tuplim+Tmargin, \
                    mdb.logsij0[frozen_mask], mdb.nu_lines[frozen_mask], mdb.elower[frozen_mask], \
                    qT=mdb.qr_interp(Tuplim+Tmargin))
    persist_freezing = SijTgue_frozen < np.max(SijTgue_frozen) / threshold_persist_freezing
    index_persist_freezing = np.where(frozen_mask)[0][persist_freezing]
    frozen_mask = np.isin(np.arange(len(frozen_mask)), index_persist_freezing)
        
    #qlogsij0
    Sij0 = np.exp(mdb.logsij0)
    Sij0 = np.where(Sij0 >= 1e-45, Sij0, np.zeros_like(Sij0)) #For the case where "jax_enable_x64" is True.
    qSij0 = np.bincount(eindex, weights=Sij0*(1.0-cont)*frozen_mask, minlength=Ng)
    qSij0 = qSij0 + np.bincount(eindex+1, weights=Sij0*cont*frozen_mask, minlength=Ng)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        qlogsij0 = np.log(qSij0)

    #qcnu
    qcnu_den = np.bincount(eindex, weights=Sij0*frozen_mask, minlength=Ng)
    qcnu_den = qcnu_den + np.bincount(eindex+1, weights=Sij0*frozen_mask, minlength=Ng)
    qcnu_den[qcnu_den==0.0] = 1.0
    
    qcnu_num = np.bincount(eindex, weights=Sij0*cnu*frozen_mask, minlength=Ng)
    qcnu_num = qcnu_num + np.bincount(eindex+1, weights=Sij0*cnu*frozen_mask, minlength=Ng)
    qcnu = qcnu_num/qcnu_den

    return qlogsij0, qcnu, num_unique, frozen_mask


def plg_elower(cnu,indexnu,Nnugrid,logsij0,elower,elower_grid=None,Nelower=10,Ncrit=0,reshape=True):
    """Pseudo Line Grid for elower
    
    Args:
       cnu: contribution of wavenumber for LSD
       indexnu: index of wavenumber
       Nnugrid: number of Ngrid
       logsij0: log line strength
       elower: elower
       elower_grid: elower_grid (optional)
       Nelower: # of division of elower between min to max values when elower_grid is not given
       Ncrit: frozen line number per bin

    Returns:
       qlogsij0
       qcnu
       num_unique
       elower_grid

    """
    
    kT0=10000.0
    expme=np.exp(-elower/kT0)
    if elower_grid is None:
        margin=1.0
        min_expme=np.min(expme)*np.exp(-margin/kT0)
        max_expme=np.max(expme)*np.exp(margin/kT0)
        expme_grid=np.linspace(min_expme,max_expme,Nelower)
        elower_grid=-np.log(expme_grid)*kT0
    else:
        expme_grid=np.exp(-elower_grid/kT0)
        Nelower=len(expme_grid)
        
    qlogsij0,qcnu,num_unique,frozen_mask=get_qlogsij0(cnu,indexnu,Nnugrid,logsij0,expme,expme_grid,Nelower=10,Ncrit=Ncrit)

    if reshape==True:
        qlogsij0=qlogsij0.reshape(Nnugrid,Nelower)
        qcnu=qcnu.reshape(Nnugrid,Nelower)
        num_unique=num_unique.reshape(Nnugrid,Nelower)
    
    print("# of unfrozen lines:",np.sum(~frozen_mask))
    print("# of pseudo lines:",len(qlogsij0[qlogsij0>0.0]))
    
    return qlogsij0,qcnu,num_unique,elower_grid


def get_qlogsij0(cnu,indexnu,Nnugrid,logsij0,expme,expme_grid,Nelower=10,Ncrit=0):
    """gather (freeze) lines

    Args:
       cnu: contribution of wavenumber for LSD
       indexnu: index of wavenumber
       logsij0: log line strength
       expme: exp(-elower/kT0)
       expme_grid: exp(-elower/kT0)_grid
       Nelower: # of division of elower between min to max values

    """
    m=len(expme_grid)
    cont,index=npgetix(expme,expme_grid) #elower contribution and elower index of lines
    eindex=index+m*indexnu #extended index
    
    #frozen criterion
    Ng=m*Nnugrid
    num_unique=np.bincount(eindex,minlength=Ng) # number of the lines in a bin
    
    lmask=(num_unique>=Ncrit)
    erange=range(0,Ng)
    frozen_eindex=np.array(erange)[lmask]
    frozen_mask=np.isin(eindex,frozen_eindex)
        
    Sij=np.exp(logsij0)
    #qlogsij0
    qlogsij0=np.bincount(eindex,weights=Sij*(1.0-cont)*frozen_mask,minlength=Ng)
    qlogsij0=qlogsij0+np.bincount(eindex+1,weights=Sij*cont*frozen_mask,minlength=Ng)    
    qlogsij0=np.log(qlogsij0)

    #qcnu
    qcnu_den=np.bincount(eindex,weights=Sij*frozen_mask,minlength=Ng)
    qcnu_den=qcnu_den+np.bincount(eindex+1,weights=Sij*frozen_mask,minlength=Ng)
    qcnu_den[qcnu_den==0.0]=1.0
    
    qcnu_num=np.bincount(eindex,weights=Sij*cnu*frozen_mask,minlength=Ng)
    qcnu_num=qcnu_num+np.bincount(eindex+1,weights=Sij*cnu*frozen_mask,minlength=Ng)
    qcnu=qcnu_num/qcnu_den

    return qlogsij0,qcnu,num_unique,frozen_mask


def optimize_coefTgue(Tgue, nus, mdb, molmass, Nelower, errTgue=500., Mgue=41., Rgue=1., MMRgue=0.001):
    """Optimize coefTgue for adjusting elower_grid
    
    Args:
        Tgue: rough guess on the typical atmospheric temperature
        nus: wavenumber grid for model
        mdb: molecular database (instance made by the MdbExomol/MdbHit class in moldb.py)
        molmass: mass of the molecule (amu)
        Nelower: # of elower grid
        errTgue: expected one-sided error of Tgue
        Mgue: rough guess on the mass
        Rgue: rough guess on the radius
        MMRgue: rough guess on the MMR of the molecule

    Returns:
        optimized coefTgue
        
    """
    import scipy.optimize as opt
    coefTgue_opt = opt.least_squares(\
    fun=diff_frozen_vs_pseudo, x0=[1.0,], bounds=(0.5, 2.0), args=[Tgue, errTgue, Mgue, Rgue, MMRgue, nus, mdb, molmass, Nelower])
    return coefTgue_opt.x[0]

    
def diff_frozen_vs_pseudo(coefTgue_init, Tgue, errTgue, Mgue, Rgue, MMRgue, nus, mdb_orig, molmass, Nelower, verbose=False):
    """Evaluate difference between pseudo lines and the original lines
    
    Args:
        coefTgue_init: initial value of coefTgue (list/array with length=1)
        Tgue: rough guess on the typical atmospheric temperature
        errTgue: expected one-sided error of Tgue
        Mgue: rough guess on the mass
        Rgue: rough guess on the radius
        MMRgue: rough guess on the MMR of the molecule
        nus: wavenumber grid for model
        mdb_orig: original molecular database before compression (instance made by the MdbExomol/MdbHit class in moldb.py)
        molmass: mass of the molecule (amu)
        Nelower: # of elower grid
        verbose: vervose printing or not
    
    Returns:
        abs(np.mean(mu_pseudo - mu_orig))

    """
    coefTgue = coefTgue_init[0]
    Ncrit = 0 #10
    from exojax.spec import initspec
    import matplotlib.pyplot as plt
    import copy
    
    mdb = copy.deepcopy(mdb_orig)
    cnu, indexnu, R, pmarray = initspec.init_modit(mdb.nu_lines, nus)
    cnu_orig = copy.deepcopy(cnu)
    indexnu_orig = copy.deepcopy(indexnu)

    gammaL_set=mdb.alpha_ref+mdb.n_Texp*(1j) #complex value
    gammaL_set_unique=np.unique(gammaL_set)
    Ngamma=np.shape(gammaL_set_unique)[0]
    index_gamma=np.zeros_like(mdb.alpha_ref,dtype=int)
    alpha_ref_grid=gammaL_set_unique.real
    n_Texp_grid=gammaL_set_unique.imag
    for j,a in enumerate(gammaL_set_unique):
        index_gamma=np.where(gammaL_set==a,j,index_gamma)

    qlogsij0,qcnu,num_unique,elower_grid,frozen_mask,nonzeropl_mask = plg_elower_addcon(index_gamma,Ngamma,cnu,indexnu,nus, mdb,Tgue, errTgue=errTgue,Ncrit=Ncrit,Nelower=Nelower,reshape=False, coefTgue=coefTgue,verbose=verbose)
    Nnugrid=len(nus)
    
    mdb, cnu, indexnu = gather_lines(mdb,Ngamma,Nnugrid,Nelower,nus,cnu,indexnu,qlogsij0,qcnu,elower_grid,alpha_ref_grid,n_Texp_grid,frozen_mask,nonzeropl_mask)

    from exojax.spec import rtransfer as rt
    from exojax.spec import dit, modit
    from exojax.spec.rtransfer import rtrun, dtauM
    from exojax.spec import planck, response

    NP=100
    Parr, dParr, k=rt.pressure_layer(NP=NP)
    Pref=1.0 #bar
    ONEARR=np.ones_like(Parr)
    
    from exojax.spec.modit import set_ditgrid_matrix_exomol
    fT = lambda T0,alpha: T0[:,None]*(Parr[None,:]/Pref)**alpha[:,None]
    T0_test=np.array([Tgue+errTgue-300.,Tgue+errTgue+300.,Tgue+errTgue-300.,Tgue+errTgue+300.])
    alpha_test=np.array([0.2,0.2,0.05,0.05])
    res=0.2
    dgm_ngammaL = set_ditgrid_matrix_exomol(mdb, fT, Parr, R, molmass, res, T0_test, alpha_test)
    dgm_ngammaL_orig = set_ditgrid_matrix_exomol(mdb_orig, fT, Parr, R, molmass, res, T0_test, alpha_test)

    def frun_pseu(Tarr,MMR_,Mp,Rp,u1,u2,RV,vsini):
        g=2478.57730044555*Mp/Rp**2
        SijM_,ngammaLM_,nsigmaDl_=modit.exomol(mdb,Tarr,Parr,R,molmass)
        xsm_=modit.xsmatrix(cnu,indexnu,R,pmarray,nsigmaDl_,ngammaLM_,SijM_,nus,dgm_ngammaL)
        dtaum=dtauM(dParr,jnp.abs(xsm_),MMR_*ONEARR,molmass,g)
        #Ignore continuous absorption by CIA or Hminus here.
        dtau = dtaum
        sourcef = planck.piBarr(Tarr,nus)
        F0=rtrun(dtau,sourcef)
        Frot=response.rigidrot(nus,F0,vsini,u1,u2)
        mu=Frot
        return mu
    def frun_orig(Tarr, MMR_, Mp, Rp, u1, u2, RV, vsini):
        g = 2478.57730044555*Mp/Rp**2
        SijM_, ngammaLM_, nsigmaDl_ = modit.exomol(mdb_orig, Tarr, Parr, R, molmass)
        xsm_ = modit.xsmatrix(cnu_orig, indexnu_orig, R, pmarray, nsigmaDl_, ngammaLM_, SijM_, nus, dgm_ngammaL_orig)
        dtaum = dtauM(dParr, jnp.abs(xsm_), MMR_*ONEARR, molmass, g)
        #Ignore continuous absorption by CIA or Hminus here.
        dtau = dtaum
        sourcef  =  planck.piBarr(Tarr, nus)
        F0 = rtrun(dtau, sourcef)
        Frot = response.rigidrot(nus, F0, vsini, u1, u2)
        mu = Frot
        return mu
        
    mu_pseu = frun_pseu(Tarr=(Tgue+errTgue)*(Parr/Pref)**0.1, MMR_=MMRgue,Mp=Mgue,Rp=Rgue,u1=0.0,u2=0.0,RV=0.0,vsini=0.1)
    mu_orig = frun_orig(Tarr=(Tgue+errTgue)*(Parr/Pref)**0.1, MMR_=MMRgue,Mp=Mgue,Rp=Rgue,u1=0.0,u2=0.0,RV=0.0,vsini=0.1)
    return abs(np.mean(mu_pseu - mu_orig))


def make_gamma_grid_exomol(mdb):
    """Make grids of parameters for Lorentzian width and an associated index.

    Args:
        mdb: molecular database (instance made by the MdbExomol/MdbHit class in moldb.py)

    Returns:
        alpha_ref_grid: grid of alpha_ref (gamma0)
        n_Texp_grid: grid of temperature exponent of gamma
        index_gamma: grid-index of the parameters above for each original line

    """
    gammaL_set = mdb.alpha_ref+mdb.n_Texp*(1j) #complex value
    gammaL_set_unique = np.unique(gammaL_set)
    alpha_ref_grid = gammaL_set_unique.real
    n_Texp_grid = gammaL_set_unique.imag
    index_gamma = np.zeros_like(mdb.alpha_ref,dtype = int)
    for j,a in enumerate(gammaL_set_unique):
        index_gamma = np.where(gammaL_set == a, j, index_gamma)
    return alpha_ref_grid, n_Texp_grid, index_gamma
    
        
def gather_lines(mdb, Na, Nnugrid, Nelower, nu_grid, cnu, indexnu, qlogsij0, qcnu, elower_grid, alpha_ref_grid, n_Texp_grid, frozen_mask, nonzeropl_mask):
    """gather pseudo lines and unfrozen lines into lines for exomol

    Args:

       mdb: molecular database (instance made by the MdbExomol/MdbHit class in moldb.py)
       Na: the number of the additional condition grid (gammaL set for Exomol)
       Nnugrid: # of nu_grid
       Nelower: # of elower grid
       nu_grid: nu grid
       cnu: contribution of wavenumber for LSD
       indexnu: index nu
       qlogsij0: log line strength
       qcnu: cnu for pseudo line (contribution of wavenumber for LSD)
       elower_grid: elower_grid
       alpha_ref_grid: grid of alpha_ref
       n_Texp_grid: grid of n_Texp
       frozen_mask: mask for frozen lines into pseudo lines 
       nonzeropl_mask: mask for pseudo-lines w/ non-zero

    Returns:
       mdb for the gathered lines
       cnu for the gathered lines
       indexnu for the gathered lines


    """
    
    
    #gathering
    ## q-part should be ((Na,Nnugrid,Nelower).flatten)[nonzeropl_mask]
    import jax.numpy as jnp

    ## MODIT
    arrone = np.ones((Na,Nelower))
    qnu_grid = (arrone[:, np.newaxis, :] * nu_grid[np.newaxis, :, np.newaxis]).flatten()
    indexnu_grid = np.array(range(0, len(nu_grid)), dtype=int)
    qindexnu = (arrone[:, np.newaxis, :] * indexnu_grid[np.newaxis, :, np.newaxis]).flatten()
    cnu = np.hstack([qcnu[nonzeropl_mask], cnu[~frozen_mask]])
    indexnu = np.array(np.hstack([qindexnu[nonzeropl_mask], indexnu[~frozen_mask]]), dtype=int)
    
    #mdb
    mdb.logsij0 = np.hstack([qlogsij0[nonzeropl_mask], mdb.logsij0[~frozen_mask]])
    mdb.nu_lines = np.hstack([qnu_grid[nonzeropl_mask], mdb.nu_lines[~frozen_mask]])
    mdb.dev_nu_lines = jnp.array(mdb.nu_lines)

    onearr = np.ones((Na,Nnugrid))
    qelower = (onearr[:, :, np.newaxis] * elower_grid[np.newaxis, np.newaxis,:]).flatten()
    mdb.elower = np.hstack([qelower[nonzeropl_mask], mdb.elower[~frozen_mask]])
    
    #gamma     #Na,Nnugrid,Nelower
    onearr_ = np.ones((Nnugrid, Nelower))
    alpha_ref_grid = alpha_ref_grid[:, np.newaxis, np.newaxis] * onearr_
    alpha_ref_grid = alpha_ref_grid.flatten()
    n_Texp_grid = n_Texp_grid[:, np.newaxis, np.newaxis] * onearr_
    n_Texp_grid = n_Texp_grid.flatten()
    mdb.alpha_ref = np.hstack([alpha_ref_grid[nonzeropl_mask], mdb.alpha_ref[~frozen_mask]])
    mdb.n_Texp = np.hstack([n_Texp_grid[nonzeropl_mask], mdb.n_Texp[~frozen_mask]])
    mdb.A = jnp.zeros_like(mdb.logsij0) #no natural width

    lenarr = [len(mdb.logsij0), len(mdb.elower), len(cnu), len(indexnu), len(mdb.nu_lines), len(mdb.dev_nu_lines), len(mdb.alpha_ref), len(mdb.n_Texp), len(mdb.A)]
    
    Ngat = np.unique(lenarr)
    if len(Ngat) > 1:
        print("Error: Length mismatch")
    #else:
    #    print("Nline gathered=",Ngat[0])
    
    return mdb, cnu, indexnu

    
def MdbExomol_plg(path_database, nus, Tgue, errTgue=500., Nelower=7, assess_width=1., threshold_persist_freezing=10000., crit=0., coefTgue=0.):
    """Obtain molecular database (mdb) from the Exomol while reducing the number of weak lines by plg. (This function summarizes the plg procedure.)
    
    Args:
        path_database: path for Exomol data directory/tag. For instance, "/home/CO/12C-16O/Li2015"
        nus: wavenumber grid for model
        Tgue: rough guess on the typical atmospheric temperature
        errTgue: expected one-sided error of Tgue
        Nelower: The number of elower grid
        assess_width: To save computation time, let us use only this wavelength width in the middle of the whole range to optimize coefTgue. (Note that too narrow might cause a ValueError.)
        threshold_persist_freezing: How weak lines compared to the deepest one will you persist freezing (not retaken)?
        crit: line strength lower limit for extraction in exojax.spec.moldb.MdbExomol
        coefTgue: coefficient for Tgue to optimize elower_grid

    Returns:
        mdb: molecular database instance
        cnu: contribution of wavenumber for LSD
        indexnu: nu index
        
    """
    from exojax.spec import moldb, initspec
    from exojax.utils.grids import wavenumber_grid
    mdb = moldb.MdbExomol(path_database, nus, crit=crit, Ttyp=Tgue)
    molmass = mdb.molmass
    
    if coefTgue == 0.:
        wls, wll = min(1.e8/nus), max(1.e8/nus)
        nusc, wavc, resoc = wavenumber_grid( (wll+wls-assess_width)/2, (wll+wls+assess_width)/2, int(assess_width/0.05), unit="AA", xsmode="modit")
        mdbc = moldb.MdbExomol(path_database, nusc)
        coefTgue = optimize_coefTgue(Tgue, nusc, mdbc, molmass, Nelower, errTgue)
        print("coefTgue =", coefTgue, " (by assessing", len(mdbc.A), "lines in", assess_width, "angstrom width.)"  )
    
    cnu,indexnu,R,pmarray = initspec.init_modit(mdb.nu_lines, nus)
    alpha_ref_grid, n_Texp_grid, index_gamma = make_gamma_grid_exomol(mdb)
    Ngamma = len(alpha_ref_grid)

    #Generate plg and Update mdb
    qlogsij0, qcnu, num_unique, elower_grid, frozen_mask, nonzeropl_mask = plg_elower_addcon(\
        index_gamma, Ngamma, cnu, indexnu, nus, mdb, Tgue, errTgue, \
        Nelower=Nelower, Tmargin=0., coefTgue=coefTgue, threshold_persist_freezing=threshold_persist_freezing)
    mdb, cnu, indexnu = gather_lines(mdb, Ngamma, len(nus), Nelower, nus, cnu, indexnu, qlogsij0, qcnu, elower_grid, alpha_ref_grid, n_Texp_grid, frozen_mask, nonzeropl_mask)
    
    return mdb, cnu, indexnu
    
