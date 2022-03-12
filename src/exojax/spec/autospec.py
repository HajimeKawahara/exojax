"""Automatic Opacity and Spectrum Generator."""
import time
from exojax.spec import defmol, defcia, moldb, contdb, planck, molinfo, lpf, dit, modit, initspec, response
from exojax.spec.opacity import xsection
from exojax.spec.hitran import SijT, doppler_sigma,  gamma_natural, gamma_hitran, normalized_doppler_sigma
from exojax.spec.exomol import gamma_exomol
from exojax.spec.rtransfer import rtrun, dtauM, dtauCIA
from exojax.spec.check_nugrid import check_scale_nugrid
from exojax.spec.make_numatrix import make_numatrix0
from exojax.utils.constants import c
from exojax.utils.instfunc import R2STD

import numpy as np
from jax import jit, vmap
import jax.numpy as jnp
import pathlib
import tqdm

__all__ = ['AutoXS', 'AutoRT']


class AutoXS(object):
    """exojax auto cross section generator."""

    def __init__(self, nus, database, molecules, databasedir='.database', memory_size=30, broadf=True, crit=0.0, xsmode='auto', autogridconv=True, pdit=1.5):
        """
        Args:
           nus: wavenumber bin (cm-1)
           database: database= HITRAN, HITEMP, ExoMol
           molecules: molecule name
           memory_size: memory_size required
           broadf: if False, the default broadening parameters in .def file is used
           crit: line strength criterion, ignore lines whose line strength are below crit.
           xsmode: xsmode for opacity computation (auto/LPF/DIT/MODIT)
           autogridconv: automatic wavenumber grid conversion (True/False). If you are quite sure the wavenumber grid you use, set False.
           pdit: threshold for DIT folding to x=pdit*STD_voigt 

        """
        self.molecules = molecules
        self.database = database
        self.nus = nus
        self.databasedir = databasedir
        self.memory_size = memory_size
        self.broadf = broadf
        self.crit = crit
        self.xsmode = xsmode
        self.identifier = defmol.search_molfile(database, molecules)
        self.pdit = pdit
        self.autogridconv = autogridconv

        if self.identifier is None:
            print('ERROR: '+molecules +
                  ' is an undefined molecule. Add your molecule in defmol.py and do pull-request!')
        else:
            print(self.identifier)
            self.init_database()

    def init_database(self):
        molpath = pathlib.Path(self.databasedir)/pathlib.Path(self.identifier)
        if self.database == 'HITRAN' or self.database == 'HITEMP':
            self.mdb = moldb.MdbHit(
                molpath, nurange=[self.nus[0], self.nus[-1]], crit=self.crit)
        elif self.database == 'ExoMol':
            print('broadf=', self.broadf)
            self.mdb = moldb.MdbExomol(molpath, nurange=[
                                       self.nus[0], self.nus[-1]], broadf=self.broadf, crit=self.crit)
        else:
            print('Select database from HITRAN, HITEMP, ExoMol.')

    def linest(self, T):
        """line strength.

        Args:
           T: temperature (K)

        Returns:
           line strength (cm)
        """
        if self.database == 'ExoMol':
            qt = self.mdb.qr_interp(T)
        elif self.database == 'HITRAN' or self.database == 'HITEMP':
            qt = self.mdb.Qr_line_HAPI(T)

        return SijT(T, self.mdb.logsij0, self.mdb.nu_lines, self.mdb.elower, qt)

    def xsection(self, T, P):
        """cross section.

        Args:
           T: temperature (K)
           P: pressure (bar)

        Returns:
           cross section (cm2)
        """

        mdb = self.mdb
        if self.database == 'ExoMol':
            gammaL = gamma_exomol(
                P, T, mdb.n_Texp, mdb.alpha_ref) + gamma_natural(mdb.A)
            molmass = mdb.molmass
        elif self.database == 'HITRAN' or self.database == 'HITEMP':
            gammaL = gamma_hitran(
                P, T, P, mdb.n_air, mdb.gamma_air, mdb.gamma_self) + gamma_natural(mdb.A)
            molmass = molinfo.molmass(self.molecules)

        Sij = self.linest(T)
        if self.xsmode == 'auto':
            xsmode = self.select_xsmode(len(mdb.nu_lines))
        else:
            xsmode = self.xsmode

        if xsmode == 'lpf' or xsmode == 'LPF':
            sigmaD = doppler_sigma(mdb.nu_lines, T, molmass)
            xsv = xsection(self.nus, mdb.nu_lines, sigmaD,
                           gammaL, Sij, memory_size=self.memory_size)
        elif xsmode == 'modit' or xsmode == 'MODIT':
            checknus = check_scale_nugrid(self.nus, gridmode='ESLOG')
            nus = self.autonus(checknus, 'ESLOG')
            cnu, indexnu, R_mol, pmarray = initspec.init_modit(
                mdb.nu_lines, nus)
            nsigmaD = normalized_doppler_sigma(T, molmass, R_mol)
            ngammaL = gammaL/(mdb.nu_lines/R_mol)
            ngammaL_grid = modit.ditgrid(ngammaL, res=0.1)
            xsv = modit.xsvector(cnu, indexnu, R_mol, pmarray,
                                 nsigmaD, ngammaL, Sij, nus, ngammaL_grid)
            if ~checknus and self.autogridconv:
                xsv = jnp.interp(self.nus, nus, xsv)
        elif xsmode == 'dit' or xsmode == 'DIT':
            sigmaD = doppler_sigma(mdb.nu_lines, T, molmass)
            checknus = check_scale_nugrid(self.nus, gridmode='ESLIN')
            nus = self.autonus(checknus, 'ESLIN')
            sigmaD_grid = dit.ditgrid(sigmaD, res=0.1)
            gammaL_grid = dit.ditgrid(gammaL, res=0.1)
            cnu, indexnu, pmarray = initspec.init_dit(mdb.nu_lines, nus)
            xsv = dit.xsvector(cnu, indexnu, pmarray, sigmaD,
                               gammaL, Sij, nus, sigmaD_grid, gammaL_grid)
            if ~checknus and self.autogridconv:
                xsv = jnp.interp(self.nus, nus, xsv)
        else:
            print('Error:', xsmode, ' is unavailable (auto/LPF/DIT).')
            xsv = None
        return xsv

    def autonus(self, checknus, tag='ESLOG'):
        if ~checknus:
            print('WARNING: the wavenumber grid does not look '+tag)
            if self.autogridconv:
                print('the wavenumber grid is interpolated.')
                if tag == 'ESLOG':
                    return np.logspace(jnp.log10(self.nus[0]), jnp.log10(self.nus[-1]), len(self.nus))
                if tag == 'ESLIN':
                    return np.linspace(self.nus[0], self.nus[-1], len(self.nus))
        return self.nus

    def select_xsmode(self, Nline):
        print('# of lines=', Nline)
        if Nline > 1000 and check_scale_nugrid(self.nus, gridmode='ESLOG'):
            print('MODIT selected')
            return 'MODIT'
        elif Nline > 1000 and check_scale_nugrid(self.nus, gridmode='ESLIN'):
            print('DIT selected')
            return 'DIT'
        else:
            print('LPF selected')
            return 'LPF'

    def xsmatrix(self, Tarr, Parr):
        """cross section matrix.

        Args:
           Tarr: temperature layer (K)
           Parr: pressure layer (bar)

        Returns:
           cross section (cm2)
        """
        mdb = self.mdb
        if self.database == 'ExoMol':
            qt = vmap(mdb.qr_interp)(Tarr)
            gammaLMP = jit(vmap(gamma_exomol, (0, 0, None, None)))(
                Parr, Tarr, mdb.n_Texp, mdb.alpha_ref)
            gammaLMN = gamma_natural(mdb.A)
            gammaLM = gammaLMP+gammaLMN[None, :]
            self.molmass = mdb.molmass
            SijM = jit(vmap(SijT, (0, None, None, None, 0)))(
                Tarr, mdb.logsij0, mdb.nu_lines, mdb.elower, qt)

        elif self.database == 'HITRAN' or self.database == 'HITEMP':
            qt = mdb.Qr_layer(Tarr)
            gammaLM = jit(vmap(gamma_hitran, (0, 0, 0, None, None, None)))(Parr, Tarr, Parr, mdb.n_air, mdb.gamma_air, mdb.gamma_self)\
                + gamma_natural(mdb.A)
            self.molmass = molinfo.molmass(self.molecules)
            SijM = jit(vmap(SijT, (0, None, None, None, 0)))(
                Tarr, mdb.logsij0, mdb.nu_lines, mdb.elower, qt)

        print('# of lines', len(mdb.nu_lines))
        memory_size = 15.0
        d = int(memory_size/(len(mdb.nu_lines)*4/1024./1024.))+1
        d2 = 100
        Nlayer, Nline = np.shape(SijM)
        if self.xsmode == 'auto':
            xsmode = self.select_xsmode(Nline)
        else:
            xsmode = self.xsmode
        print('xsmode=', xsmode)

        if xsmode == 'lpf' or xsmode == 'LPF':
            sigmaDM = jit(vmap(doppler_sigma, (None, 0, None)))(
                mdb.nu_lines, Tarr, self.molmass)
            Nj = int(Nline/d2)
            xsm = []
            for i in tqdm.tqdm(range(0, int(len(self.nus)/d)+1)):
                s = int(i*d)
                e = int((i+1)*d)
                e = min(e, len(self.nus))
                xsmtmp = np.zeros((Nlayer, e-s))
                for j in range(0, Nj+1):
                    s2 = int(j*d2)
                    e2 = int((j+1)*d2)
                    e2 = min(e2, Nline)
                    numatrix = make_numatrix0(
                        self.nus[s:e], mdb.nu_lines[s2:e2])
                    xsmtmp = xsmtmp +\
                        lpf.xsmatrix(
                            numatrix, sigmaDM[:, s2:e2], gammaLM[:, s2:e2], SijM[:, s2:e2])
                if i == 0:
                    xsm = np.copy(xsmtmp.T)
                else:
                    xsm = np.concatenate([xsm, xsmtmp.T])
            xsm = xsm.T
        elif xsmode == 'modit' or xsmode == 'MODIT':
            cnu, indexnu, R_mol, pmarray = initspec.init_modit(
                mdb.nu_lines, self.nus)
            nsigmaDl = normalized_doppler_sigma(
                Tarr, self.molmass, R_mol)[:, np.newaxis]
            ngammaLM = gammaLM/(mdb.nu_lines/R_mol)
            dgm_ngammaL = modit.dgmatrix(ngammaLM, 0.1)
            xsm = modit.xsmatrix(cnu, indexnu, R_mol, pmarray,
                                 nsigmaDl, ngammaLM, SijM, self.nus, dgm_ngammaL)
            xsm = self.nonnegative_xsm(xsm)
        elif xsmode == 'dit' or xsmode == 'DIT':
            cnu, indexnu, pmarray = initspec.init_dit(mdb.nu_lines, self.nus)
            sigmaDM = jit(vmap(doppler_sigma, (None, 0, None)))(
                mdb.nu_lines, Tarr, self.molmass)
            dgm_sigmaD = dit.dgmatrix(sigmaDM, 0.1)
            dgm_gammaL = dit.dgmatrix(gammaLM, 0.2)
            xsm = dit.xsmatrix(cnu, indexnu, pmarray, sigmaDM,
                               gammaLM, SijM, self.nus,
                               dgm_sigmaD, dgm_gammaL)
            xsm = self.nonnegative_xsm(xsm)
        else:
            print('No such xsmode=', xsmode)
            xsm = None

        return xsm

    def nonnegative_xsm(self, xsm):
        """check negative value of xsm.

        Args:

           xsm: xs matrix

        Returns:

           xsm negative eliminated
        """
        Nneg = len(xsm[xsm < 0.0])
        if Nneg > 0:
            print('Warning: negative cross section detected #=', Nneg,
                  ' fraction=', Nneg/float(jnp.shape(xsm)[0]*jnp.shape(xsm)[1]))
            return jnp.abs(xsm)
        else:
            return xsm


class AutoRT(object):
    """exojax auto radiative transfer."""

    def __init__(self, nus, gravity, mmw, Tarr, Parr, dParr=None, databasedir='.database', xsmode='auto', autogridconv=True):
        """
        Args:
           nus: wavenumber bin (cm-1)
           gravity: gravity (cm/s2)
           mmw: mean molecular weight of the atmosphere
           Tarr: temperature layer (K)
           Parr: pressure layer (bar)
           dParr: delta pressure (bar) optional
           databasedir: directory for saving database
           xsmode: xsmode for opacity computation (auto/LPF/DIT/MODIT)
           autogridconv: automatic wavenumber grid conversion (True/False). If you are quite sure the wavenumber grid you use, set False.

        """
        self.nus = nus
        self.gravity = gravity
        self.mmw = mmw
        self.nlayer = len(Tarr)
        self.Tarr = Tarr
        self.Parr = Parr
        self.xsmode = xsmode
        self.autogridconv = autogridconv

        if check_scale_nugrid(nus, gridmode='ESLOG'):
            print('nu grid is evenly spaced in log space (ESLOG).')
        elif check_scale_nugrid(nus, gridmode='ESLIN'):
            print('nu grid is evenly spaced in linear space (ESLIN).')
        else:
            print('nu grid is NOT evenly spaced in log nor linear space.')

        if dParr is None:
            from exojax.utils.chopstacks import buildwall
            wParr = buildwall(Parr)
            self.dParr = wParr[1:]-wParr[0:-1]
        else:
            self.dParr = dParr
        self.databasedir = databasedir
        self.sourcef = planck.piBarr(self.Tarr, self.nus)
        self.dtau = np.zeros((self.nlayer, len(nus)))
        print('xsmode=', self.xsmode)

    def addmol(self, database, molecules, mmr, crit=0.):
        """
        Args:
           database: database= HITRAN, HITEMP, ExoMol
           molecules: molecule name
           mmr: mass mixing ratio (float or ndarray for the layer)
           crit: line strength criterion, ignore lines whose line strength are below crit

        """
        mmr = mmr*np.ones_like(self.Tarr)
        axs = AutoXS(self.nus, database, molecules, crit=crit,
                     xsmode=self.xsmode, autogridconv=self.autogridconv)
        xsm = axs.xsmatrix(self.Tarr, self.Parr)
        dtauMx = dtauM(self.dParr, xsm, mmr, axs.molmass, self.gravity)
        self.dtau = self.dtau+dtauMx

    def addcia(self, interaction, mmr1, mmr2):
        """
        Args:
           interaction: e.g. H2-H2, H2-He
           mmr1: mass mixing ratio for molecule 1
           mmr2: mass mixing ratio for molecule 2

        """
        mol1, mol2 = defcia.interaction2mols(interaction)
        molmass1 = molinfo.molmass(mol1)
        molmass2 = molinfo.molmass(mol2)
        vmr1 = (mmr1*self.mmw/molmass1)
        vmr2 = (mmr2*self.mmw/molmass2)
        ciapath = pathlib.Path(self.databasedir) / \
            pathlib.Path(defcia.ciafile(interaction))
        cdb = contdb.CdbCIA(str(ciapath), [self.nus[0], self.nus[-1]])
        dtauc = dtauCIA(self.nus, self.Tarr, self.Parr, self.dParr, vmr1, vmr2,
                        self.mmw, self.gravity, cdb.nucia, cdb.tcia, cdb.logac)
        self.dtau = self.dtau+dtauc

    def rtrun(self):
        """running radiative transfer.

        Returns:
           spectrum (F0) in the unit of erg/s/cm2/cm-1

        Note:
           If you want to use the unit of erg/cm2/s/Hz, divide the output by the speed of light in cgs as Fx0=Fx0/ccgs, where  ccgs=29979245800.0. See #84
        """
        self.F0 = rtrun(self.dtau, self.sourcef)
        return self.F0

    def spectrum(self, nuobs, Rinst, vsini, RV, u1=0.0, u2=0.0, zeta=0., betamic=0., direct=True):
        """generating spectrum.

        Args:
           nuobs: observation wavenumber array
           Rinst: instrumental resolving power
           vsini: vsini for a stellar/planet rotation
           RV: radial velocity (km/s)
           u1: Limb-darkening coefficient 1
           u2: Limb-darkening coefficient 2
           zeta: macroturbulence distrubunce (km/s) in the radial-tangential model (Gray2005)
           betamic: microturbulence beta (STD, km/s)
           direct: True=use rigidrot/ipgauss_sampling, False=use rigidrot2, ipgauss2, sampling

        Returns:
           spectrum (F)
        """

        self.nuobs = nuobs
        self.Rinst = Rinst
        self.vsini = vsini
        self.u1 = u1
        self.u2 = u2
        self.zeta = zeta
        self.betamic = betamic
        self.RV = RV
        self.betaIP = R2STD(self.Rinst)
        beta = np.sqrt((self.betaIP)**2+(self.betamic)**2)
        F0 = self.rtrun()

        if len(self.nus) < 50000 and direct == True:
            print('rotation (1)')
            Frot = response.rigidrot(
                self.nus, F0, self.vsini, u1=self.u1, u2=self.u2)
            self.F = response.ipgauss_sampling(
                self.nuobs, self.nus, Frot, beta, self.RV)
        else:
            print('rotation (2): Require CuDNN')
            dv = c*(np.log(self.nus[1])-np.log(self.nus[0]))
            Nv = int(self.vsini/dv)+1
            vlim = Nv*dv
            varr_kernel = jnp.linspace(-vlim, vlim, 2*Nv+1)
            Frot = response.rigidrot2(
                self.nus, F0, varr_kernel, self.vsini, u1=self.u1, u2=self.u2)
            maxp = 5.0  # 5sigma
            Nv = int(maxp*beta/dv)+1
            vlim = Nv*dv
            varr_kernel = jnp.linspace(-vlim, vlim, 2*Nv+1)
            Fgrot = response.ipgauss2(self.nus, Frot, varr_kernel, beta)
            self.F = response.sampling(self.nuobs, self.nus, Fgrot, self.RV)

        return self.F
