#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluation of molecular lines (center)

* This module evaluates the contibution of molecular lines to an emission spectrum.
* This module only consider the line center to reduces computation complexity.
"""

from jax import jit, vmap
import jax.numpy as jnp
from exojax.special.erfcx import erfcx
from exojax.spec.rtransfer import dtauM, dtauCIA
from exojax.utils.constants import hcperk
import numpy as np
import tqdm
from exojax.spec.lpf import exomol


def reduceline_exomol(mdb, Parr, dParr, mmw, gravity, vmrH2, cdb, maxMMR, molmass, Tmodel, *Tparams):
    """Reduce lines for Exomol.

    Args:
       mdb: mdb (exomol)
       Parr: pressure layer (bar)
       dParr: delta pressure layer (bar)
       mmw: mean molecular weight of the atmosphere
       gravity: gravity (cm/s2)
       vmrH2: volume mixing ratio of H2
       cdb: cdb for continuum
       maxMMR: max Mass Mixing Ratio
       molmass_mol: molecualr mass of the molecule
       Tmodel: Tmodel function
       *Tparams: parameter sets

    Returns:
       mask: mask for exomol mdb
       maxcf: max contribution function,but for the last value of parameter set.
       maxcia: max cia level but for the last value of parameter set.
    """
    ONEARR = np.ones_like(Parr)
    parM = np.array(Tparams)
    Nparam, Msample = np.shape(parM)
    for k in tqdm.tqdm(range(Msample), desc='reduce lines'):
        Tarr = Tmodel(Parr, parM[:, k])
        SijM, gammaLM, sigmaDM = exomol(mdb, Tarr, Parr, molmass)
        mask_tmp, maxcf, maxcia = mask_weakline(
            mdb, Parr, dParr, Tarr, SijM, gammaLM, sigmaDM, maxMMR*ONEARR, molmass, mmw, gravity, vmrH2, cdb)
        if k == 0:
            mask = np.copy(mask_tmp)
        else:
            mask = mask+mask_tmp

    print('REDUCED ', len(mask), ' lines to ', np.sum(mask), ' lines')
    mdb.masking(mask)
    return mask, maxcf, maxcia


def mask_weakline(mdb_mol, Parr, dParr, Tarr, SijM, gammaLM, sigmaDM, MMR_mol, molmass_mol, mmw, g, vmrH2, cdbH2H2, margin=2, mask=None, Nlim=1000):
    """masking weak lines compared to CIA H2-H2 continuum.

    Args:
       mdb_mol: mdb
       Parr: pressure layer (bar)
       dParr: delta pressure layer (bar)
       Tarr: temperature layer (K)
       SijM: Sij matrix
       gammaLM: gamma coefficient matrix
       sigmaDM: Doppler broadening matrix
       MMR_mol: Mass Mixing Ratio of the molecule
       molmass_mol: molecualr mass of the molecule
       mmw: mean molecular weight of the atmosphere
       gravity: gravity (cm/s2)
       vmrH2: volume mixing ratio of H2
       cdbH2H2: cdb

    Returns:
       mask (weak line mask), maxcf (P at max contribution function for the molecule), maxcia (P at max contribution function for CIA)
    """

    xsm0 = xsmatrix0(sigmaDM, gammaLM, SijM)  # cross section at line centers
    dtaumol = dtauM(dParr, xsm0, MMR_mol, molmass_mol, g)
    ndtaumol = np.asarray(dtaumol)
    dtaucH2H2 = dtauCIA(mdb_mol.nu_lines, Tarr, Parr, dParr, vmrH2, vmrH2,
                        mmw, g, cdbH2H2.nucia, cdbH2H2.tcia, cdbH2H2.logac)
    ndtaucH2H2 = np.asarray(dtaucH2H2)

    Nl = len(mdb_mol.nu_lines)

    if Nl < Nlim:
        cf_mol = contfunc(dtaumol, mdb_mol.nu_lines, Parr, dParr, Tarr)
        maxcf = np.argmax(cf_mol, axis=0)
    else:
        M = int(float(Nl)/float(Nlim))+1
        maxcf = np.array([], dtype=np.int)
        for n in range(0, M):
            i = n*Nlim
            j = min((n+1)*Nlim, Nl)
            cf_mol = contfunc(
                ndtaumol[:, i:j], mdb_mol.nu_lines[i:j], Parr, dParr, Tarr)
            maxcf_tmp = np.argmax(cf_mol, axis=0)
            maxcf = np.concatenate([maxcf, maxcf_tmp])

    if Nl < Nlim:
        cfCIA = contfunc(dtaucH2H2, mdb_mol.nu_lines, Parr, dParr, Tarr)
        maxcia = np.argmax(cfCIA, axis=0)
    else:
        M = int(float(Nl)/float(Nlim))+1
        maxcia = np.array([], dtype=np.int)
        for n in range(0, M):
            i = n*Nlim
            j = min((n+1)*Nlim, Nl)
            cfCIA = contfunc(
                ndtaucH2H2[:, i:j], mdb_mol.nu_lines[i:j], Parr, dParr, Tarr)
            maxcia_tmp = np.argmax(cfCIA, axis=0)
            maxcia = np.concatenate([maxcia, maxcia_tmp])

    mask1 = (maxcf > 0)*(maxcf < maxcia+margin)
    if mask is None:
        mask = mask1
    else:
        mask = mask1+mask

    return mask, maxcf, maxcia


def contfunc(dtau, nu, Parr, dParr, Tarr):
    """contribution function.

    Args:
       dtau: delta tau array [N_layer, N_lines]
       nu: wavenumber array [N_lines]
       Parr: pressure array  [N_layer]
       dParr: delta pressure array  [N_layer]
       Tarr: temperature array  [N_layer]

    Returns:
       contribution function
    """

    tau = np.cumsum(dtau, axis=0)
    cf = np.exp(-tau)*dtau \
        * (Parr[:, None]/dParr[:, None]) \
        * nu[None, :]**3/(np.exp(hcperk*nu[None, :]/Tarr[:, None])-1.0)
    return cf


@jit
def voigt0(sigmaD, gammaL):
    """Voigt-Hjerting function at nu=nu0.

    Args:
       nu: wavenumber
       sigmaD: sigma parameter in Doppler profile
       gammaL: broadening coefficient in Lorentz profile

    Returns:
       v: Voigt profile at nu=nu0
    """

    sfac = 1.0/(jnp.sqrt(2)*sigmaD)
    v = sfac*erfcx(sfac*gammaL)/jnp.sqrt(jnp.pi)
    return v


@jit
def xsvector0(sigmaD, gammaL, Sij):
    """cross section at nu=nu0.

    Args:
       sigmaD: doppler sigma vector in R^Nline
       gammaL: gamma factor vector in R^Nline
       Sij: line strength vector in R^Nline

    Return:
       cross section vector in R^Nwav
    """
    vmap_voigt0 = vmap(voigt0, (0, 0), 0)
    return Sij*vmap_voigt0(sigmaD, gammaL)


@jit
def xsmatrix0(sigmaDM, gammaLM, SijM):
    """cross section matrix at nu=nu0.

    Args:
       sigmaDM: doppler sigma matrix in R^(Nlayer x Nline)
       gammaLM: gamma factor matrix in R^(Nlayer x Nline)
       SijM: line strength matrix in R^(Nlayer x Nline)

    Return:
       cross section matrix in R^(Nlayer x Nwav)
    """
    return vmap(xsvector0, (0, 0, 0))(sigmaDM, gammaLM, SijM)
