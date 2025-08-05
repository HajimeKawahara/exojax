# calculate the Transmission model with Voigt×1 + cross-section
# updated 2024/08/06


# Import the modules
from exojax.utils.constants import Patm, Tref_original  # [bar/atm]
from exojax.database.core.line_strength import line_strength, doppler_sigma, gamma_hitran
from exojax.opacity import initspec, voigt
from exojax.opacity.lpf.lpf import xsmatrix as lpf_xsmatrix
from jax import jit, vmap
import numpy as np
from jax import config
import copy

config.update("jax_enable_x64", True)
# from HMC_defs_8data_nsep import gamma_hitran_nsep


# multiply the alpha at Sij
def Trans_model_MultiVoigt_test(
    nu_offset,
    alphas,
    gamma_refs,
    gamma_selfs,
    ns,
    Tarr,
    P_total,
    P_self,
    L,
    nMolecule,
    nu_grid,
    nu_data,
    mdb_voigt,  # masked mdb for Voigt fitting
    opa,
    sop_inst,
):

    Lbin = L / len(Tarr)  # spliting the bins
    # include the offset to wavenumber grid(it is shifted to the oppsite direction of actual offset)
    nu_data_offset_grid = nu_data

    # create the pressure array
    P_total_array = np.full(len(Tarr), P_total)
    P_self_array = np.full(len(Tarr), P_self)

    # cross-section matrix of weak lines at each Temperature channel
    xsmatrix_opa = opa.xsmatrix(Tarr, P_total_array)

    # Calculation for Voigt fitting
    # doppler width
    doppler_array = jit(vmap(doppler_sigma, (None, 0, None)))(
        mdb_voigt.nu_lines, Tarr, mdb_voigt.molmass
    )
    # partition function
    Tref_array = np.full(len(Tarr), Tref_original)
    qt = vmap(mdb_voigt.qr_interp_lines)(Tarr, Tref_array)

    # line strength
    SijM = jit(vmap(line_strength, (0, None, None, None, 0, None)))(
        Tarr,
        mdb_voigt.logsij0,
        mdb_voigt.nu_lines,
        mdb_voigt.elower,
        qt,
        Tref_original,
    )

    # test by HK
    P_self_array = np.zeros_like(P_self_array)
    gamma_selfs = np.zeros_like(gamma_selfs)

    # Lorentz width
    gamma_L_voigt = jit(vmap(gamma_hitran, (0, 0, 0, None, None, None)))(
        P_total_array,
        Tarr,
        P_self_array,
        ns,
        gamma_refs,
        gamma_selfs,
    )

    # create wavenumber matrix
    nu_matrix = initspec.init_lpf(mdb_voigt.nu_lines + nu_offset, nu_grid)

    # cross section
    xsmatrix_lpf = lpf_xsmatrix(nu_matrix, doppler_array, gamma_L_voigt, SijM * alphas)

    # summing up all layers
    xsmatrix_opa_alllayer = xsmatrix_opa.sum(axis=0)
    xsmatrix_lpf_alllayer = xsmatrix_lpf.sum(axis=0)

    # downsampling along to the instrumental resolution, include the effect of offset, trim the adjust range region
    xsmatrix_opa_specgrid = sop_inst.sampling(
        xsmatrix_opa_alllayer, 0, nu_data_offset_grid
    )
    xsmatrix_lpf_specgrid = sop_inst.sampling(
        xsmatrix_lpf_alllayer, 0, nu_data_offset_grid
    )

    return xsmatrix_opa_specgrid, xsmatrix_lpf_specgrid


def create_mdbs_multi(mdb, strline_ind_array_nu):
    # Create the mdb copy and remove the strongest line
    mdb_weak = copy.deepcopy(mdb)

    # Ensure strline_ind is a 1D array
    strline_inds = np.array(strline_ind_array_nu).flatten()
    nu_centers = mdb_weak.nu_lines[strline_inds]

    # nu_centers = [mdb_weak.nu_lines[i] for i in strline_inds]

    # Create a mask: Set True for indices not included in strline_ind_array
    mask = np.ones_like(mdb_weak.nu_lines, dtype=bool)
    mask[strline_inds] = False  # Set positions corresponding to strline_ind to False

    # Apply the mask
    mdb_weak.apply_mask_mdb(mask)

    # Check if the mdb_weak has one less data point
    if len(mdb_weak.nu_lines) != len(mdb.nu_lines) - strline_inds.size:
        raise ValueError(
            "mdb_weak does not have the correct number of data points after removing the strongest line."
        )

    # Create the mdb class only including the line for Voigt fitting
    mdb_voigt = copy.deepcopy(mdb)
    mdb_voigt.apply_mask_mdb(~mask)
    strline_inds = np.atleast_1d(strline_inds)
    mdb_voigt.logsij0 = mdb_voigt.logsij0[strline_inds]

    # Check if the mdb_voigt contains only the strongest line
    if len(mdb_voigt.nu_lines) != strline_inds.size:
        raise ValueError("mdb_voigt does not contain only the strong lines.")
    if not np.all(np.isin(mdb_voigt.nu_lines, nu_centers)):
        raise ValueError("The strong line in mdb_voigt does not match nu_center_array.")
    return mdb_weak, nu_centers, mdb_voigt


def calc_dnumber_isobaric(T_array, P0, T0):
    """
    #Calculate the number density at each region
    Assumption(if there is "i" Temperature regions with same volume)
       V1 = V2 =...= Vi = V_total / i
       P1 = P2 =...= Pi = P_total
       ∴ n1 * T1 = n2 * T2 = .... ni * Ti,
         ni = T1 * n1 / Ti

    Also, the Total amount of molecule is equal to the sum of each amount of molecule
        n_total * V_total = n1 * V1 +...+ ni * Vi
        n_total * i * V1 = (n1 + T1 / T2 * n1 + T1 / T3 * n1 +...+ T1 / Ti * n1) * V1
        n_total * i = n1 * T1 * (1 / T1 + 1 / T2 +...+1 / Ti)
        ∴ n1 = n_total * i / (T1 * Tinv_sum), (Tinv_sum = 1 / T1 + 1 / T2 +...+1 / Ti)

        P_total = P1 = n1 * k_b * T1
                     = n_total * i /(T1 * Tinv_sum) * k_b * T1
                     = n_total * i * k_B /Tinv_sum

    """

    from exojax.atm.idealgas import number_density
    from exojax.utils.constants import kB

    # Calculate the total number density and volume at the initial condition by PV=n*k_b*T
    n_total = number_density(P0, T0)  # P[bar]. n=P/k_b*T
    # V_total = (n_total * kB * T0) / P0
    nlayer = len(T_array)  # number of Temperature layer

    Tinv_sum = 0
    for i in range(nlayer):
        Tinv_sum += 1 / T_array[i]
    n_array = n_total * nlayer / (T_array * Tinv_sum)

    # Calculate the pressure at given Temperatures(1e-6:conversion factor of the Pa to bar)
    P_total = nlayer * kB * 1.0e-6 * n_total / (Tinv_sum)

    # print(nlayer/Tinv_sum)
    return n_array, P_total
