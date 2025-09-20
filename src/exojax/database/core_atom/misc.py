
import jax.numpy as jnp
import numpy as np
from jax.lax import scan

from exojax.database import core_atom.io

def get_unique_species(adb):
    """Extract a unique list of line contributing species from VALD atomic
    database (adb)

    Args:
        adb: adb instance made by the AdbVald class in moldb.py

    Returns:
        uspecies: unique elements of the combination of ielem and iion (jnp.array with a shape of N_UniqueSpecies x 2(ielem and iion))
    """
    seen = []

    def get_unique_list(seq):
        return [x for x in seq if x not in seen and not seen.append(x)]

    uspecies = jnp.array(get_unique_list(jnp.vstack([adb.ielem, adb.iion]).T.tolist()))
    return uspecies


def ielemion_to_FastChemSymbol(ielem, iion):
    """Translate atomic number and ionization level into SpeciesSymbol in
    FastChem.

    Args:
        ielem:  atomic number (int) (e.g., Fe=26)
        iion:  ionized level (int) (e.g., neutral=1, singly)

    Returns:
        SpeciesSymbol in FastChem (str) (cf. https://github.com/exoclime/FastChem/blob/master/input/logK_ext.dat)
    """
    return (core_atom.io.PeriodicTable[ielem] + "1" + "+" * (iion - 1)).rstrip("1")


def get_VMR_uspecies(
    uspecies,
    mods_ID=jnp.array(
        [
            [0, 0],
        ]
    ),
    mods=jnp.array(
        [
            0,
        ]
    ),
):
    """Extract VMR arrays of the species that contribute the opacity
    ("uspecies" made with "get_unique_species")

    Args:
        uspecies: jnp.array of unique list of the species contributing the opacity [N_species x 2(ielem and iion)]
        mods_ID: jnp.array listing the species whose abundances are different from the solar [N_modified_species x 2(ielem and iion)]
        mods: jnp.array of each abundance deviation from the Sun [dex] for each modified species listed in mods_ID [N_modified_species]

    Returns:
        VMR_uspecies: jnp.array of volume mixing ratio [N_species]
    """
    mods_ID_uspecies = jnp.zeros(len(mods_ID), dtype=int)

    def f_miu(i_and_arr, sp):
        i, arr = i_and_arr
        i_and_arr = i + 1, jnp.where(
            ((mods_ID[:, 0] == sp[0]) & (mods_ID[:, 1] == sp[1])), i, arr
        )
        return (i_and_arr, sp)

    mods_ID_uspecies = scan(f_miu, (0, mods_ID_uspecies), uspecies)[0][1]

    ipccd = core_atom.io.load_atomicdata()
    ItIoI = core_atom.io.ielem_to_index_of_ipccd
    Narr = jnp.array(10 ** (ipccd["solarA"]))  # number density in the Sun

    def f_vmr(i, sp):
        return (
            i,
            jnp.where(
                sp[1] == 1,
                Narr[ItIoI[sp[0]]] / jnp.sum(Narr),
                Narr[ItIoI[sp[0]]] / jnp.sum(Narr) * 1e-10,
            ),
        )

    VMR_uspecies = scan(f_vmr, 0, uspecies)[1]

    def f_mod(i_and_VMR, i_MIU):
        i, VMR_uspecies = i_and_VMR
        i_and_VMR = i + 1, VMR_uspecies.at[i_MIU].set(
            VMR_uspecies[i_MIU] * 10 ** mods[i]
        )
        return (i_and_VMR, i_MIU)

    VMR_uspecies = scan(f_mod, (0, VMR_uspecies), mods_ID_uspecies)[0][1]

    return VMR_uspecies


def get_VMR_uspecies_FC(FCSpIndex_uspecies, mixing_ratios):
    """By using FastChem, extract volume mixing ratio (VMR) of the species that
    contribute the opacity ("uspecies" made with "get_unique_species")

    Args:
        FCSpIndex_uspecies: SpeciesIndex in FastChem for each species of interest [N_species]
        mixing_ratios: volume mixing ratios of all available gases calculated using fastchem2_call.run_fastchem [N_layer x N_species]

    Returns:
        VMR_uspecies: VMR of each species in each atmospheric layer [N_species x N_layer]
    """

    def floop(i_sp, VMR_sp):
        VMR_sp = mixing_ratios[:, FCSpIndex_uspecies[i_sp]]
        i_sp = i_sp + 1
        return (i_sp, VMR_sp)

    i, VMR_uspecies = scan(floop, 0, jnp.zeros(len(FCSpIndex_uspecies)))
    return VMR_uspecies


def uspecies_info(
    uspecies,
    ielem_to_index_of_ipccd,
    mods_ID=jnp.array(
        [
            [0, 0],
        ]
    ),
    mods=jnp.array(
        [
            0,
        ]
    ),
    mods_id_trans=jnp.array([]),
):
    """Provide arrays of information of the species that contribute the opacity
    ("uspecies" made with "get_unique_species")

    Args:
        uspecies: jnp.array of unique list of the species contributing the opacity
        ielem_to_index_of_ipccd: jnp.array for conversion from ielem to the index of ipccd
        mods_ID: jnp.array listing the species whose abundances are different from the solar
        mods: jnp.array of each abundance deviation from the Sun [dex] for each modified species in mods_ID
        mods_id_trans: jnp.array for converting index in "mods_ID" of each species into index in uspecies

    Returns:
        MMR_uspecies_list: jnp.array of mass mixing ratio in the Sun of each species in "uspecies"
        atomicmass_uspecies_list: jnp.array of atomic mass [amu] of each species in "uspecies"
        mods_uspecies_list: jnp.array of abundance deviation from the Sun [dex] for each species in "uspecies"
    """
    ipccd = core_atom.io.load_atomicdata()
    Narr = jnp.array(10 ** (ipccd["solarA"]))  # number density
    # mass of each neutral atom per particle [amu]
    massarr = jnp.array(ipccd["mass"])
    Nmassarr = Narr * massarr  # mass density of each neutral species

    def floopMMR(i, arr):
        arr = Nmassarr[ielem_to_index_of_ipccd[uspecies[i][0]]] / jnp.sum(Nmassarr)
        i = i + 1
        return (i, arr)

    MMR_uspecies_list = scan(floopMMR, 0, np.zeros(len(uspecies)))[1]

    def floopAM(i, arr):
        arr = massarr[ielem_to_index_of_ipccd[uspecies[i][0]]]
        i = i + 1
        return (i, arr)

    atomicmass_uspecies_list = scan(floopAM, 0, np.zeros(len(uspecies)))[1]  # [amu]

    # for i, mit in enumerate(mods_id_trans):
    # mods_uspecies_list[mit] = mods[i]
    def f_Mmul(msi, null):
        ms, i = msi
        mit = mods_id_trans[i]
        ms = ms.at[mit].set(mods[i])
        i = i + 1
        msi = [ms, i]
        return (msi, null)

    length = len(mods)

    def g_Mmul(msi0):
        msi, null = scan(f_Mmul, msi0, None, length)
        return msi[0]

    mods_uspecies_list = jnp.zeros(len(uspecies))
    mods_uspecies_list = g_Mmul([mods_uspecies_list, 0])

    return (MMR_uspecies_list, atomicmass_uspecies_list, mods_uspecies_list)


def sep_arr_of_sp(arr, adb, trans_jnp=True, inttype=False):
    """Separate by species (atoms or ions) the jnp.array stored as an instance variable in adb, and pad with zeros to adjust the length

    Args:
        arr: array of a parameter (one of the attributes of adb below) to be separated [N_line]
        adb: adb instance made by the AdbVald class in moldb.py
        trans_jnp: if True, the output is converted to jnp.array (dtype='float32')
        inttype: if True (along with trans_jnp = True), the output is converted to jnp.array of dtype='int32'

    Returns:
        arr_stacksp: species-separated array [N_species x N_line_max]
    """
    uspecies = get_unique_species(adb)
    N_usp = len(uspecies)
    len_of_eachsp = np.zeros(N_usp, dtype="int")
    for i, sp in enumerate(uspecies):
        len_of_eachsp[i] = len(np.where((adb.ielem == sp[0]) * (adb.iion == sp[1]))[0])
    L_max = np.max(len_of_eachsp)

    arr_stacksp = np.zeros([N_usp, L_max])

    def pad0(arr, L):
        return np.pad(arr, ((0, L - len(arr))))

    for i, sp in enumerate(uspecies):
        index_sp = np.where((adb.ielem == sp[0]) * (adb.iion == sp[1]))[0]
        arr_t = jnp.take(arr, index_sp)
        arr_tp = pad0(arr_t, L_max)
        arr_stacksp[i] = arr_tp
    if trans_jnp:
        if inttype:
            arr_stacksp = jnp.array(arr_stacksp, dtype="int32")
        else:
            arr_stacksp = jnp.array(arr_stacksp)

    return arr_stacksp


def padding_2Darray_for_each_atom(orig_arr, adb, sp):
    """Extract only data of the species of interest from 2D-array and pad with
    zeros to adjust the length.

    Args:
        orig_arr: array [N_any (e.g., N_nu or N_layer), N_line]
            Note that if your ARRAY is 1D, it must be broadcasted with ARRAY[None,:], and the output must be also reshaped with OUTPUTARRAY.reshape(ARRAY.shape)
        adb: adb instance made by the AdbVald class in moldb.py
        sp: array of [ielem, iion]

    Returns:
       padded_valid_arr
    """
    orig_arr = orig_arr.T
    valid_indices = jnp.where(
        (adb.ielem == sp[0]) * (adb.iion == sp[1]),
        jnp.arange(adb.ielem.shape[0]),
        adb.ielem.shape[0],
    )
    padding_zero = jnp.zeros([1, orig_arr.shape[1]])
    padded_arr = jnp.concatenate([orig_arr, padding_zero])
    padded_valid_arr = padded_arr[jnp.sort(valid_indices)]
    padded_valid_arr = padded_valid_arr.T
    return padded_valid_arr


