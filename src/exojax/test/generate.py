"""generate test data for exomol

"""

from exojax.spec import api
from exojax.test.emulate_mdb import mock_wavenumber_grid
import numpy as np
import pathlib
import os
from jax import config

config.update("jax_enable_x64", True)

sample_directory_name = {"CO": "CO/12C-16O/SAMPLE", "H2O": "H2O/1H2-16O/SAMPLE"}


def gendata_moldb(molecule="CO", compress_states=False):
    """generate test data for CO/H2O exomol
    
    Args:
        molecule (str, optional): "CO" or "H2O". Defaults to "CO".
        compress_states (bool, optional): compress states file. Defaults to False.
        input_state_filename (str, optional): input state filename. Defaults to "1H2-16O__POKAZATEL.states".
    """

    nus, wav, res = mock_wavenumber_grid()

    exomoldb = {
        "CO": ".database/CO/12C-16O/Li2015",
        "H2O": ".database/H2O/1H2-16O/POKAZATEL",
    }
    line_strength_criterion = {"CO": 0.0, "H2O": 1.0e-30}
    mdb = api.MdbExomol(
        exomoldb[molecule], nus, inherit_dataframe=True, gpu_transfer=True, broadf=False
    )
    line_mask = mdb.df["Sij0"] > line_strength_criterion[molecule]
    trans_filename = {"CO": "12C-16O__SAMPLE.trans", "H2O": "1H2-16O__SAMPLE__04300-04400.trans"}
    state_filename = {"CO": "12C-16O__SAMPLE.states", "H2O": "1H2-16O__SAMPLE.states"}
    input_state_filename = {"CO":"12C-16O__Li2015.states", "H2O":"1H2-16O__POKAZATEL.states"}
    mask = (mdb.df["nu_lines"] <= nus[-1]) * (mdb.df["nu_lines"] >= nus[0]) * line_mask
    masked_df = mdb.df[mask]
    print(len(masked_df["nu_lines"].values), "lines are selected")
    if not os.path.exists(sample_directory_name[molecule]):
        os.makedirs(sample_directory_name[molecule])
    cdir = os.getcwd()
    os.chdir(sample_directory_name[molecule])
    save_trans(trans_filename[molecule], masked_df)
    if compress_states:
        save_states(input_state_filename[molecule], state_filename[molecule], masked_df)
    os.chdir(cdir)
    
def make_hdf(molecule):
    path = pathlib.Path(sample_directory_name[molecule]).expanduser()
    print(path)
    nus, wav, res = mock_wavenumber_grid()
    mdb = api.MdbExomol(str(path), nus, inherit_dataframe=True, gpu_transfer=True, broadf=False)


def save_trans(trans_filename, masked_df):
    masked_df.export_csv(
        trans_filename,
        columns=["i_upper", "i_lower", "A", "nu_lines"],
        sep="\t",
        header=False,
    )
    import bz2

    print("bunzip2 ",trans_filename)
    with open(trans_filename, "rb") as f_in:
        with bz2.open(trans_filename + ".bz2", "wb") as f_out:
            f_out.writelines(f_in)

def save_states(input_state_filename, state_filename, masked_df):
    iup=masked_df["i_upper"].values
    ilow=masked_df["i_lower"].values
    ilist = np.unique(np.concatenate([iup,ilow]))
    
    with open(input_state_filename, 'r', encoding='utf-8') as infile, \
        open(state_filename, 'w', encoding='utf-8') as outfile:
        for line in infile:
            row = line.strip().split()
            if not row:
                continue
            try:
                if int(row[0]) in ilist:
                    outfile.write(line)
            except ValueError:
                outfile.write(line)

    import bz2

    print("bunzip2 ",state_filename)
    with open(state_filename, "rb") as f_in:
        with bz2.open(state_filename + ".bz2", "wb") as f_out:
            f_out.writelines(f_in)



if __name__ == "__main__":
    # regenerate hdf for exomol

    molecule = "H2O" #
    #molecule = "CO" #
    
    masked_df = gendata_moldb(molecule, compress_states=True)
    print("********************* NOTICE *****************************")
    print("cp some files (.broad, .def, .pf, .states (bunzip2))")
    print(" from "
        + sample_directory_name[molecule]
        + " or other .database/"
        + molecule
        + " to "
        + sample_directory_name[molecule]        
    )
    print("then do make_hdf")
    print("**********************************************************")
    
    make_hdf(molecule)
    print(">cp -rf " + molecule + " ../data/testdata/")
