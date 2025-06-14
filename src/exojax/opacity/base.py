import numpy as np
from exojax.signal.ola import ola_output_length
from exojax.utils.instfunc import resolution_eslog


class OpaCalc:
    """Common Opacity Calculator Class

    Attributes:
        opainfo: information set used in each opacity method
        method (str,None): opacity calculation method, i.e. "premodit", "modit", "lpf"
        ready (bool): ready for opacity computation
        alias (bool): mode of the aliasing part for the convolution (MODIT/PreMODIT).
            False = the closed mode, left and right alising sides are overlapped and won't be used.
            True = the open mode, left and right aliasing sides are not overlapped and the alias part will be used in OLA.
        nu_grid_extended (jnp.array): extended wavenumber grid for the open mode
        filter_length_oneside (int): oneside number of points to be added to the left and right of the nu_grid based on the cutwing ratio
        filter_length (int): total number of points to be added to the left and right of the nu_grid based on the cutwing ratio
        cutwing (float): wingcut for the convolution used in open cross section. Defaults to 1.0. For alias="close", always 1.0 is used by definition.
        wing_cut_width (list): min and max wing cut width in cm-1


    """

    def __init__(self, nu_grid):
        self.nu_grid = nu_grid
        self.opainfo = None
        self.method = None  # which opacity calc method is used
        self.ready = False  # ready for opacity computation
        self.alias = "close"  # close or open
        self.nstitch = 1

        # open xsvector/xsmatrix
        self.cutwing = 1.0
        self.nu_grid_extended = None
        self.filter_length_oneside = 0

    def set_aliasing(self):
        """set the aliasing

        Raises:
            ValueError: alias should be 'close' or 'open'
        """
        from exojax.utils.grids import extended_wavenumber_grid

        self.set_filter_length_oneside_from_cutwing()

        if self.nstitch > 1:
            print("cross section is calculated in the stitching mode.")
            self.nu_grid_array = np.array(np.array_split(self.nu_grid, self.nstitch))
            self.nu_grid_extended_array = []
            for i in range(self.nstitch):
                self.nu_grid_extended_array.append(
                    extended_wavenumber_grid(
                        self.nu_grid_array[i, :],
                        self.filter_length_oneside,
                        self.filter_length_oneside,
                    )
                )
            self.nu_grid_extended_array = np.array(self.nu_grid_extended_array)
            self.wing_cut_width = [
                self.nu_grid[0] - self.nu_grid_extended_array[0, 0],
                self.nu_grid_extended_array[-1, -1] - self.nu_grid[-1],
            ]
        elif self.alias == "close":
            print(
                "cross section (xsvector/xsmatrix) is calculated in the closed mode. The aliasing part cannnot be used."
            )
            resolution = resolution_eslog(self.nu_grid)
            lnx0 = np.log10(self.nu_grid[0]) - len(self.nu_grid) / resolution / np.log(
                10
            )
            lnx1 = np.log10(self.nu_grid[-1]) + len(self.nu_grid) / resolution / np.log(
                10
            )
            self.wing_cut_width = [
                self.nu_grid[0] - 10**lnx0,
                10**lnx1 - self.nu_grid[-1],
            ]
        elif self.alias == "open":
            print(
                "cross section (xsvector/xsmatrix) is calculated in the open mode. The aliasing part can be used."
            )

            self.nu_grid_extended = extended_wavenumber_grid(
                self.nu_grid, self.filter_length_oneside, self.filter_length_oneside
            )
            self.wing_cut_width = [
                self.nu_grid[0] - self.nu_grid_extended[0],
                self.nu_grid_extended[-1] - self.nu_grid[-1],
            ]

        else:
            raise ValueError(
                "nstitch > 1 or when nstitch =1 then alias should be 'close' or 'open'."
            )

        print("wing cut width = ", self.wing_cut_width, "cm-1")

    def set_filter_length_oneside_from_cutwing(self):
        """sets the number of points to be added to the left and right (filter_lenth_oneside) of the nu_grid based on the cutwing ratio"""
        self.div_length = len(self.nu_grid) // self.nstitch
        self.filter_length_oneside = int(len(self.nu_grid) * self.cutwing)
        self.filter_length = 2 * self.filter_length_oneside + 1
        self.output_length = ola_output_length(
            self.nstitch, self.div_length, self.filter_length
        )

    def check_nu_grid_reducible(self):
        """check if nu_grid is reducible by ndiv

        Raises:
            ValueError: if nu_grid is not reducible by ndiv
        """
        if len(self.nu_grid) % self.nstitch != 0:
            msg = (
                "nu_grid_all length = "
                + str(len(self.nu_grid))
                + " cannot be divided by stitch="
                + str(self.nstitch)
            )
            raise ValueError(msg)
