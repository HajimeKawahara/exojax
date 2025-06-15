"""Base opacity calculator class for ExoJAX.

This module provides the foundational OpaCalc class that serves as the base
for all opacity calculation methods in ExoJAX (PreMODIT, MODIT, LPF).
It handles common functionality like grid management, aliasing modes,
and frequency domain stitching.
"""

from typing import Optional, Union, Literal, List
import logging

import jax.numpy as jnp
import numpy as np
from exojax.signal.ola import ola_output_length
from exojax.utils.instfunc import resolution_eslog

logger = logging.getLogger(__name__)

class OpaCont:
    """Common Opacity Calculator Class"""

    __slots__ = [
        "opainfo",
    ]

    def __init__(self):
        self.method = None  # which opacity cont method is used
        self.ready = False  # ready for opacity computation

class OpaCalc:
    """Base Opacity Calculator Class for ExoJAX.

    This class provides common functionality for all opacity calculation methods
    including grid management, aliasing modes, and frequency domain stitching.
    It serves as the foundation for PreMODIT, MODIT, and LPF opacity calculators.

    Attributes:
        nu_grid: Wavenumber grid in cm⁻¹
        opainfo: Method-specific opacity information and grids
        method: Opacity calculation method ("premodit", "modit", "lpf", or None)
        ready: Whether the calculator is ready for opacity computation
        alias: Aliasing mode for convolution ("close" or "open")
            - "close": Closed mode, aliasing sides overlap and won't be used
            - "open": Open mode, aliasing sides don't overlap and can be used in OLA
        nstitch: Number of frequency domain stitching segments
        cutwing: Wing cut ratio for convolution (1.0 for "close" mode)
        nu_grid_extended: Extended wavenumber grid for open mode
        filter_length_oneside: One-sided filter length based on cutwing ratio
        filter_length: Total filter length (2 * filter_length_oneside + 1)
        wing_cut_width: Min and max wing cut widths in cm⁻¹
    """

    def __init__(self, nu_grid: Union[np.ndarray, jnp.ndarray]) -> None:
        """Initialize the opacity calculator base class.

        Args:
            nu_grid: Wavenumber grid in cm⁻¹
        """
        self.nu_grid = nu_grid
        self.opainfo = None
        self.method: Optional[str] = None  # which opacity calc method is used
        self.ready: bool = False  # ready for opacity computation
        self.alias: Literal["close", "open"] = "close"  # close or open
        self.nstitch: int = 1

        # Parameters for open cross-section calculations
        self.cutwing: float = 1.0
        self.nu_grid_extended: Optional[Union[np.ndarray, jnp.ndarray]] = None
        self.filter_length_oneside: int = 0

    def set_aliasing(self) -> None:
        """Set up aliasing configuration for opacity calculations.

        Configures the extended wavenumber grid and wing cut widths based on
        the aliasing mode (close/open) and stitching parameters.

        Raises:
            ValueError: If aliasing configuration is invalid
        """
        from exojax.utils.grids import extended_wavenumber_grid

        self.set_filter_length_oneside_from_cutwing()

        if self.nstitch > 1:
            logger.info(
                "Cross section calculated in stitching mode with %d segments",
                self.nstitch,
            )
            self._setup_stitching_mode(extended_wavenumber_grid)
        elif self.alias == "close":
            logger.info(
                "Cross section calculated in closed mode - aliasing part cannot be used"
            )
            self._setup_closed_mode()
        elif self.alias == "open":
            logger.info(
                "Cross section calculated in open mode - aliasing part can be used"
            )
            self._setup_open_mode(extended_wavenumber_grid)

        else:
            raise ValueError(
                "nstitch > 1 or when nstitch =1 then alias should be 'close' or 'open'."
            )

        logger.info("Wing cut width = %s cm⁻¹", self.wing_cut_width)

    def _setup_stitching_mode(self, extended_wavenumber_grid) -> None:
        """Set up configuration for stitching mode.

        Args:
            extended_wavenumber_grid: Function to create extended grid
        """
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

    def _setup_closed_mode(self) -> None:
        """Set up configuration for closed aliasing mode."""
        resolution = resolution_eslog(self.nu_grid)
        lnx0 = np.log10(self.nu_grid[0]) - len(self.nu_grid) / resolution / np.log(10)
        lnx1 = np.log10(self.nu_grid[-1]) + len(self.nu_grid) / resolution / np.log(10)
        self.wing_cut_width = [
            self.nu_grid[0] - 10**lnx0,
            10**lnx1 - self.nu_grid[-1],
        ]

    def _setup_open_mode(self, extended_wavenumber_grid) -> None:
        """Set up configuration for open aliasing mode.

        Args:
            extended_wavenumber_grid: Function to create extended grid
        """
        self.nu_grid_extended = extended_wavenumber_grid(
            self.nu_grid, self.filter_length_oneside, self.filter_length_oneside
        )
        self.wing_cut_width = [
            self.nu_grid[0] - self.nu_grid_extended[0],
            self.nu_grid_extended[-1] - self.nu_grid[-1],
        ]

    def set_filter_length_oneside_from_cutwing(self) -> None:
        """Set the number of points to be added to each side of nu_grid based on cutwing ratio.

        Sets filter_length_oneside, filter_length, div_length, and output_length attributes.
        """
        self.div_length = len(self.nu_grid) // self.nstitch
        self.filter_length_oneside = int(len(self.nu_grid) * self.cutwing)
        self.filter_length = 2 * self.filter_length_oneside + 1
        self.output_length = ola_output_length(
            self.nstitch, self.div_length, self.filter_length
        )

    def check_nu_grid_reducible(self) -> None:
        """Check if nu_grid is reducible by nstitch.

        Raises:
            ValueError: If nu_grid length is not divisible by nstitch
        """
        if len(self.nu_grid) % self.nstitch != 0:
            raise ValueError(
                f"nu_grid length {len(self.nu_grid)} cannot be divided by nstitch={self.nstitch}"
            )
