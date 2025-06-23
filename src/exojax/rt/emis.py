from exojax.rt.common import ArtCommon
from exojax.rt.planck import piBarr
from exojax.rt.rtransfer import (
    rtrun_emis_pureabs_fbased2st,
    rtrun_emis_pureabs_ibased,
    rtrun_emis_pureabs_ibased_linsap,
    rtrun_emis_scat_fluxadding_toonhm,
    rtrun_emis_scat_lart_toonhm,
)



class ArtEmisScat(ArtCommon):
    """Atmospheric RT for emission w/ scattering

    Attributes:
        pressure_layer: pressure profile in bar

    """

    def __init__(
        self,
        pressure_top=1.0e-8,
        pressure_btm=1.0e2,
        nlayer=100,
        nu_grid=None,
        rtsolver="fluxadding_toon_hemispheric_mean",
    ):
        """initialization of ArtEmisScat

        Args:
            pressure_top (float, optional): top pressure in bar. Defaults to 1.0e-8.
            pressure_btm (float, optional): bottom pressure in bar. Defaults to 1.0e2.
            nlayer (int, optional): the number of the atmospheric layers. Defaults to 100.
            nu_grid (float, array, optional): the wavenumber grid. Defaults to None.
            rtsolver (str): Radiative Transfer Solver,
                "fluxadding_toon_hemispheric_mean" (default),
                "lart_toon_hemispheric_mean"

        """
        super().__init__(pressure_top, pressure_btm, nlayer, nu_grid)
        self.rtsolver = rtsolver
        self.method = "emission_with_scattering_using_" + self.rtsolver

    def run(
        self,
        dtau,
        single_scattering_albedo,
        asymmetric_parameter,
        temperature,
        nu_grid=None,
        show=False,
    ):
        """run radiative transfer

        Args:
            dtau (2D array): optical depth matrix, dtau  (N_layer, N_nus)
            temperature (1D array): temperature profile (Nlayer)
            nu_grid (1D array): if nu_grid is not initialized, provide it.
            show: plot intermediate results

        Returns:
            1D array: spectrum
        """
        if self.nu_grid is not None:
            sourcef = piBarr(temperature, self.nu_grid)
        elif nu_grid is not None:
            sourcef = piBarr(temperature, nu_grid)
        else:
            raise ValueError("the wavenumber grid is not given.")

        if self.rtsolver == "lart_toon_hemispheric_mean":
            (
                spectrum,
                cumTtilde,
                Qtilde,
                trans_coeff,
                scat_coeff,
                piB,
            ) = rtrun_emis_scat_lart_toonhm(
                dtau, single_scattering_albedo, asymmetric_parameter, sourcef
            )
            if show:
                from exojax.plot.rtplot import comparison_with_pure_absorption

                spec, spec_pure = comparison_with_pure_absorption(
                    cumTtilde, Qtilde, spectrum, trans_coeff, scat_coeff, piB
                )
                return spectrum, spec, spec_pure

        elif self.rtsolver == "fluxadding_toon_hemispheric_mean":
            spectrum = rtrun_emis_scat_fluxadding_toonhm(
                dtau, single_scattering_albedo, asymmetric_parameter, sourcef
            )

        else:
            print("rtsolver=", self.rtsolver)
            raise ValueError("Unknown radiative transfer solver (rtsolver).")

        return spectrum


class ArtEmisPure(ArtCommon):
    """Atmospheric RT for emission w/ pure absorption

    Notes:
        The default radiative transfer scheme has been the intensity-based transfer since version 1.5

    Attributes:
        pressure_layer: pressure profile in bar

    """

    def __init__(
        self,
        pressure_top=1.0e-8,
        pressure_btm=1.0e2,
        nlayer=100,
        nu_grid=None,
        rtsolver="ibased",
        nstream=8,
    ):
        """
        initialization of ArtEmisPure

        Args:
            pressure_top (float, optional): top pressure in bar. Defaults to 1.0e-8.
            pressure_btm (float, optional): bottom pressure in bar. Defaults to 1.0e2.
            nlayer (int, optional): the number of the atmospheric layers. Defaults to 100.
            nu_grid (float, array, optional): the wavenumber grid. Defaults to None.
            rtsolver (str, optional): radiative transfer solver (ibased, fbased2st, ibased_linsap). Defaults to "ibased".
            nstream (int, optional): the number of stream. Defaults to 8. Should be 2 for rtsolver = fbased2st
        """
        super().__init__(pressure_top, pressure_btm, nlayer, nu_grid)
        self.method = "emission_with_pure_absorption"
        self.set_capable_rtsolvers()
        self.validate_rtsolver(rtsolver, nstream)

    def set_capable_rtsolvers(self):
        self.rtsolver_dict = {
            "fbased2st": rtrun_emis_pureabs_fbased2st,
            "ibased": rtrun_emis_pureabs_ibased,
            "ibased_linsap": rtrun_emis_pureabs_ibased_linsap,
        }

        self.valid_rtsolvers = list(self.rtsolver_dict.keys())

        # source function to be used in rtsolver
        self.source_position_dict = {
            "fbased2st": "representative",
            "ibased": "representative",
            "ibased_linsap": "upper_boundary",
        }

        self.rtsolver_explanation = {
            "fbased2st": "Flux-based two-stream solver, isothermal layer (ExoJAX1, HELIOS-R1 like)",
            "ibased": "Intensity-based n-stream solver, isothermal layer (e.g. NEMESIS, pRT like)",
            "ibased_linsap": "Intensity-based n-stream solver w/ linear source approximation (linsap), see Olson and Kunasz (e.g. HELIOS-R2 like)",
        }

    def validate_rtsolver(self, rtsolver, nstream):
        """validates rtsolver

        Args:
            rtsolver (str): rtsolver
            nstream (int): the number of streams

        """

        if rtsolver in self.valid_rtsolvers:
            self.rtsolver = rtsolver
            self.source_position = self.source_position_dict[self.rtsolver]
            print("rtsolver: ", self.rtsolver)
            print(self.rtsolver_explanation[self.rtsolver])
        else:
            str_valid_rtsolvers = (
                ", ".join(self.valid_rtsolvers[:-1])
                + f", or {self.valid_rtsolvers[-1]}"
            )
            raise ValueError("Unknown rtsolver. Use " + str_valid_rtsolvers)
        if rtsolver == "fbased2st" and nstream != 2:
            raise ValueError(
                "fbased2st (flux-based two-stream) rtsolver requires nstream = 2."
            )
        self.nstream = nstream

    def run(self, dtau, temperature, nu_grid=None):
        """run radiative transfer

        Args:
            dtau (2D array): optical depth matrix, dtau  (N_layer, N_nus)
            temperature (1D array): temperature profile (Nlayer)
            nu_grid (1D array): if nu_grid is not initialized, provide it.

        Returns:
            1D array: emission spectrum
        """
        if self.nu_grid is not None:
            nu_grid = self.nu_grid

        sourcef = piBarr(temperature, nu_grid)
        rtfunc = self.rtsolver_dict[self.rtsolver]

        if self.rtsolver == "fbased2st":
            return rtfunc(dtau, sourcef)
        elif self.rtsolver == "ibased" or self.rtsolver == "ibased_linsap":
            from exojax.rt.rtransfer import initialize_gaussian_quadrature

            mus, weights = initialize_gaussian_quadrature(self.nstream)
            return rtfunc(dtau, sourcef, mus, weights)
