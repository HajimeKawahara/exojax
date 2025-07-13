from exojax.rt.common import ArtCommon
from exojax.rt.chord import (
    chord_geometric_matrix,
    chord_geometric_matrix_lower,
    chord_optical_depth,
)
from exojax.rt.rtransfer import (
    rtrun_trans_pureabs_simpson,
    rtrun_trans_pureabs_trapezoid,
)


class ArtTransPure(ArtCommon):
    """Atmospheric Radiative Transfer for transmission spectroscopy

    Args:
        ArtCommon: ArtCommon class
    """

    def __init__(
        self, pressure_top=1.0e-8, pressure_btm=1.0e2, nlayer=100, integration="simpson"
    ):
        """initialization of ArtTransPure

        Args:
            pressure_top (float, optional): layer top pressure in bar. Defaults to 1.0e-8.
            pressure_btm (float, optional): layer bottom pressure in bar. Defaults to 1.0e2.
            nlayer (int, optional): The number of the layers Defaults to 100.
            integration (str, optional): Integration scheme ("simpson", "trapezoid"). Defaults to "simpson".

        Note:
            The users can choose the integration scheme of the chord integration from Trapezoid method or Simpson method.

        """
        super().__init__(pressure_top, pressure_btm, nlayer, nu_grid=None)
        self.method = "transmission_with_pure_absorption"
        self.set_capable_integration()
        self.set_integration_scheme(integration)

    def set_capable_integration(self):
        """sets integration scheme directory"""
        self.integration_dict = {
            "trapezoid": rtrun_trans_pureabs_trapezoid,
            "simpson": rtrun_trans_pureabs_simpson,
        }

        self.valid_integration = list(self.integration_dict.keys())

        self.integration_explanation = {
            "trapezoid": "Trapezoid integration, uses the chord optical depth at the lower boundary of the layers only",
            "simpson": "Simpson integration, uses the chord optical depth at the lower boundary and midppoint of the layers.",
        }

    def set_integration_scheme(self, integration):
        """sets and validates integration

        Args:
            integration (str): integration scheme, i.e. "trapezoid" or "simpson"

        """

        if integration in self.valid_integration:
            self.integration = integration
            print("integration: ", self.integration)
            print(self.integration_explanation[self.integration])
        else:
            str_valid_integration = (
                ", ".join(self.valid_integration[:-1])
                + f", or {self.valid_integration[-1]}"
            )
            raise ValueError(
                "Unknown integration (scheme). Use " + str_valid_integration
            )

    def run(self, dtau, temperature, mean_molecular_weight, radius_btm, gravity_btm):
        """run radiative transfer

        Args:
            dtau (2D array): optical depth matrix, dtau  (N_layer, N_nus)
            temperature (1D array): temperature profile (Nlayer)
            mean_molecular_weight (1D array): mean molecular weight profile, (Nlayer, from atmospheric top to bottom)
            radius_btm (float): radius (cm) at the lower boundary of the bottom layer, R0 or r_N
            gravity_btm (float): gravity (cm/s2) at the lower boundary of the bottom layer, g_N

        Returns:
            1D array: transit squared radius normalized by radius_btm**2, i.e. it returns (radius/radius_btm)**2

        Notes:
            This function gives the sqaure of the transit radius.
            If you would like to obtain the transit radius, take sqaure root of the output and multiply radius_btm.
            If you would like to compute the transit depth, divide the output by (stellar radius/radius_btm)**2

        """

        normalized_height, normalized_radius_lower = self.atmosphere_height(
            temperature, mean_molecular_weight, radius_btm, gravity_btm
        )
        normalized_radius_top = normalized_radius_lower[0] + normalized_height[0]
        cgm = chord_geometric_matrix_lower(normalized_height, normalized_radius_lower)
        dtau_chord_lower = chord_optical_depth(cgm, dtau)

        func = self.integration_dict[self.integration]
        if self.integration == "trapezoid":
            return func(
                dtau_chord_lower, normalized_radius_lower, normalized_radius_top
            )
        elif self.integration == "simpson":
            cgm_midpoint = chord_geometric_matrix(
                normalized_height, normalized_radius_lower
            )
            dtau_chord_midpoint = chord_optical_depth(cgm_midpoint, dtau)
            return func(
                dtau_chord_midpoint,
                dtau_chord_lower,
                normalized_radius_lower,
                normalized_height,
            )

    def run_ckd(self, dtau_ckd, temperature, mean_molecular_weight, radius_btm, 
                gravity_btm, weights):
        """run radiative transfer for CKD transmission
        
        Args:
            dtau_ckd (3D array): optical depth tensor, dtau (N_layer, Ng, Nbands)
            temperature (1D array): temperature profile (Nlayer)
            mean_molecular_weight (1D array): mean molecular weight profile (Nlayer)
            radius_btm (float): radius (cm) at the lower boundary of the bottom layer
            gravity_btm (float): gravity (cm/s2) at the lower boundary of the bottom layer
            weights (1D array): weights for the Gaussian quadrature (Ng,)
            
        Returns:
            1D array: transit squared radius normalized by radius_btm**2 (Nbands,)
        """
        import jax.numpy as jnp
        
        nlayer, Ng, Nbands = dtau_ckd.shape
        
        # Compute atmosphere geometry once (independent of frequency)
        normalized_height, normalized_radius_lower = self.atmosphere_height(
            temperature, mean_molecular_weight, radius_btm, gravity_btm
        )
        normalized_radius_top = normalized_radius_lower[0] + normalized_height[0]
        
        # Reshape dtau_ckd to 2D for chord calculations
        dtau_2d = dtau_ckd.reshape((nlayer, Ng * Nbands))
        
        # Compute chord optical depths
        cgm = chord_geometric_matrix_lower(normalized_height, normalized_radius_lower)
        dtau_chord_lower = chord_optical_depth(cgm, dtau_2d)
        
        func = self.integration_dict[self.integration]
        if self.integration == "trapezoid":
            transit_2d = func(
                dtau_chord_lower, normalized_radius_lower, normalized_radius_top
            )
        elif self.integration == "simpson":
            cgm_midpoint = chord_geometric_matrix(
                normalized_height, normalized_radius_lower
            )
            dtau_chord_midpoint = chord_optical_depth(cgm_midpoint, dtau_2d)
            transit_2d = func(
                dtau_chord_midpoint,
                dtau_chord_lower,
                normalized_radius_lower,
                normalized_height,
            )
        
        # Reshape back to (Ng, Nbands) and integrate over g-ordinates
        transit_ckd = transit_2d.reshape((Ng, Nbands))
        return jnp.einsum("n,nm->m", weights, transit_ckd)
