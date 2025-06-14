import jax.numpy as jnp
import numpy as np


def _setdgm(mdb, dit_grid_resolution, R, Tarr_list, Parr, Pself_ref=None):
        """_summary_

        Args:
            Tarr_list (1d or 2d array): tempearture array to be tested such as [Tarr_1, Tarr_2, ..., Tarr_n]
            Parr (1d array): pressure array in bar
            Pself_ref (1d array, optional): self pressure array in bar. Defaults to None. If None Pself = 0.0.

        Returns:
            _type_: dgm (DIT grid matrix) for gammaL
        """
        from exojax.opacity.modit.modit import exomol, hitran
        from exojax.opacity._common.set_ditgrid import (
            minmax_ditgrid_matrix,
            precompute_modit_ditgrid_matrix,
        )

        #cont_nu, index_nu, R, pmarray = self.opainfo
        if len(np.shape(Tarr_list)) == 1:
            Tarr_list = np.array([Tarr_list])
        if Pself_ref is None:
            Pself_ref = np.zeros_like(Parr)

        set_dgm_minmax = []
        for Tarr in Tarr_list:
            if mdb.dbtype == "exomol":
                _, ngammaLM, _ = exomol(
                    mdb, Tarr, Parr, R, mdb.molmass
                )
            elif mdb.dbtype == "hitran":
                _, ngammaLM, _ = hitran(
                    mdb, Tarr, Parr, Pself_ref, R, mdb.molmass
                )
            set_dgm_minmax.append(
                minmax_ditgrid_matrix(ngammaLM, dit_grid_resolution)
            )
        dgm_ngammaL = precompute_modit_ditgrid_matrix(
            set_dgm_minmax, dit_grid_resolution=dit_grid_resolution
        )
        #self.dgm_ngammaL = jnp.array(dgm_ngammaL)
        return jnp.array(dgm_ngammaL)