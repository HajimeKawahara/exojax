# %%
from exojax.database.hitran.api import MdbHitran
from exojax.utils.grids import wavenumber_grid

nus, wav, r = wavenumber_grid(24000.0, 26000.0, 1000, unit="AA", xsmode="premodit")

# when
mdb = MdbHitran("CO", nus, inherit_dataframe=True, nonair_broadening=True)

# %%

# check pressure shift
import numpy as np
from exojax.utils.constants import ccgs
df_mask = mdb.df[mdb.df_load_mask]
dnu = df_mask["delta_h2"].values/mdb.nu_lines
maxdv = np.max(dnu * ccgs*1.e-5)
print("maximum velocity shift by nonair shift = ", maxdv, "km/s")

# %%
# manual non-air broadening
from exojax.database.molinfo  import m_transition_state
from exojax.database.nonair  import gamma_nonair, temperature_exponent_nonair
from exojax.database.nonair  import nonair_coeff_CO_in_H2

df_mask = mdb.df[mdb.df_load_mask]
m = m_transition_state(df_mask["jl"],df_mask["branch"])
n_Texp_H2 = temperature_exponent_nonair(m, nonair_coeff_CO_in_H2)
gamma_ref_H2 = gamma_nonair(m, nonair_coeff_CO_in_H2)

# %%
import matplotlib.pyplot as plt
# %%
plt.plot(mdb.n_h2, n_Texp_H2.values, ".")
plt.plot([0.4,0.7],[0.4,0.7])
plt.xlabel("temperature exponent given by radis")
plt.ylabel("temperature exponent given by ExoJAX")
plt.savefig("comp_nonair_1.png")
# %%
plt.plot(mdb.gamma_h2, gamma_ref_H2.values, ".")
plt.plot([0.065,0.078],[0.065,0.078])
plt.xlabel("gamma given by radis")
plt.ylabel("gamma given by ExoJAX")
plt.savefig("comp_nonair_2.png")

# %%
