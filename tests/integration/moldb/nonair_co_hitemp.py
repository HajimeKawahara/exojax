# %%
from exojax.database.exomol.api import MdbExomol 
from exojax.utils.grids import wavenumber_grid

nus, wav, r = wavenumber_grid(24000.0, 26000.0, 1000, unit="AA", xsmode="premodit")
mdb = MdbHitemp("CO", nus, inherit_dataframe=True)

# %%
# manual non-air broadening
from exojax.database.molinfo  import m_transition_state
from exojax.database.nonair  import gamma_nonair, temperature_exponent_nonair
from exojax.database.nonair  import nonair_coeff_CO_in_H2

df_mask = mdb.df[mdb.df_load_mask]
m = m_transition_state(df_mask["jl"],df_mask["branch"])
n_Texp_H2 = temperature_exponent_nonair(m, nonair_coeff_CO_in_H2).values
gamma_ref_H2 = gamma_nonair(m, nonair_coeff_CO_in_H2).values

# %%
import matplotlib.pyplot as plt
plt.plot(n_Texp_H2, gamma_ref_H2, ".")
plt.title("CO broadening in H2 atmosphere")
plt.xlabel("temperature exponent given by ExoJAX")
plt.ylabel("gamma cm-1")
plt.savefig("hitemp_co_noair_h2.png")

# %%
plt.plot(n_Texp_H2, mdb.n_air, ".")
plt.xlabel("temperature exponent (H2)")
plt.ylabel("temperature exponent (air)")
plt.plot([0.2,0.85],[0.2,0.85])
plt.savefig("comph2_1.png")

# %%
plt.plot(gamma_ref_H2, mdb.gamma_air, ".")
plt.xlabel("gamma cm-1 (H2)")
plt.ylabel("gamma cm-1 (air)")
plt.plot([0.05,0.085],[0.05,0.085])
plt.savefig("comph2_2.png")


# %%
