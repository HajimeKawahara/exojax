# %%
from exojax.spec import api
from exojax.utils.grids import wavenumber_grid
nus,wav,r = wavenumber_grid(22900.0,24000.0,1000,unit="AA")
#emf='CO/12C-16O/Li2015'   #lifetime=0, Lande=0
#emf="NaH/23Na-1H/Rivlin/" #lifetime=1, Lande=0
#emf="MgH/24Mg-1H/XAB/" #lifetime=1, Lande=1
#emf="FeH/56Fe-1H/MoLLIST/"
#emf="OH/16O-1H/MoLLIST/"
#emf="CO2/12C-16O2/UCL-4000/"
emf="CO/12C-16O/Li2015/"
#emf="CaOH/40Ca-16O-1H/OYT6/"
#emf="H3O_p/1H3-16O_p/eXeL/"
#emf="H2/1H2/RACPPK/"
#emf="CH4/12C-1H4/YT34to10/"
#emf="H2O/1H2-16O/POKAZATEL/"
#emf="HCl/1H-35Cl/HITRAN-HCl/"
#emf="NO/14N-16O/XABC/"
mdb = api.MdbExomol(emf,nus,inherit_dataframe=True, skip_optional_data=False)

# %%
print(mdb.df)
# Current column
# i_upper	i_lower	A	nu_lines	gup	jlower	jupper	elower	Si
# %%
import matplotlib.pyplot as plt
for dv in range(0,6):
    mask = mdb.df["v_u"]-mdb.df["v_l"] == dv
    dfv = mdb.df[mask]
    plt.plot(dfv["nu_lines"].values,dfv["Sij0"].values,".", label="$\\Delta \\nu = $"+str(dv), alpha=0.2)
plt.legend()
plt.title("CO Li2015")
plt.ylabel("line strength at 296 K")
plt.xlabel("wavenumber cm-1")
plt.yscale("log")
plt.show()
