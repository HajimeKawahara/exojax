from exojax.database.exomol.api import MdbExomol 
from exojax.utils.grids import wavenumber_grid

nus, wav, r = wavenumber_grid(24000.0, 26000.0, 1000, unit="AA", xsmode="premodit")

# when
mdb = MdbHitemp("CO", nus, activation=False)

print(mdb.df)
import matplotlib.pyplot as plt
for dv in range(0, 6):
    mask = mdb.df["vu"] - mdb.df["vl"] == dv
    dfv = mdb.df[mask]
    plt.plot(1.e4 / dfv["wav"].values,
             dfv["int"].values,
             ".",
             label="$\\Delta \\nu = $" + str(dv),
             alpha=0.2)
#plt.show()

load_mask = (mdb.df["vu"] - mdb.df["vl"] == 3)
mdb.activate(mdb.df, load_mask)


plt.plot(1.e4 / mdb.nu_lines,
         mdb.line_strength_ref_original,
         "+",
         color="black",
         label="activated lines")
plt.legend()
#plt.title(emf)
plt.xlim(2.0,3.0)
plt.ylabel("line strength at 296 K")
plt.xlabel("micron")
plt.yscale("log")
#plt.xscale("log")
plt.show()

# %%
