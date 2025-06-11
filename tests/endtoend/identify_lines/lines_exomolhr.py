from exojax.database.exomolhr import XdbExomolHR
from exojax.database.exomolhr import list_exomolhr_molecules
from exojax.database.exomolhr import list_isotopologues
from exojax.utils.grids import wavenumber_grid
import matplotlib.pyplot as plt

nus, _, _ = wavenumber_grid(22800.0, 23600.0, 10, xsmode="premodit", unit="AA")
temperature = 1300.0
molecules = list_exomolhr_molecules()
iso_dict = list_isotopologues(molecules)


lslist = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
lwlist = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]
markers_list = [".", "o", "s", "D", "^", "v", "<", ">"]
k = 0
for molecule in iso_dict:
    isos = iso_dict[molecule]
    print("##############################")
    print(isos)
    for j, iso in enumerate(isos):
        print(iso)
        try:
            xdb = XdbExomolHR(iso, nus, temperature)
            plt.plot(
                1.0e8 / xdb.nu_lines,
                xdb.line_strength,
                markers_list[j],
                label=iso,
                ls=lslist[j],
                lw=lwlist[j],
            )
        except:
            k = k + 1
            print(f"No line? {iso}")
print(k)
plt.yscale("log")
plt.xlabel("wavelength (AA)")
plt.ylabel("Line Strength (cm/molecule)")
plt.legend()
plt.savefig("exomolhr_lines.png")
plt.show()
