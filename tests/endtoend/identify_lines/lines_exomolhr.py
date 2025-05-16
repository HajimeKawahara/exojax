from exojax.spec.exomolhr import XdbExomolHR
from exojax.spec.exomolhr import list_exomolhr_molecules
from exojax.utils.grids import wavenumber_grid
import matplotlib.pyplot as plt

nus, _, _ = wavenumber_grid(22800.0,23600.0, 10, xsmode="premodit")
temperature = 1300.0
molecules = list_exomolhr_molecules()
for molecule in molecules:
    print(molecule)
    try:
        mdb = XdbExomolHR(molecule, nus, temperature)
        plt.plot(mdb.nu_lines, mdb.line_strength, label=exact_molecule)
    except:
        print(f"Failed to load {molecule}")

plt.xlabel("Wavenumber (cm-1)")
plt.ylabel("Line Strength (cm/molecule)")
plt.legend()
plt.savefig("exomolhr_lines.png")
plt.show()

