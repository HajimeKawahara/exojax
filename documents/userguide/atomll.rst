VALD3
--------------

*Dec 1 (2021) Hiroyuki T. Ishikawa* (need to update)


See ":doc:`../tutorials/metals`" (forward) and ":doc:`../tutorials/retvald`" (retrieval) as the tutorial to compute a cross section profile using VALD3.


Atomic Database
======================

`VALD3 <http://vald.astro.uu.se/>`_ provides atomic line database of various elements
from `some references <https://www.astro.uu.se/valdwiki/VALD3linelists>`_.
You need to manually acquire the dataset you need via the request form as below.
Note that the data access is free but requires registration through the `Contact form <http://vald.astro.uu.se/~vald/php/vald.php?docpage=contact.html>`_.

After the registration, you can login and select the "Extract All" or "Extract Element" mode.
If you want to cover all transitions, select "Extract All"; if you want to focus only on transitions of a given element (or neutral atoms or ionic species), select "Extract Element".

For example, if you want a linelist of neutral iron (Fe I), the request form in the "Extract Element" mode should be filled as:

- Starting wavelength :    1500
- Ending wavelength :    100000
- Element [ + ionization ] :    Fe 1
- Extraction format :    Long format
- Retrieve data via :    FTP
- Linelist configuration :    Default
- Unit selection:    Energy unit: eV - Medium: vacuum - Wavelength unit: angstrom - VdW syntax: default

If you want a linelist of all the species that VALD handles, you should fill in the request form in the "Extract All" mode, which have all the same fields other than "Element [ + ionization ]" in the above example.

You can download the requested data via the FTP URL on an email sent by VALD (Sometimes it takes a few hours to half a day to receive it).
Note that the number of spectral lines that can be extracted in a single request is limited to 1000 in VALD (`reference <https://www.astro.uu.se/valdwiki/Restrictions%20on%20extraction%20size>`_).
See https://www.astro.uu.se/valdwiki/presformat_output for the detail of the format.

An example to use the VALD3 database for a given species ("Extract Element") from exojax is like that:

.. code:: python

	>>> from exojax.database import moldb 
	>>> from exojax.utils.grids import wavenumber_grid
	>>> nus = 1e8/np.array([1e5, 1500.])
	>>> filepath_VALD3 = '.database/vald2600.gz'
	>>> adbFe = moldb.AdbVald(filepath_VALD3, nus)
	Reading VALD file

Another example to use the VALD3 database for multiple species ("Extract All") while separating the data by species is like that:

.. code:: python

	>>> from exojax.database import moldb 
	>>> from exojax.utils.grids import wavenumber_grid
	>>> nus, wav, res = wavenumber_grid(10398, 10406, 2000, unit="AA", xsmode="modit")
	>>> filepath = '.database/vald4214450.gz'
	>>> adb = moldb.AdbVald(filepath, nus, crit = 1e-100)
	>>> asdb = moldb.AdbSepVald(adb)
	Reading VALD file


Broadening Parameters
======================

The doppler broadening is calculated for example as:

.. code:: python

		>>> from exojax.database.hitran  import doppler_sigma
		>>> T = 3000 #temperature
		>>> Amol=np.float64( adbFe.atomicmass[0] ) #atomic mass
		>>> sigmaD = doppler_sigma(adbFe.nu_lines, T, Amol)

For the pressure width gamma (HWHM of Lorentzian (cm-1) caluculated as gamma/(4*pi*c) [cm-1]),
you can choose from five types of handling.

1. atomll.gamma_vald3
	Use the van der Waals gamma in the line list (VALD or Kurucz), otherwise estimated according to the `Unsoeld (1955) <https://ui.adsabs.harvard.edu/abs/1955psmb.book.....U>`_

2. atomll.gamma_uns
	Use the gamma estimated with the classical approximation by `Unsoeld (1955) <https://ui.adsabs.harvard.edu/abs/1955psmb.book.....U>`_

3. atomll.gamma_KA3
	Use the gamma caluculated with the 3rd equation in p.4 of `Kurucz & Avrett (1981) <https://ui.adsabs.harvard.edu/abs/1981SAOSR.391.....K>`_

4. atomll.gamma_KA4
	Use the gamma caluculated with the 4th equation in p.4 of `Kurucz & Avrett (1981) <https://ui.adsabs.harvard.edu/abs/1981SAOSR.391.....K>`_

5. atomll.gamma_KA3s
	Use the gamma caluculated with the 3rd equation in p.4 of `Kurucz & Avrett (1981) <https://ui.adsabs.harvard.edu/abs/1981SAOSR.391.....K>`_ but without discriminating iron group elements

The example is as:

.. code:: python

		>>> from exojax.opacity.lpf.lpf import auto_xsection, moldb, atomll
		>>> gammaL = atomll.gamma_vald3(T, PH, PHH, PHe, adbFe.ielem, adbFe.iion, \
		adbFe.dev_nu_lines, adbFe.elower, adbFe.eupper, adbFe.atomicmass, adbFe.ionE, \
		adbFe.gamRad, adbFe.gamSta, adbFe.vdWdamp, enh_damp=1.0)
