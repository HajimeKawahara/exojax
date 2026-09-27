# H2-H2 CIA for the CO sensitivity tutorial

`H2-H2_2011_4320-4370.cia` is a spectral subset of the public
[HITRAN H2-H2 CIA file](https://hitran.org/data/CIA/H2-H2_2011.cia).
It retains all 113 native temperatures from 200 to 3000 K at 25 K intervals,
and all 51 native wavenumbers from 4320 to 4370 cm^-1 at 1 cm^-1 intervals.
The absorption coefficients and their precision are unchanged: no interpolation
or downsampling was used to create this file. Only each block's wavenumber bounds,
point count, and maximum absorption coefficient were updated in its header.
Other header metadata, including reference code 6, are preserved.

The source calculation is M. Abel, L. Frommhold, X. Li, and K. L. C. Hunt,
"Collision-Induced Absorption by H2 Pairs: From Hundreds to Thousands of Kelvin,"
*J. Phys. Chem. A* **115**, 6805-6812 (2011),
[doi:10.1021/jp109441f](https://doi.org/10.1021/jp109441f).
See the [HITRAN CIA documentation](https://hitran.org/cia/) and its
[reference list](https://hitran.org/data/CIA/CIA_References_2024.pdf).
The current database reference is I. E. Gordon et al., "The HITRAN2024 molecular
spectroscopic database," *J. Quant. Spectrosc. Radiat. Transfer* **353**, 109807
(2026); see the [HITRAN citation policy](https://hitran.org/citepolicy/).

SHA-256 checksums:

```text
14f56ff889a474402136be721fcc286921a1b47ce5f7c3dff90af2d4c74b20e5  H2-H2_2011.cia
1128711d94c75889a79c38241f038a3bf2ef851bba689b6614d935a59609b601  H2-H2_2011_4320-4370.cia
```

The first checksum identifies the local copy of the full upstream file used for
extraction. To reproduce the subset, download that file to
`.database/H2-H2_2011.cia` and run the following standard-library Python code from
the repository root:

```python
from pathlib import Path

source = Path(".database/H2-H2_2011.cia")
target = Path("examples/spectral_sensitivity_data/H2-H2_2011_4320-4370.cia")
with source.open() as source_file, target.open("w") as target_file:
    for header in source_file:
        rows = [source_file.readline() for _ in range(int(header[40:47]))]
        selected = [row for row in rows if 4320 <= float(row.split()[0]) <= 4370]
        maximum = max(float(row.split()[1]) for row in selected)
        target_file.write(
            header[:20] + f"{4320.:10.3f}{4370.:10.3f}{len(selected):7d}"
            + header[47:54] + f"{maximum:10.3E}" + header[64:]
        )
        target_file.writelines(selected)
```
