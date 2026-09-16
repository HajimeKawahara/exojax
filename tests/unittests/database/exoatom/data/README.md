# ExoAtom test subsets

Small subsets downloaded from the official ExoAtom database on 2026-09-16.
The NIST and Kurucz source labels and physical data are retained separately.
These files are regression fixtures, not complete spectroscopic line lists.

States and transitions retain original rows and IDs. Only states referenced by
the selected transitions are kept. Definition row counts and maximum state energy
and wavenumber are updated to describe the subset; other metadata is retained,
including source inconsistencies in optional NIST g-factor declarations.
Partition files retain endpoints and the original rows bracketing 296, 1000,
3000, 5000, and 6000 K. Their interpolation is checked only at these temperatures.

Li NIST includes the resonance doublet whose supplied second wavenumber differs
from the rounded state-energy difference. Li Kurucz includes infinite ground-state
lifetime and an excited-to-excited transition. Li+ includes an excited lower level
whose reference strength underflows at 296 K. The hydrogen fixture checks the
element/isotope/dataset path and isotope metadata.

Reference: [ExoAtom paper](https://arxiv.org/abs/2512.24612).

The SHA-256 values below identify the complete downloaded source files, before
subsetting or metadata count updates.

## Li/NIST

Selected transitions (upper, lower): `[(2, 1), (3, 1), (6, 1), (15, 13)]`.

| Source | SHA-256 |
| --- | --- |
| [Li__NIST.adef.json](https://www.exomol.com/exoatom/db/Li/NIST/Li__NIST.adef.json) | `891f10be796744f1530b8af13ec815b420add9edc1332fe470b4a1ff52329482` |
| [Li__NIST.states](https://www.exomol.com/exoatom/db/Li/NIST/Li__NIST.states) | `0b7b3fd9324dce61ecf8af321ee217263bfefd869cce91f7dbc0efaaf349a9be` |
| [Li__NIST.trans](https://www.exomol.com/exoatom/db/Li/NIST/Li__NIST.trans) | `207871042ed5a7b0accfefa8bfbc2b9b6b34a7fb433caea224e150cfb907b225` |
| [Li__NIST.pf](https://www.exomol.com/exoatom/db/Li/NIST/Li__NIST.pf) | `2180ddbf58707b0c2be531caee23109b492a6ad3d7ce2c67e8d3f20d8ed1f5e3` |

## Li/Kurucz

Selected transitions (upper, lower): `[(2, 1), (3, 1), (5, 1), (58, 54)]`.

| Source | SHA-256 |
| --- | --- |
| [Li__Kurucz.adef.json](https://www.exomol.com/exoatom/db/Li/Kurucz/Li__Kurucz.adef.json) | `b31a5d0891cd855d2b36bc833d17a8435d145cb1b5aec9d568d72f4484084037` |
| [Li__Kurucz.states](https://www.exomol.com/exoatom/db/Li/Kurucz/Li__Kurucz.states) | `62c9a7d14b36066ecd82b9daa339dd79bf610ed09dcc4bef713cf4afee318878` |
| [Li__Kurucz.trans](https://www.exomol.com/exoatom/db/Li/Kurucz/Li__Kurucz.trans) | `a4031b1a1372ff2a9fbf01c9aa5ca34847257b86faef411211d1aa73446fa969` |
| [Li__Kurucz.pf](https://www.exomol.com/exoatom/db/Li/Kurucz/Li__Kurucz.pf) | `695ba38fcc3783a6336d4cfe2d1b1e7656132c718262ae58c42cbed27b5fd184` |

## Li_p/NIST

Selected transitions (upper, lower): `[(2, 1), (7, 1), (49, 40)]`.

| Source | SHA-256 |
| --- | --- |
| [Li_p__NIST.adef.json](https://www.exomol.com/exoatom/db/Li_p/NIST/Li_p__NIST.adef.json) | `c54f2618358b7e73c0cd4cbdb4067c11be38bdf7832fa8c829019f20b8e56581` |
| [Li_p__NIST.states](https://www.exomol.com/exoatom/db/Li_p/NIST/Li_p__NIST.states) | `133d1c845bfbace050640ab1bc9b3d87e551f559766f33dd1f40a9970c3264a0` |
| [Li_p__NIST.trans](https://www.exomol.com/exoatom/db/Li_p/NIST/Li_p__NIST.trans) | `be47b5f1904ae03bde075ce6e819666928484f9bd520e278c6bb66941973fc53` |
| [Li_p__NIST.pf](https://www.exomol.com/exoatom/db/Li_p/NIST/Li_p__NIST.pf) | `45a971c48f6fb50913e32e9d2fee903d39aeb78b4fb79946ddcb5314ce755e04` |

## H/1H/NIST

Selected transitions (upper, lower): `[(4, 1), (5, 1), (27, 24)]`.

| Source | SHA-256 |
| --- | --- |
| [1H__NIST.adef.json](https://www.exomol.com/exoatom/db/H/1H/NIST/1H__NIST.adef.json) | `7340d41b01c752df07969cf313580a0e18878c7674ca2936e98cb5fb92635340` |
| [1H__NIST.states](https://www.exomol.com/exoatom/db/H/1H/NIST/1H__NIST.states) | `52acdc484c359c102e3a6596aefb02e56f7ea40e899749ab56a542d540d18b51` |
| [1H__NIST.trans](https://www.exomol.com/exoatom/db/H/1H/NIST/1H__NIST.trans) | `003c9a1f60e37df528fd42c5ec1723c9111b4312d364c7583098119bbc1bc78d` |
| [1H__NIST.pf](https://www.exomol.com/exoatom/db/H/1H/NIST/1H__NIST.pf) | `6c5e3f20beb5480dc9b9e6912c52272c19bf93ef74969cc1d04f8808eb67ea0a` |
