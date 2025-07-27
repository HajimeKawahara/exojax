# data

- src/exojax/data/atom/atomic.txt
  std1999c, Asplund et al. 2009 only for abundance
  
- src/exojax/data/atom/barklem_collet_2016_pff.txt

- src/exojax/data/atom/iso_mn.txt
  mass number, Taken from https://www.chem.ualberta.ca/~massspec/atomic_mass_abund.pdf The isotopic mass data is from G. Audi, A. H. Wapstra Nucl. Phys A. 1993, 565, 1-65 and G. Audi, A. H. Wapstra Nucl. Phys A. 1995, 595, 409-480.  The percent natural abundance data is from the 1997 report of the IUPAC Subcommittee for Isotopic Abundance Measurements by K.J.R. Rosman, P.D.P. Taylor Pure Appl. Chem. 1999, 71, 1593-1607.

- src/exojax/data/opacity/FeH_Hargreaves2010.csv
  Line list of the E ^4^{Pi} - A ^4^{Pi} transition of FeH, taken from Table 5 of Hargreaves et al., 2010, AJ, 140, 919 https://content.cld.iop.org/journals/1538-3881/140/4/919/revision1/aj357217t5_mrt.txt
  (paper: https://iopscience.iop.org/article/10.1088/0004-6256/140/4/919)

It can be loaded by utils.isodata.read_mnlist.

- src/exojax/data/clouds/drag_force.txt
  Reynolds number vs Drag force coefficient, Table 10.1 p381, Pruppacher and Klett 

- src/exojax/data/abundance/AAG2021.dat
  Abundunce (log 10) taken from Asplund+ (2021) https://arxiv.org/abs/2105.01661 (v2)