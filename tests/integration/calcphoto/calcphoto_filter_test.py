from exojax.postproc.specop import SopPhoto    
import matplotlib.pyplot as plt

# http://svo2.cab.inta-csic.es/theory/fps/
#filter_name = "2MASS/2MASS.Ks"
filter_name = "SLOAN/SDSS.g"
sop_photo = SopPhoto(filter_name, download=True)

# Sun
from exojax.rt.planck import piB
from exojax.utils.constants import RJ, Rs
from exojax.utils.constants import pc

flux = piB(5772.0, sop_photo.nu_grid_filter) * (Rs/RJ) ** 2 / (10.0) ** 2 * (RJ / pc)**2

mag = sop_photo.apparent_magnitude(flux)
print(mag)

plt.plot(sop_photo.nu_ref, sop_photo.transmission_ref)
plt.savefig("transmission.png")
#plt.show()


