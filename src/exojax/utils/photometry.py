import numpy as np
from exojax.spec.unitconvert import wav2nu

def get_filter_from_svo(filter_name):
    #http://svo2.cab.inta-csic.es/theory/fps/
    from astroquery.svo_fps import SvoFps
    data = SvoFps.get_transmission_data(filter_name)
    unit = str(data['Wavelength'].unit)
    wl_ref = np.array(data['Wavelength'])
    nu_ref, transmission_ref = wav2nu(wl_ref, unit=unit, values=np.array(data['Transmission']))
    return nu_ref, transmission_ref

def average_resolution(nu_ref):
    nu_ref_min = np.min(nu_ref)
    nu_ref_max = np.max(nu_ref)
    dnu_ave = (nu_ref_max - nu_ref_min) / len(nu_ref)
    nuave = (nu_ref_max + nu_ref_min)/2.0
    return nuave/dnu_ave
