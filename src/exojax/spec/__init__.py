__all__ = []

__version__ = '1.2'
__uri__ = ''
__author__ = 'ExoJAX contributors'
__email__ = 'divrot@gmail.com'
__license__ = ''
__description__ = 'auto-differentiable spectral modules in exojax'

from exojax.spec.hitran import (
    line_strength,
    line_strength,
    doppler_sigma,
    gamma_natural,
    normalized_doppler_sigma
)


from exojax.spec.lpf import (
    hjert,
    voigt,
    voigtone,
    vvoigt,
)


from exojax.spec.make_numatrix import (
    make_numatrix0,
)
