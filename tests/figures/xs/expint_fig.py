from jax import config                                                 #
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expn
from exojax.special.expn import E1
from exojax.rt import rtransfer as rt
import jax.numpy as jnp

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expn
from exojax.special.expn import E1
from exojax.rt import rtransfer as rt
import jax.numpy as jnp
x = np.logspace(-4, 1.9, 1000)

d_f32 = np.abs(rt.trans2E3(x)-(2.0*expn(3, x)))

config.update('jax_enable_x64', True)


fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(211)
plt.plot(x, 2.0*expn(3, x), label='$\mathcal{T}(x)$ by scipy.special.expn')
plt.plot(x, rt.trans2E3(x), ls='dashed',
         label='$\mathcal{T}(x)$ by ours (AS70 w/ jax.numpy)')
plt.ylabel('$\mathcal{T}(x)$', fontsize=14)
plt.tick_params(labelsize=13)

plt.plot(x, 1-2.0*expn(3, x), label='$1-\mathcal{T}(x)$ by scipy.special.expn')
plt.plot(x, 1-rt.trans2E3(x), ls='dotted',
         label='$1-\mathcal{T}(x)$ by ours (AS70 w/ jax.numpy)')
plt.ylim(1.e-8, 1.e1)
plt.ylabel('$\mathcal{T}(x),1-\mathcal{T}(x)$', fontsize=14)
plt.tick_params(labelsize=13)
plt.legend()
plt.xscale('log')
plt.yscale('log')

ax = fig.add_subplot(212)
plt.plot(x, d_f32, '.', alpha=0.5, color='C1', label='F32')
plt.plot(x, np.abs(rt.trans2E3(x)-(2.0*expn(3, x))),
         '+', alpha=0.5, color='C2', label='F64')
plt.legend(loc='center right')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('|ours-scipy| ', fontsize=14)
plt.xlabel('x', fontsize=14)
plt.ylim(1.e-12, 1.e-6)
plt.axhline(2.e-7, ls='dashed')
plt.tick_params(labelsize=13)

plt.savefig('E3.pdf', bbox_inches='tight', pad_inches=0.0)
plt.savefig('E3.png', bbox_inches='tight', pad_inches=0.0)
