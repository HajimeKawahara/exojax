"""Faddeeva (wofz= w of z) functions (real and imag parts), an asymptotic form of wofz.

Note:
   We adopt ncut=27 as the summation of n in Algorithm 916. For Sigma1, we use ncut=8 because higher order than 9 does not affect the results.
"""

from jax import jit
from jax import custom_vjp
import jax.numpy as jnp
from jax.lax import scan
from exojax.special.erfcx import erfcx

an = jnp.array([0.5,  1.,  1.5,  2.,  2.5,  3.,  3.5,  4.,  4.5,  5., 5.5,  6.,
               6.5,  7.,  7.5,  8.,  8.5,  9.,  9.5, 10., 10.5, 11., 11.5, 12., 12.5, 13., 13.5])

a2n2 = jnp.array([0.25,   1.,   2.25,   4.,   6.25,   9.,  12.25, 16.,  20.25,  25.,  30.25,  36.,  42.25,
                 49., 56.25,  64.,  72.25,  81.,  90.25, 100., 110.25, 121., 132.25, 144., 156.25, 169., 182.25])


@jit
def rewofz(x, y):
    """Real part of wofz (Faddeeva) function based on Algorithm 916.

    We apply a=0.5 for Algorithm 916.

    Args:
        x: x < ncut/2
        y:

    Returns:
         jnp.array: Real(wofz(x+iy))
    """
    xy = x*y
    exx = jnp.exp(-x*x)
    f = exx*(erfcx(y)*jnp.cos(2.0*xy)+x*jnp.sin(xy)/jnp.pi*jnp.sinc(xy/jnp.pi))
    y2 = y*y
    Sigma23 = jnp.sum((jnp.exp(-(an+x)**2)+jnp.exp(-(an-x)**2))/(a2n2+y2))
    Sigma1 = exx*(7.78800786e-01/(0.25+y2)+3.67879450e-01/(1.+y2)+1.05399221e-01/(2.25+y2)+1.83156393e-02 /
                  (4.+y2)+1.93045416e-03/(6.25+y2)+1.23409802e-04/(9.+y2)+4.78511765e-06/(12.25+y2)+1.12535176e-07/(16.+y2))
    f = f + y/jnp.pi*(-jnp.cos(2.0*xy)*Sigma1 + 0.5*Sigma23)
    return f


@jit
def imwofz(x, y):
    """Imaginary part of wofz (Faddeeva) function based on Algorithm 916.

    We apply a=0.5 for Algorithm 916.

    Args:
        x: x < ncut/2
        y:

    Returns:
         jnp.array: Imag(wofz(x+iy))
    """
    wxy = 2.0*x*y
    exx = jnp.exp(-x*x)
    f = -exx*erfcx(y)*jnp.sin(wxy)+x/jnp.pi*exx*jnp.sinc(wxy/jnp.pi)
    y2 = y*y
    Sigma1 = exx*(7.78800786e-01/(0.25+y2)+3.67879450e-01/(1.+y2)+1.05399221e-01/(2.25+y2)+1.83156393e-02 /
                  (4.+y2)+1.93045416e-03/(6.25+y2)+1.23409802e-04/(9.+y2)+4.78511765e-06/(12.25+y2)+1.12535176e-07/(16.+y2))
    Sigma45 = jnp.sum(an*(-jnp.exp(-(an+x)**2)+jnp.exp(-(an-x)**2))/(a2n2+y2))
    f = f + 1.0/jnp.pi*(y*jnp.sin(wxy)*Sigma1 + 0.5*Sigma45)

    return f


@jit
def wofzs2(x, y):
    """Asymptotic representation of wofz (Faddeeva) function 1 for |z|**2 > 112 (for e = 10e-6)

    See Zaghloul (2018) arxiv:1806.01656

    Args:
        x: 
        y:

    Returns:
         jnp.array, jnp.array: H=real(wofz(x+iy)),L=imag(wofz(x+iy))

    """

    z = x+y*(1j)
    a = 1.0/(2.0*z*z)
    q = (1j)/(z*jnp.sqrt(jnp.pi))*(1.0 + a*(1.0 + a*(3.0 + a*15.0)))
    return jnp.real(q), jnp.imag(q)


@jit
def rewofzs2(x, y):
    """Real part of Asymptotic representation of wofz (Faddeeva) function 1 for |z|**2 > 112 (for e = 10e-6)

    See Zaghloul (2018) arxiv:1806.01656

    Args:
        x: 
        y:

    Returns:
         jnp.array: real(wofz(x+iy))

    """

    z = x+y*(1j)
    a = 1.0/(2.0*z*z)
    q = (1j)/(z*jnp.sqrt(jnp.pi))*(1.0 + a*(1.0 + a*(3.0 + a*15.0)))
    return jnp.real(q)


@jit
def imwofzs2(x, y):
    """Imag part of Asymptotic representation of wofz (Faddeeva) function 1 for |z|**2 > 112 (for e = 10e-6)

    See Zaghloul (2018) arxiv:1806.01656

    Args:
        x: 
        y:

    Returns:
         jnp.array: imag(wofz(x+iy))

    """

    z = x+y*(1j)
    a = 1.0/(2.0*z*z)
    q = (1j)/(z*jnp.sqrt(jnp.pi))*(1.0 + a*(1.0 + a*(3.0 + a*15.0)))
    return jnp.imag(q)


@custom_vjp
def rewofzx(x, y):
    """[VJP custom defined] Real part of wofz (Faddeeva) function based on
    Algorithm 916.

    We apply a=0.5 for Algorithm 916.

    Args:
        x: x < ncut/2
        y:

    Returns:
        jnp.array: Real(wofz(x+iy))
    """
    xy = x*y
    exx = jnp.exp(-x*x)
    f = exx*erfcx(y)*jnp.cos(2.0*xy)+x*jnp.sin(xy) / \
        jnp.pi*exx*jnp.sinc(xy/jnp.pi)
    y2 = y*y
    Sigma1 = exx*(7.78800786e-01/(0.25+y2)+3.67879450e-01/(1.+y2)+1.05399221e-01/(2.25+y2)+1.83156393e-02 /
                  (4.+y2)+1.93045416e-03/(6.25+y2)+1.23409802e-04/(9.+y2)+4.78511765e-06/(12.25+y2)+1.12535176e-07/(16.+y2))
    Sigma23 = jnp.sum((jnp.exp(-(an+x)**2)+jnp.exp(-(an-x)**2))/(a2n2+y*y))

    f = f + y/jnp.pi*(-jnp.cos(2.0*xy)*Sigma1 + 0.5*Sigma23)
    return f


def h_fwd(x, y):
    hh = rewofzx(x, y)
    return hh, (hh, imwofz(x, y), x, y)


def h_bwd(res, g):
    """backward.

    Note:
        V=Real(wofz), L=Imag(wofz)

    Args:
        res: res  from h_fwd
        g: g

    Returns:
           jnp.array, jnp.array: g* partial_x h(x,y), g* partial_y h(x,y)
    """
    V, L, x, y = res
    return (2.0 * (y*L - x*V) * g, 2.0 * (x*L + y*V) * g - 2.0/jnp.sqrt(jnp.pi))


rewofzx.defvjp(h_fwd, h_bwd)
