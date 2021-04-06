"""special functions in exojax

   exojax.scipy.special  -- special functions

"""

from jax import jit
from jax import custom_vjp
import jax.numpy as jnp
from jax.lax import scan
#from jax.interpreters.ad import defvjp
from exojax.special.erfcx import erfcx

@jit
def rewofz(x,y):
    """Real part of wofz function based on Algorithm 916
    
    We apply a=0.5 for Algorithm 916.
    
    Args:
        x: x < ncut/2 
        y:
        
    Returns:
         jnp.array: Real(wofz(x+iy))

    """
    ncut=27
    xy=x*y
    xyp=xy/jnp.pi
    exx=jnp.exp(-x*x)
    f=exx*erfcx(y)*jnp.cos(2.0*xy)+x*jnp.sin(xy)/jnp.pi*exx*jnp.sinc(xyp)
    an=0.5*jnp.arange(1,ncut+1)
    a2n2=an*an
    vec0=1.0/(a2n2+ y*y)
    vec1=jnp.exp(-(a2n2+x*x))
    vec2=jnp.exp(-(an+x)**2)
    vec3=jnp.exp(-(an-x)**2)
    Sigma1=jnp.dot(vec0,vec1)
    Sigma2=jnp.dot(vec0,vec2)
    Sigma3=jnp.dot(vec0,vec3)
    f = f + 1.0/jnp.pi*(-y*jnp.cos(2.0*xy)*Sigma1 + 0.5*y*Sigma2 + 0.5*y*Sigma3)
    return f


@jit
def imwofz(x,y):
    """Imaginary part of wofz function based on Algorithm 916
    
    We apply a=0.5 for Algorithm 916.
    
    Args:
        x: x < ncut/2 
        y:
        
    Returns:
         jnp.array: Imag(wofz(x+iy))

    """
    ncut=27
    xy=x*y                             
    xyp=2.0*xy/jnp.pi                      
    exx=jnp.exp(-x*x)                  
    f=-exx*erfcx(y)*jnp.sin(2.0*xy)+x/jnp.pi*exx*jnp.sinc(xyp)           
    an=0.5*jnp.arange(1,ncut+1)             
    a2n2=an*an                             
    vec0=1.0/(a2n2+ y*y)
    vec1=jnp.exp(-(a2n2+x*x))   
    Sigma1=jnp.dot(vec0,vec1)

    vecm=an*vec0
    vec4=jnp.exp(-(an+x)*(an+x)) 
    vec5=jnp.exp(-(an-x)*(an-x)) 
    Sigma4=jnp.dot(vecm,vec4)
    Sigma5=jnp.dot(vecm,vec5)
    f = f + 1.0/jnp.pi*(y*jnp.sin(2.0*xy)*Sigma1 + 0.5*Sigma5 -0.5*Sigma4)
    return f


@jit
def wofzs2(x,y):
    """Asymptotic representation of wofz function 1 for |z|**2 > 112 (for e = 10e-6)

    See Zaghloul (2018) arxiv:1806.01656
    
    Args:
        x: 
        y:
        
    Returns:
         jnp.array, jnp.array: H=real(wofz(x+iy)),L=imag(wofz(x+iy))

    """

    z=x+y*(1j)
    a=1.0/(2.0*z*z)
    q=(1j)/(z*jnp.sqrt(jnp.pi))*(1.0 + a*(1.0 + a*(3.0 + a*15.0)))
    return jnp.real(q),jnp.imag(q)

@jit
def rewofzs2(x,y):
    """Real part of Asymptotic representation of wofz function 1 for |z|**2 > 112 (for e = 10e-6)

    See Zaghloul (2018) arxiv:1806.01656
    
    Args:
        x: 
        y:
        
    Returns:
         jnp.array: real(wofz(x+iy))

    """

    z=x+y*(1j)
    a=1.0/(2.0*z*z)
    q=(1j)/(z*jnp.sqrt(jnp.pi))*(1.0 + a*(1.0 + a*(3.0 + a*15.0)))
    return jnp.real(q)


@jit
def imwofzs2(x,y):
    """Imag part of Asymptotic representation of wofz function 1 for |z|**2 > 112 (for e = 10e-6)

    See Zaghloul (2018) arxiv:1806.01656
    
    Args:
        x: 
        y:
        
    Returns:
         jnp.array: imag(wofz(x+iy))

    """

    z=x+y*(1j)
    a=1.0/(2.0*z*z)
    q=(1j)/(z*jnp.sqrt(jnp.pi))*(1.0 + a*(1.0 + a*(3.0 + a*15.0)))
    return jnp.imag(q)



@custom_vjp
def rewofzx(x, y):
    """[VJP custom defined] Real part of wofz function based on Algorithm 916
    
    We apply a=0.5 for Algorithm 916.
    
    Args:
        x: x < ncut/2 
        y:
        
    Returns:
        jnp.array: Real(wofz(x+iy))

    """
    ncut=27           
    xy=x*y
    xyp=xy/jnp.pi       
    exx=jnp.exp(-x*x)   
    f=exx*erfcx(y)*jnp.cos(2.0*xy)+x*jnp.sin(xy)/jnp.pi*exx*jnp.sinc(xyp)    
    n=jnp.arange(1,ncut+1)      
    n2=n*n
    vec0=1.0/(0.25*n2+ y*y)     
    vec1=jnp.exp(-(0.25*n2+x*x))
    vec2=jnp.exp(-(0.5*n+x)*(0.5*n+x))        
    vec3=jnp.exp(-(0.5*n-x)*(0.5*n-x))        
    Sigma1=jnp.dot(vec0,vec1)
    Sigma2=jnp.dot(vec0,vec2)
    Sigma3=jnp.dot(vec0,vec3)
    f = f + 1.0/jnp.pi*(-y*jnp.cos(2.0*xy)*Sigma1 + 0.5*y*Sigma2 + 0.5*y*Sigma3)
    return f

def h_fwd(x, y):
    hh=rewofzx(x, y)
    return hh, (hh, imwofz(x, y), x, y)

def h_bwd(res, g):
    """backward

    Note: 
        V=Real(wofz), L=Imag(wofz)

    Args:
        res: res  from h_fwd
        g: g

    Returns:
           jnp.array, jnp.array: g* partial_x h(x,y), g* partial_y h(x,y)


    """
    V, L, x, y = res 
    return (2.0 * (y*L - x*V) * g , 2.0 * (x*L + y*V) * g - 2.0/jnp.sqrt(jnp.pi))

rewofzx.defvjp(h_fwd, h_bwd)
