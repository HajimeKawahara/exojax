"""_special functions in exojax

   exojax.scipy._special  -- under score special functions

"""

from jax import jit
from jax import custom_vjp
import jax.numpy as jnp
from jax.lax import scan
from jax.interpreters.ad import defvjp

@jit
def rewofz_vectorized(x, y):
    """Real part of wofz function based on Algorithm 916
    
    We apply a=0.5 for Algorithm 916.
    
    Args:
        x: x < ncut/2 
        y:
        
    Returns:
        f: Real(wofz(x+iy))

    """
    ncut=27
    xy=x*y
    xyp=xy/jnp.pi
    exx=jnp.exp(-x*x)
    f=exx*erfcx(y)*jnp.cos(2.0*xy)+x*jnp.sin(xy)/jnp.pi*exx*jnp.sinc(xyp)
    n=jnp.arange(1,ncut+1)
    n2=n*n
    x2=x*x
    y2=y*y
    vec0=1.0/(0.25*n2+y2)
    vec1=jnp.exp(-(0.25*n2[None,:]+x2[:,None]))
    vec2=jnp.exp(-(0.5*n[None,:]+x[:,None])*(0.5*n[None,:]+x[:,None]))
    vec3=jnp.exp(-(0.5*n[None,:]-x[:,None])*(0.5*n[None,:]-x[:,None]))
    Sigma1=jnp.sum(vec0*vec1,axis=1)
    Sigma2=jnp.sum(vec0*vec2,axis=1)
    Sigma3=jnp.sum(vec0*vec3,axis=1)
    
    f = f + 1.0/jnp.pi*(-y*jnp.cos(2.0*xy)*Sigma1 + 0.5*y*Sigma2 + 0.5*y*Sigma3)
    return f

@jit
def imwofz_vectorized(x, y):
    """Imaginary part of wofz function based on Algorithm 916
    
    We apply a=0.5 for Algorithm 916.
    
    Args:
        x: x < ncut/2 
        y:
        
    Returns:
        f: Imag(wofz(x+iy))

    """
    ncut=27
    xy=x*y
    xyp=2.0*xy/jnp.pi
    exx=jnp.exp(-x*x)
    f=-exx*erfcx(y)*jnp.sin(2.0*xy)+x/jnp.pi*exx*jnp.sinc(xyp)
    n=jnp.arange(1,ncut+1)
    n2=n*n
    x2=x*x
    y2=y*y
    vec0=0.5*n/(0.25*n2+y2)
    vec1=jnp.exp(-(0.25*n2[None,:]+x2[:,None]))
    vec4=jnp.exp(-(0.5*n[None,:]+x[:,None])*(0.5*n[None,:]+x[:,None]))
    vec5=jnp.exp(-(0.5*n[None,:]-x[:,None])*(0.5*n[None,:]-x[:,None]))
    Sigma1=jnp.sum(vec0*vec1,axis=1)
    Sigma4=jnp.sum(vec0*vec4,axis=1)
    Sigma5=jnp.sum(vec0*vec5,axis=1)
    f = f + 1.0/jnp.pi*(y*jnp.sin(2.0*xy)*Sigma1 + 0.5*(Sigma5-Sigma4))

    return f


@jit
def rewofz_nonvector(x,y):
    """Real part of wofz function based on Algorithm 916
    
    We apply a=0.5 for Algorithm 916.
    
    Args:
        x: x < ncut/2 
        y:
        
    Returns:
        f: Real(wofz(x+iy))

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


@jit
def imwofz_nonvector(x, y):
    """Imaginary part of wofz function based on Algorithm 916
    
    We apply a=0.5 for Algorithm 916.
    
    Args:
        x: x < ncut/2 
        y:
        
    Returns:
        f: Imag(wofz(x+iy))

    """
    ncut=27
    xy=x*y                             
    xyp=2.0*xy/jnp.pi                      
    exx=jnp.exp(-x*x)                  
    f=-exx*erfcx(y)*jnp.sin(2.0*xy)+x/jnp.pi*exx*jnp.sinc(xyp)           
    n=jnp.arange(1,ncut+1)             
    n2=n*n                             
    vec0=0.5*n/(0.25*n2+ y*y)            
    vec1=jnp.exp(-(0.25*n2+x*x))   
    vec4=jnp.exp(-(0.5*n+x)*(0.5*n+x)) 
    vec5=jnp.exp(-(0.5*n-x)*(0.5*n-x)) 
    Sigma1=jnp.dot(vec0,vec1)
    Sigma4=jnp.dot(vec0,vec4)
    Sigma5=jnp.dot(vec0,vec5)
    f = f + 1.0/jnp.pi*(y*jnp.sin(2.0*xy)*Sigma1 + 0.5*(Sigma5-Sigma4))
    
    return f


@jit
def rewofz_scan(x, y):
    ncut=27
    xy=x*y
    xyp=xy/jnp.pi
    exx=jnp.exp(-x*x)
    f=exx*erfcx(y)*jnp.cos(2.0*xy)+x*jnp.sin(xy)/jnp.pi*exx*jnp.sinc(xyp)
    narr=jnp.arange(1,ncut+1)
    def fscan(sv,n):
        n2=n*n
        fac0=1.0/(0.25*n2+y*y)
        Sigma1, Sigma2, Sigma3=sv
        Sigma1=Sigma1+fac0*jnp.exp(-(0.25*n2+x*x))
        Sigma2=Sigma2+fac0*jnp.exp(-(0.5*n+x)*(0.5*n+x))
        Sigma3=Sigma3+fac0*jnp.exp(-(0.5*n-x)*(0.5*n-x))
        return (Sigma1, Sigma2, Sigma3), None
    s0=jnp.zeros(jnp.shape(x))
    sv,null=scan(fscan,(s0,s0,s0),narr)
    Sigma1,Sigma2,Sigma3=sv
    f = f + 1.0/jnp.pi*(-y*jnp.cos(2.0*xy)*Sigma1 + 0.5*y*Sigma2 + 0.5*y*Sigma3)
    return f

@jit
def rewofzt2(x,y):
    """Real part of asymptotic representation of wofz function 1 for |z|**2 > 111 (for e = 10e-6)
    
    See Zaghloul (2018) arxiv:1806.01656

    Args:
        x: 
        y:
        
    Returns:
        f: Real(wofz(x+iy))

    """
    z=x+y*(1j)
    q=(1j)*z/(jnp.sqrt(jnp.pi))*(z*z - 2.5)/(z*z*(z*z-3.0) + 0.75)
    return jnp.real(q)

@jit
def rewofzs1(x,y):
    """Real part of asymptotic representation of wofz function 1 for |z|**2 > 236 (for e = 10e-6)
    
    See Zaghloul (2018) arxiv:1806.01656

    Args:
        x: 
        y:
        
    Returns:
        f: Real(wofz(x+iy))

    """
    z=x+y*(1j)
    a=1.0/(2.0*z*z)
    q=(1j)/(z*jnp.sqrt(jnp.pi))*(1.0 + a*(1.0 + a*3.0))
    return jnp.real(q)

@jit
def rewofzs3(x,y):
    """Real part of asymptotic representation of wofz function 1 for |z|**2 > 111 (for e = 10e-6)

    See Zaghloul (2018) arxiv:1806.01656
    
    Args:
        x: 
        y:
        
    Returns:
        f: Real(wofz(x+iy))

    """

    z=x+y*(1j)
    a=1.0/(2.0*z*z)
    q=(1j)/(z*jnp.sqrt(jnp.pi))*(1.0 + a*(1.0 + a*(3.0 + a*(15.0+a*105.0))))
    return jnp.real(q)
