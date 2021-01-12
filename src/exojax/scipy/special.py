"""special functions in exojax

   exojax.scipy.special  -- special functions

"""

from jax import jit
from jax import custom_vjp
import jax.numpy as jnp
from jax.lax import scan
from jax.interpreters.ad import defvjp

@jit
def erfcx(x):
    """erfcx (float) based on Shepherd and Laframboise (1981)
    
    Scaled complementary error function exp(-x*x) erfc(x)

    Args:
         x: should be larger than -9.3

    Returns:
         f: erfcx(x)

    """
    a=jnp.abs(x)
    q = (-a*(a-2.0)/(a+2.0)-2.0*((a-2.0)/(a+2.0)+1.0)+a)/(a+2.0) + (a-2.0)/(a+2.0)
    _CHEV_COEFS_=[5.92470169e-5,1.61224554e-4, -3.46481771e-4,-1.39681227e-3,1.20588380e-3, 8.69014394e-3,-8.01387429e-3,-5.42122945e-2,1.64048523e-1,-1.66031078e-1, -9.27637145e-2, 2.76978403e-1]
    chev=jnp.array(_CHEV_COEFS_)
    
    def fmascan(c,x):
        return c*q + x,None
    
    p,n = scan(fmascan, 0.0, chev)
    q = (p+1.0)/(1.0+2.0*a)
    d = (p+1.0)-q*(1.0+2.0*a)
    f = 0.5*d/(a+0.5) + q    
    f=jnp.where(x>=0.0, f, 2.0*jnp.exp(a**2) - f) 
    
    return f


@jit
def rewofz(x,y):
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
def imwofz(x, y):
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
def rewofzs2(x,y):
    """Real part of asymptotic representation of wofz function 1 for |z|**2 > 112 (for e = 10e-6)

    See Zaghloul (2018) arxiv:1806.01656
    
    Args:
        x: 
        y:
        
    Returns:
        f: Real(wofz(x+iy))

    """

    z=x+y*(1j)
    a=1.0/(2.0*z*z)
    q=(1j)/(z*jnp.sqrt(jnp.pi))*(1.0 + a*(1.0 + a*(3.0 + a*15.0)))
    return jnp.real(q)


@jit
def rewofz_ej2(x,y):
    """Combined version of rewofz and rewofzs2
    
    Args:
        x: 
        y:
        
    Returns:
        f: Real(wofz(x+iy))

    """
    r2=x*x+y*y
    return jnp.where(r2<111., rewofz(x,y), rewofzs2(x,y))

@custom_vjp
def rewofzx(x, y):
    """[VJP custom defined] Real part of wofz function based on Algorithm 916
    
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

def h_fwd(x, y):
    hh=rewofzx(x, y)
    return hh, (hh, imwofz(x, y), x, y)

def h_bwd(res, g):
    """backward
    Args:
        res, g:

    Returns:
        h1,h2: g* partial_x h(x,y), g* partial_y h(x,y)

    V=Real(wofz), L=Imag(wofz)
    """
    V, L, x, y = res 
    return (2.0 * (y*L - x*V) * g , 2.0 * (x*L + y*V) * g - 2.0/jnp.sqrt(jnp.pi))

rewofzx.defvjp(h_fwd, h_bwd)
