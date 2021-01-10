import jax.numpy as jnp
from jax import jit
from jax.lax import scan

@jit
def erfcx(x):
    """erfcx (float) based on Shepherd and Laframboise (1981)
    
    Params:
        x: should be larger than -9.3
        
    Return:
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
    
    Params:
        x: x < nend/2 
        y:
        
    Return:
        f: Real(wofz(x+iy))
    """
    nend=4000
    xy=x*y
    xyp=xy/jnp.pi
    exx=jnp.exp(-x*x)
    f=exx*erfcx(y)*jnp.cos(2.0*xy)+x*jnp.sin(xy)/jnp.pi*exx*jnp.sinc(xyp)
       
#    n=jnp.arange(1,nend+1,dtype=float)
    n=jnp.arange(1,nend+1)
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
def perfcx(x):
    """erfcx (float) based on Shepherd and Laframboise (1981) for semi-positive x
    
    Params:
        x: should be larger than 0.0
        
    Return:
        f: erfcx(x)
    """
    q = (-x*(x-2.0)/(x+2.0)-2.0*((x-2.0)/(x+2.0)+1.0)+x)/(x+2.0) + (x-2.0)/(x+2.0)
    
    _CHEV_COEFS_=[5.92470169e-5,1.61224554e-4, -3.46481771e-4,-1.39681227e-3,1.20588380e-3, 8.69014394e-3,
     -8.01387429e-3,-5.42122945e-2,1.64048523e-1,-1.66031078e-1, -9.27637145e-2, 2.76978403e-1]
    chev=jnp.array(_CHEV_COEFS_)
    def fmascan(c,y):
        return c*q + y,None
    p,n = scan(fmascan, 0.0, chev)

    q = (p+1.0)/(1.0+2.0*x)
    d = (p+1.0)-q*(1.0+2.0*x)
    f = 0.5*d/(x+0.5) + q    
    return f
