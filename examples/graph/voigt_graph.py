#https://gist.github.com/niklasschmitz/559a1f717f3535db0e26d0edccad0b46
import jax
from jax import core
from graphviz import Digraph
import itertools
import jax.numpy as jnp
from jax import vmap
from jax.lax import scan

styles = {
  'const': dict(style='filled', color='#78934F'),
  'invar': dict(color='#E7AB20', style='filled'),
  'outvar': dict(style='filled,dashed', fillcolor='#DE726B', color='black'),
  'op_node': dict(shape='box', color='#FBF7D8', style='filled'),
  'intermediate': dict(style='filled', color='#E7AB20')
}

def _jaxpr_graph(jaxpr):
  id_names = (f'id{id}' for id in itertools.count())
  graph = Digraph(engine='dot')
  graph.attr(size='6,10!')
  for v in jaxpr.constvars:
    graph.node(str(v), core.raise_to_shaped(v.aval).str_short(), styles['const'])
  for v in jaxpr.invars:
    graph.node(str(v), v.aval.str_short(), styles['invar'])
  for eqn in jaxpr.eqns:
    for v in eqn.invars:
      if isinstance(v, core.Literal):
        graph.node(str(id(v.val)), core.raise_to_shaped(core.get_aval(v.val)).str_short(),
                   styles['const'])
    if eqn.primitive.multiple_results:
      id_name = next(id_names)
      graph.node(id_name, str(eqn.primitive), styles['op_node'])
      for v in eqn.invars:
        graph.edge(str(id(v.val) if isinstance(v, core.Literal) else v), id_name)
      for v in eqn.outvars:
        graph.node(str(v), v.aval.str_short(), styles['intermediate'])
        graph.edge(id_name, str(v))
    else:
      outv, = eqn.outvars
      graph.node(str(outv), str(eqn.primitive), styles['op_node'])
      for v in eqn.invars:
        graph.edge(str(id(v.val) if isinstance(v, core.Literal) else v), str(outv))
  for i, v in enumerate(jaxpr.outvars):
    outv = 'out_'+str(i)
    graph.node(outv, outv, styles['outvar'])
    graph.edge(str(v), outv)
  return graph


def jaxpr_graph(fun, *args):
  jaxpr = jax.make_jaxpr(fun)(*args).jaxpr
  return _jaxpr_graph(jaxpr)


def grad_graph(fun, *args):
  _, fun_vjp = jax.vjp(fun, *args)
  jaxpr = fun_vjp.args[0].func.args[1]
  return _jaxpr_graph(jaxpr)


def erfcx(x):
    """erfcx (float) based on Shepherd and Laframboise (1981)
    
    Scaled complementary error function exp(-x*x) erfc(x)

    Args:
         x: should be larger than -9.3

    Returns:
         jnp.array: erfcx(x)

    Note: 
       We acknowledge the post in stack overflow (https://stackoverflow.com/questions/39777360/accurate-computation-of-scaled-complementary-error-function-erfcx)

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

def hjert(x,a):
    """Voigt-Hjerting function, consisting of a combination of rewofz and real(wofzs2).
    
    Args:
        x: 
        a:
        
    Returns:
        hjert: H(x,a) or Real(wofz(x+ia))

    Examples:
       
       hjert provides a Voigt-Hjerting function. 
       
       >>> hjert(1.0,1.0)
          DeviceArray(0.3047442, dtype=float32)

       This function accepts a scalar value as an input. Use jax.vmap to use a vector as an input.

       >>> from jax import vmap
       >>> x=jnp.linspace(0.0,1.0,10)
       >>> vmap(hjert,(0,None),0)(x,1.0)
          DeviceArray([0.42758358, 0.42568347, 0.4200511 , 0.41088563, 0.39850432,0.3833214 , 0.3658225 , 0.34653533, 0.32600054, 0.3047442 ],dtype=float32)
       >>> a=jnp.linspace(0.0,1.0,10)
       >>> vmap(hjert,(0,0),0)(x,a)
          DeviceArray([1.        , 0.8764037 , 0.7615196 , 0.6596299 , 0.5718791 ,0.49766064, 0.43553388, 0.3837772 , 0.34069115, 0.3047442 ],dtype=float32)

    """
    r2=x*x+a*a
#    return jnp.where(r2<111., rewofz(x,a), jnp.real(wofzs2(x,a)))
    return jnp.where(r2<111., rewofz(x,a), rewofzs2(x,a))

def voigt(nu,sigmaD,gammaL):
    """Voigt profile using Voigt-Hjerting function 

    Args:
       nu: wavenumber
       sigmaD: sigma parameter in Doppler profile 
       gammaL: broadening coefficient in Lorentz profile 
 
    Returns:
       v: Voigt profile

    """
    
    sfac=1.0/(jnp.sqrt(2)*sigmaD)
    vhjert=vmap(hjert,(0,None),0)
    v=sfac*vhjert(sfac*nu,sfac*gammaL)/jnp.sqrt(jnp.pi)
    return v
  
# example use

f = lambda x: voigt(x,1.0,1.0)
x = jnp.linspace(-1,1,10)
#g = grad_graph(f, x)
g = jaxpr_graph(f, x)
g.view() # will show inline in a notebook (alt: g.view() creates & opens Digraph.gv.pdf)
