import jax.numpy as jnp
from jax import jit
from jax.lax import scan


xi = jnp.array([-0.9914553711208126392069,-0.9491079123427585245262,-0.8648644233597690727897,-0.7415311855993944398639,-0.5860872354676911302941,-0.4058451513773971669066,-0.2077849550078984676007,0,0.2077849550078984676007,0.4058451513773971669066,0.5860872354676911302941,0.7415311855993944398639,0.8648644233597690727897,0.9491079123427585245262,0.9914553711208126392069])
wi = jnp.array([0.0229353220105292249637,0.0630920926299785532907,0.1047900103222501838399,0.140653259715525918745,0.1690047266392679028266,0.1903505780647854099133,0.2044329400752988924142,0.209482141084727828013,0.2044329400752988924142,0.190350578064785409913,0.1690047266392679028266,0.1406532597155259187452,0.10479001032225018384,0.0630920926299785532907,0.02293532201052922496373])

@jit
def kernel(varr, zeta, vsini, u1, u2, beta):
    """Response Kernel including rotation, a Gaussian broadening (IP+microturbulence), radial-tangential turbulence,   
    Args:
       varr: velocity array
       zeta: macroturbulence disturbunce (Gray 2005)
       vsini: V sini for rotation
       u1: Limb-darkening coefficient 1
       u2: Limb-darkening coefficient 2
       beta: STD of a Gaussian broadening (IP+microturbulence)

    Returns:
       normalized response kernel

    Note:
       The unit of varr, zeta, vsini, and beta should be same, such as km/s.
       A small beta (<~1 km/s) induces an instability of the kernel. 

    """
    dv = varr[:,None] - vsini * xi
    dv2 = dv * dv
    ominx2 = 1. - xi * xi
    a, b = -jnp.sqrt(ominx2), jnp.sqrt(ominx2)
    d = 0.5 * (b - a)
    xs = d * xi[:,None]
    cosg2 = ominx2 - xs * xs
    cosg = jnp.sqrt(cosg2)
    sing2 = 1. - cosg2
    sigma2_cos = beta*beta + 0.5*zeta*zeta*cosg2
    sigma2_sin = beta*beta + 0.5*zeta*zeta*sing2
    ys = 0.5 * (jnp.exp(-0.5*dv2[:,None]/sigma2_cos)/jnp.sqrt(2*jnp.pi*sigma2_cos) + jnp.exp(-0.5*dv2[:,None]/sigma2_sin)/jnp.sqrt(2*jnp.pi*sigma2_sin))
    ys *= 1. - (1. - cosg) * (u1 + u2 * (1. - cosg))
    ys /= jnp.pi * (1. - u1/3. - u2/6.)
    xintegrand = d * jnp.sum(wi[:,None] * ys, axis=1)
    k=jnp.sum(wi * xintegrand, axis=1)
    ksum=jnp.sum(k)
    return k/ksum


def response(dvmat,F0,u1,u2,vsini,beta,RV,zeta=0.0):

    def respense_fscan(carry,varr):
        """function for scanning response
        Args:
        carry: dummy
        varr: velocity array

        Returns:
        dummy, kernel multiplied F       
        """
        Fr=jnp.sum(F0*kernel(varr, zeta, vsini, u1, u2, beta))
        return carry,Fr

    c=299792.458
    car,F=scan(respense_fscan,0.0,dvmat-RV)

    return F
