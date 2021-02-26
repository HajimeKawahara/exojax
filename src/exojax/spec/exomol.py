from jax import jit, vmap
import jax.numpy as jnp

@jit
def SijT(T,A,g,nu_ij,elower,QT):
    """Line strength as a function of temperature

    Args:
       T: temperature (K)
       nu_ij: line center wavenumber (cm-1)
       elower: elower 
       QT: partition function Q(T)

    Returns:
       Sij(T): Line strength

    """


@jit
def gamma_exomol(P, T, n_air):
    """gamma factor by a pressure broadening 
    
    Args:
       P: pressure (bar)
       T: temperature (K)
       n_air: coefficient of the  temperature  dependence  of  the  air-broadened halfwidth
       alpha_ref: broadening parameter

    Returns:
       gamma: pressure gamma factor (cm-1) 

    """
    Pref=1.01325 #atm (bar)
    Tref=296.0 #reference tempearture (K)
    gamma=alpha_ref*(P/Pref)*(Tref/T)**n_air
    return gamma


@jit
def gamma_natural(A):
    """gamma factor by natural broadning
    
    1/(4 pi c) = 2.6544188e-12 (cm-1 s)

    Args:
       A: Einstein A-factor (1/s)

    Returns:
       gamma_natural: natural width (cm-1)

    """
    return 2.6544188e-12*A


@jit
def doppler_sigma(nu_ij,T,M):
    """Dopper width (sigma)
    
    Note:
       c3 is sqrt(kB/m_u)/c

    Args:
       nu_ij: line center wavenumber (cm-1)
       T: temperature (K)
       M: atom/molecular mass
    
    Returns:
       sigma: doppler width (standard deviation) (cm-1)

    """
    c3=3.0415595e-07
    return c3*jnp.sqrt(T/M)*nu_ij
