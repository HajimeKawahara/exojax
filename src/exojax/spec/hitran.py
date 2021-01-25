from jax import jit, vmap
import jax.numpy as jnp

@jit
def SijT(T,logsij0,nu_ij,glower,elower,qT):
    """Line strength as a function of temperature

    Args:
       T: temperature (K)
       logsij0: log(Sij(Tref)) (Tref=296K)
       nu_ij: line center wavenumber (cm-1)
       glower: g_lower or gpp
       elower: elower 
       qT: Q(Tref)/Q(T)

    Returns:
       Sij(T): Line strength

    """
    Tref=296.0 #reference tempearture (K)
    c_2 = 1.4387773 #hc/k_B (cm K)
    expow=logsij0-c_2*(elower/T-elower/Tref)
    fac=(1.0-jnp.exp(-c_2*nu_ij/T) )/(1.0-jnp.exp(-c_2*nu_ij/Tref))
    return jnp.exp(expow)*qT*fac





@jit
def gamma_hitran(P, T, Pself, n_air, gamma_air_ref, gamma_self_ref):
    """gamma factor by a pressure broadening 
    
    Args:
       P: pressure (atm)
       T: temperature (K)
       gamma_air_ref: gamma air 
       gamma_self_ref: gamma self 

    Returns:
       gamma: pressure gamma factor (cm-1) 

    """
    Tref=296.0 #reference tempearture (K)
    gamma=(Tref/T)**n_air *(gamma_air_ref*(P-Pself) + gamma_self_ref*(Pself))
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
