def vf_stokes(r,g,eta,drho,Nkn=0.):
    """terminal velocity of Stokes flow (Reynolds number << 1)
    
    Args:
        r: particle size (cm)
        g: gravity (cm/s2)
        eta: dynamic viscosity (g/s/cm)
        drho: density difference between condensates and atmosphere (g/cm3)
        Nkn: Knudsen number
        
    
    Return:
        terminal velocity (cm/s)

    Note:
        (1.0+1.255*Nkn) is the Cunningham factor

    Note:
       See also (10-138) p415 in Hans R Pruppacher and James D Klett. Microstructure of atmospheric clouds and precipitation. In Microphysics of clouds and precipitation, pages 10–73. Springer, 2010. Equation (B1) in Appendix B of Ackerman and Marley 01. 

        
    """
    return 2.0*g*r*r*drho*(1.0+1.255*Nkn)/(9.0*eta)


def logNre_midNre(logNd):
    """Log (Davies number) - log (Reynolds number) fitting relation for mid Reynolds number (N_re = 1--1000)

    Args:
       logNd: logarithm of Davies number

    Returns:
       logarithm of Reynolds number
    
    Note:
       A fitting form (0.8x-0.01x^2)  written in Ackerman and Marley (2001) cannot explain the data in Table 10.1 in Pruppacher and Klett. Y. Ito found this fact and told us. 

    Examples:

       >>> # coefficient can be obtained by fitting a poly model to Table 10.1 in Pruppacher and Klett
       >>> data = pd.read_csv("/home/kawahara/exojax/data/clouds/drag_force.txt",comment="#",delimiter=",")
       >>> logNre=np.log(data["Nre"].values) #Reynolds number
       >>> Cd=(data["Cd_rigid"].values)
       >>> logNd=np.log(Nre**2*Cd)
       >>> coeff=np.polyfit(logNd,logNre,2)

    """

    return -0.00883374*xarr**2+0.84514511*xarr-2.49105354    

def logNre_midNre(logNd):
    """Log (Davies number) - log (Reynolds number) fitting relation for large Reynolds number (N_re >=1000)

    Args:
       logNd: logarithm of Davies number

    Returns:
       logarithm of Reynolds number

    """
    return 0.5*logNd+0.4
