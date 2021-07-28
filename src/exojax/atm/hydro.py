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
