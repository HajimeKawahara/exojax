"""Extract tau=1 height pressure at different wavelengths"""
import numpy as np
from scipy import interpolate

def pressure_at_given_opacity(dtau, Parr, tauextracted=1.0):
    """Extract pressure of tau=1(or a given value) at different wavelengths
    
    Args:
        dtau: delta tau for each layer [N_layer x N_nu]
        Parr: pressure at each layer [N_layer]
        tauextracted: the tau value at which pressure is extracted
    
    Returns:
        P_tauone:  [N_nu]
        
    Note:
        Return 0 if tau is too optically thin and does not reach tauextracted.
    """
    tau = np.cumsum(dtau, axis=0)
    P_tauone = np.zeros(tau.shape[1])
    for i in range(tau.shape[1]):
        if np.max(tau[:, i]) > tauextracted:
            P_tauone[i] = interpolate.interp1d(tau[:, i], Parr, kind="linear")(tauextracted)
    return P_tauone
