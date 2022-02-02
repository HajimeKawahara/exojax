"""Plotting tool for DIT/MODIT."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_dgmn(Parr, dgm_ngammaL, ngammaLM, js, je):
    """plot MODIT grid matrix (dgm)

    Args:
       Parr: pressure layer (Nlayer)
       dgm_ngammaL: DIT grid matrix of ngammaL (Nlayer x Ngrid)
       ngammaLM: ngammaL matrix (Nlayer x Nline)
       js: layer index for start to display
       je: layer index for end to display
    """

    Ngy = np.shape(dgm_ngammaL)[1]
    if ngammaLM is not None:
        Nlines = np.shape(ngammaLM)[1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(js, je):
        if ngammaLM is not None:
            plt.plot(ngammaLM[i, :], Parr[i]*np.ones(Nlines), '.')
        for k in range(0, Ngy):
            plt.plot(dgm_ngammaL[i, k], Parr[i], '+', color='black')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Pressure')
    plt.xlabel('$\hat{\gamma}_L$')
    plt.gca().invert_yaxis()


def plot_dgm(dgm_sigmaD, dgm_gammaL, sigmaDM, gammaLM, js=0, je=10):
    """plot DIT grid matrix (dgm)

    Args:
       dgm_sigmaD: DIT grid matrix of sigmaD (Nlayer x Ngrid)
       dgm_gammaL: DIT grid matrix of gammaL (Nlayer x Ngrid)
       sigmaDM: sigmaD matrix (Nlayer x Nline)
       gammaLM: gammaL matrix (Nlayer x Nline)
       js: layer index for start to display
       je: layer index for end to display
    """
    Ngx = np.shape(dgm_sigmaD)[1]
    Ngy = np.shape(dgm_gammaL)[1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(js, je):
        plt.plot(sigmaDM[i, :], gammaLM[i, :], '.')
        px = np.array([dgm_sigmaD[i, 0], dgm_sigmaD[i, 0],
                      dgm_sigmaD[i, -1], dgm_sigmaD[i, -1]])
        py = np.array([dgm_gammaL[i, 0], dgm_gammaL[i, -1],
                      dgm_gammaL[i, -1], dgm_gammaL[i, 0]])
        ax.fill(px, py, color='gray', alpha=0.2)
        for j in range(0, Ngx):
            for k in range(0, Ngy):
                plt.plot(dgm_sigmaD[i, j],
                         dgm_gammaL[i, k], '+', color='black')
    plt.xlabel('$\sigma_D$')
    plt.ylabel('$\gamma_L$')
