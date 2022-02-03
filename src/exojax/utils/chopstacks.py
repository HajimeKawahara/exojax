#!/usr/bin/python
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def cutput(xw, f, hxw, hf=None, silent=None):
    """Cutting, putting, and redistributing the values.

    Args:
       xw: walls of the original (input) bins [X]
       f: input value [q/X]
       hxw: walls of the resampled bins [X]
       hf: stacking array [q/X], if None, use np.zeros array

    Returns:
       stacking array after redistribution of f
    """
    if hf is None:
        if silent is None:
            print('Reset master data (hf) to zero.')
        hf = np.zeros(len(hxw)-1)

    ind = np.digitize(hxw, xw)
    for i in range(0, len(hxw)-1):

        if ind[i] < len(xw) and ind[i+1]-1 < len(xw)-1 and ind[i] > 0:  # cx1,cx2,cx3
            if ind[i] == ind[i+1]:
                hf[i] = hf[i]+f[ind[i]-1]  # A0
            else:
                hf[i] = hf[i]+f[ind[i]-1] * \
                    (xw[ind[i]]-hxw[i])/(hxw[i+1]-hxw[i])  # B0
                for k in range(ind[i]+1, ind[i+1]):
                    hf[i] = hf[i]+f[k-1]*(xw[k]-xw[k-1]) / \
                        (hxw[i+1]-hxw[i])  # B1
                hf[i] = hf[i]+f[ind[i+1]-1] * \
                    (hxw[i+1]-xw[ind[i+1]-1])/(hxw[i+1]-hxw[i])  # B2

        elif ind[i+1]-1 == len(xw)-1:  # right boundary criterion
            if ind[i] < ind[i+1]:
                hf[i] = hf[i]+f[ind[i]-1] * \
                    (xw[ind[i]]-hxw[i])/(hxw[i+1]-hxw[i])  # B0
                for k in range(ind[i]+1, ind[i+1]):
                    hf[i] = hf[i]+f[k-1]*(xw[k]-xw[k-1]) / \
                        (hxw[i+1]-hxw[i])  # B1

        elif ind[i] == 0 and ind[i+1] > 0:  # left boundary condition
            for k in range(ind[i]+1, ind[i+1]):
                hf[i] = hf[i]+f[k-1]*(xw[k]-xw[k-1])/(hxw[i+1]-hxw[i])  # B1
            hf[i] = hf[i]+f[ind[i+1]-1] * \
                (hxw[i+1]-xw[ind[i+1]-1])/(hxw[i+1]-hxw[i])  # B2

    return hf


def buildwall(x, edge='half'):
    """building conventional walls.

    Args:
       x: input bins
       edge: how to define the edge. half=cutting a half bin at the edges. full=extending a half bin at the edge

    Returns:
       walls
    """

    nx = len(x)
    nxw = nx+1
    xw = np.zeros(nxw)
    for i in range(1, nx):
        xw[i] = (x[i]+x[i-1])/2.0

    if edge == 'half':
        xw[0] = x[0]
        xw[nx] = x[nx-1]
    elif edge == 'full':
        xw[0] = 1.5*x[0]-0.5*x[1]
        xw[nx] = 1.5*x[nx-1]-0.5*x[nx-2]
    else:
        print(edge, ' does not exist in the edge mode')
        sys.exit('exit')

    return xw


def check_preservation(xw, f, hxw, hf):
    """checking preservation of flux.

    Args:
       xw: walls of the original (input) bins [X]
       f: input value [q/X]
       hxw: walls of the resampled bins [X]
       hf: stacking array [q/X], if None, use np.zeros array
    """
    sum = 0.0
    llsum = 0.0

    for i in range(0, len(hxw)-1):
        bbw = hxw[i+1]-hxw[i]
        llsum = llsum+bbw*hf[i]

    for i in range(0, len(xw)-1):
        bw = xw[i+1]-xw[i]
        sum = sum+bw*f[i]

    print('(1 - sum input/sum output) =', 1 - sum/llsum)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(
        description='Test for the binning routine.')
    args = parser.parse_args()

    print('just for test')
    # master hat
    hxw = np.arange(0.0, 100.0, 1.5)
    hx = np.zeros(len(hxw)-1)
    for i in range(0, len(hxw)-1):
        hx[i] = (hxw[i+1]+hxw[i])/2.0
    hxw = buildwall(hx)

    # adding array
    xw = np.arange(10.0, 110.0, 1.7)
    x = np.zeros(len(xw)-1)
    xw[10] = xw[10]-1.0
    xw[18:25] = xw[18:25]+1.0
    xw[40:] = (xw[40:]-xw[40])*0.5 + xw[40]
    for i in range(0, len(xw)-1):
        x[i] = (xw[i+1]+xw[i])/2.0
        f = (x-50)**2
    xw = buildwall(x)
    hf = cutput(xw, f, hxw)

    check_preservation(xw, f, hxw, hf)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(hxw[0:len(hxw)-1], hf, '|', color='red')
    ax.plot(hxw[1:len(hxw)], hf, '|', color='red')
    ax.plot(x, f, '.', color='blue')
    ax.plot(hx, hf, '.', color='red')
    ax.plot(xw[0:len(xw)-1], f, '|', color='blue')
    ax.plot(xw[1:len(xw)], f, '|', color='blue')
    plt.show()
