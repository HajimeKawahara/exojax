import numpy as np
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
        raise ValueError(str(edge)+' does not exist in the edge mode')

    return xw


