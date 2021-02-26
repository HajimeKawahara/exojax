import numpy as np
from scipy import interpolate
def extract_pf_bc(T0,element,moloratom):
    if moloratom=='atom':
        f = open('/Users/snugroho/japan/py4cats/src/table8.dat.txt', "r")
    elif moloratom=='mol':
        f = open('/Users/snugroho/japan/py4cats/src/table6.dat.txt', "r")
    for line in f:
        line = line.strip()
        column = line.split()
        if column[0]=="T[K]":
            T=np.array(column[1:],dtype=float)
        if column[0]== element:
            QT_BC=np.array(column[1:],dtype=float)
    f.close()
    Q_inter = interpolate.interp1d(T,QT_BC,kind="cubic")(T0)
    return Q_inter
