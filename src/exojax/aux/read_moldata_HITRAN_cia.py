#!/usr/bin/python
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pylab import *
from astropy import constants as const
from astropy import units as u
import mypallete as mp
from scipy import interpolate

def read_cia(file,nus,nue):
    #read first line
    f = open(file, "r")
    header = f.readline()
    info=header.strip().split()
    print info
    nwav=int(info[3])
    wav=[]
    for i in range(0,nwav):
        column = f.readline().strip().split()
        wav.append(float(column[0]))
    f.close()

    f = open(file, "r")
    tcia=[]
    for line in f:
        line = line.strip()
        column = line.split()
        if column[0] == file.replace("_2011.cia",""):
            tcia.append(float(column[4]))
    f.close()
    tcia=np.array(tcia)

    wav=np.array(wav)
    ijwav=np.digitize([nus,nue],wav)
    newwav=np.array(wav[ijwav[0]:ijwav[1]+1])

    #read data
    print(file.replace("_2011.cia",""))
    data=np.loadtxt(file,comments=file.replace("_2011.cia",""))
    nt=data.shape[0]/nwav    
    data=data.reshape((nt,nwav,2))    
    ac=data[:,ijwav[0]:ijwav[1]+1,1]

    print((len(newwav),ac.shape))

    return 1.e7/newwav,ac,tcia


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Read HITRAN cia data')
    parser.add_argument('-f', nargs=1, required=True, help='cia file')
    parser.add_argument('-w', nargs=2, required=True, help='wavelength [nm]',type=float)
    args = parser.parse_args()    
    file = args.f[0]
    nus=1.e7/args.w[1] # cm-1
    nue=1.e7/args.w[0] # cm-1
    wav,ac,tcia=read_cia(file,nus,nue)

    out=file+"w"+str(args.w[0])+"_"+str(args.w[1])
    np.savez(out, np.array(wav),np.array(ac),np.array(tcia))


#    fig=plt.figure()
#    yscale("log")
#    xscale('log')
#    xlabel("wavelength [nm]")
#    ylabel("absorption coefficient [cm5 molecules-2]")
#    plot(wav,ac[0,:])
#    plot(wav,ac[100,:])
#    plt.show()
