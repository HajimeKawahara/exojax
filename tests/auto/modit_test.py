import numpy
from exojax.spec.rtransfer import nugrid
from exojax.spec import AutoXS
from exojax.spec import AutoRT
import matplotlib.pyplot as plt

if False:
    nus=numpy.logspace(numpy.log10(1900.0),numpy.log10(2300.0),160000,dtype=numpy.float64)
    #nus=numpy.logspace(numpy.log10(2041.6),numpy.log10(2041.7),10000,dtype=numpy.float64)
    #nus=numpy.logspace(numpy.log10(2040),numpy.log10(2043),10000,dtype=numpy.float64)

    #MODIT worse for higher-T and/or lower-P
    T=2000.0
    P=1.0
    autoxs=AutoXS(nus,"ExoMol","CO",xsmode="LPF") 
    xsv0=autoxs.xsection(T,P) 
    autoxs=AutoXS(nus,"ExoMol","CO",xsmode="MODIT",autogridconv=False)
    xsv1=autoxs.xsection(T,P)
    autoxs=AutoXS(nus,"ExoMol","CO",xsmode="DIT") 
    xsv2=autoxs.xsection(T,P) 
    
    fig=plt.figure()
    ax=fig.add_subplot(211)
    plt.plot(nus,xsv0,label="LPF",color="C0",alpha=0.4)
    plt.plot(nus,xsv1,label="MODIT",color="C1",alpha=0.4)
    plt.plot(nus,xsv2,label="DIT",color="C2",alpha=0.4)
    plt.legend(loc="upper right")
    plt.plot(nus,xsv1-xsv0,color="C1",alpha=0.4)
    plt.plot(nus,xsv2-xsv0,color="C2",alpha=0.4)
    
    ax=fig.add_subplot(212)
    plt.plot(nus,xsv2-xsv0,label="DIT-LPF",color="C2",alpha=0.4)
    plt.plot(nus,xsv1-xsv0,label="MODIT-LPF",color="C1",alpha=0.7)
    plt.legend(loc="upper right")
    plt.show()


nusobs=numpy.linspace(1900.0,2300.0,10000,dtype=numpy.float64)
    
xsmode="MODIT"
nus,wav,res=nugrid(1900.0,2300.0,160000,"cm-1",xsmode=xsmode)
Parr=numpy.logspace(-8,2,100) #100 layers from 10^-8 bar to 10^2 bar
Tarr = 500.*(Parr/Parr[-1])**0.02    
autort=AutoRT(nus,1.e5,2.33,Tarr,Parr,xsmode=xsmode,autogridconv=False) #g=1.e5 cm/s2, mmw=2.33
autort.addcia("H2-H2",0.74,0.74)       #CIA, mmr(H)=0.74
autort.addmol("ExoMol","CO",0.01)      #CO line, mmr(CO)=0.01
F1=autort.rtrun()
#F1o=autort.spectrum(nusobs,100000.0,20.0,0.0)

xsmode="DIT"
nus,wav,res=nugrid(1900.0,2300.0,160000,"cm-1",xsmode=xsmode)
Parr=numpy.logspace(-8,2,100) #100 layers from 10^-8 bar to 10^2 bar
Tarr = 500.*(Parr/Parr[-1])**0.02    
autort=AutoRT(nus,1.e5,2.33,Tarr,Parr,xsmode=xsmode) #g=1.e5 cm/s2, mmw=2.33
autort.addcia("H2-H2",0.74,0.74)       #CIA, mmr(H)=0.74
autort.addmol("ExoMol","CO",0.01)      #CO line, mmr(CO)=0.01
F2=autort.rtrun()
#F2o=autort.spectrum(nusobs,100000.0,20.0,0.0)
   
xsmode="LPF"
nus,wav,res=nugrid(1900.0,2300.0,160000,"cm-1",xsmode=xsmode)
Parr=numpy.logspace(-8,2,100) #100 layers from 10^-8 bar to 10^2 bar
Tarr = 500.*(Parr/Parr[-1])**0.02    
autort=AutoRT(nus,1.e5,2.33,Tarr,Parr,xsmode=xsmode) #g=1.e5 cm/s2, mmw=2.33
autort.addcia("H2-H2",0.74,0.74)       #CIA, mmr(H)=0.74
autort.addmol("ExoMol","CO",0.01)      #CO line, mmr(CO)=0.01
F0=autort.rtrun()
#F0o=autort.spectrum(nusobs,100000.0,20.0,0.0)

fig=plt.figure()
ax=fig.add_subplot(211)
plt.plot(nus,F0,label="LPF",color="C0",alpha=0.4)
plt.plot(nus,F1,label="MODIT",color="C1",alpha=0.4)
plt.plot(nus,F1-F0,color="C1",alpha=0.4)
plt.legend(loc="upper right")


ax=fig.add_subplot(212)
plt.plot(nus,F1-F0,label="MODIT-LPF",color="C1",alpha=0.4)
plt.legend(loc="upper right")


#ax=fig.add_subplot(413)
#plt.plot(nusobs,F0o,label="LPF (obs)",color="C0",alpha=0.4,ls="dashed",lw=2)
#plt.plot(nusobs,F1o,label="MODIT (obs)",color="C1",alpha=0.4,ls="dashed",lw=2)
#plt.plot(nusobs,F1o-F0o,color="C1",alpha=0.4)
#plt.legend(loc="upper right")

#ax=fig.add_subplot(414)
#plt.plot(nusobs,F1o-F0o,color="C1",alpha=0.4,lw=2,label="MODIT-LPF (obs)")
#plt.legend(loc="upper right")

plt.show()
