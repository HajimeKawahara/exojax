
def rotgray(vsini,epsilon):
    denominator = np.pi * vsini * (1.0 - epsilon / 3.0)
    lambda_ratio_sqr = (delta_lambdas / delta_lambda_l) ** 2.0
    c1 = 2.0 * (1.0 - epsilon) / denominator
    c2 = 0.5 * np.pi * epsilon / denominator
    kernel = c1 * np.sqrt(1.0 - lambda_ratio_sqr) + c2 * (1.0 - lambda_ratio_sqr)

    return kernel



if __name__ == "__main__":
    from exojax.spec import AutoRT
    import numpy as np 
    
    N=1000
    wav=np.linspace(22900,23000,N,dtype=np.float64)#AA
    nus=1.e8/wav[::-1]
    Parr=np.logspace(-8,2,100) #100 pressure layers (10**-8 to 100 bar)
    Tarr = 1500.*(Parr/Parr[-1])**0.02    #some T-P profile
    autort=AutoRT(nus,1.e5,Tarr,Parr)     #g=1.e5 cm/s2
    autort.addmol("ExoMol","CO",0.01)     #mmr=0.01
    F=autort.rtrun()

    ###########################
    from PyAstronomy import pyasl
    rF = pyasl.rotBroad(wav, F[::-1], 0.0, 5.3)[::-1]
    
    import matplotlib.pyplot as plt
    plt.plot(wav,F[::-1])
    plt.plot(wav,rF[::-1])

    plt.show()
    

