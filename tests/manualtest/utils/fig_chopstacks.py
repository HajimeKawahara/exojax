def test_chopstack():
    from exojax.utils.chopstacks import buildwall, cutput, check_preservation
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
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

if __name__ == "__main__":
    test_chopstack()
