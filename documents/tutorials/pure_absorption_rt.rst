We compare trans2E3 (2 E3) with a simple transmission.

.. code:: ipython3

    from exojax.rt.rtransfer import trans2E3

.. code:: ipython3

    def simple_trans(dtau, mu):
        return jnp.exp(-dtau/mu)

.. code:: ipython3

    import jax.numpy as jnp
    dtau_arr=jnp.logspace(-3,1,100)

.. code:: ipython3

    import matplotlib.pyplot as plt
    plt.plot(dtau_arr,trans2E3(dtau_arr),color="black",label="$\\mathcal{T}= 2E_3 = 2 \\int_0^1 d \\mu \\, \\mu \\, e^{-\\Delta \\tau/\\mu}$")
    plt.plot(dtau_arr,simple_trans(dtau_arr,1),ls="dotted",color="gray", label="$\\mathcal{T}= e^{-\\Delta \\tau/\\mu} \\, \\, (\\mu=1)$")
    plt.plot(dtau_arr,simple_trans(dtau_arr,2.0/3.0),ls="dashed",color="gray", label="$\\mathcal{T}= e^{-\\Delta \\tau/\\mu} \\, \\, (\\mu=2/3)$")
    plt.plot(dtau_arr,simple_trans(dtau_arr,0.3),ls="dashdot",color="gray", label="$\\mathcal{T}= e^{-\\Delta \\tau/\\mu} \\, \\, (\\mu=0.3)$")
    #plt.yscale("log")
    plt.legend()
    plt.xscale("log")
    plt.xlabel("$\\Delta \\tau$")
    plt.ylabel("transmission")
    plt.savefig("transrt.png")
    plt.savefig("transrt.pdf")
    plt.show()



.. image:: pure_absorption_rt_files/pure_absorption_rt_4_0.png


