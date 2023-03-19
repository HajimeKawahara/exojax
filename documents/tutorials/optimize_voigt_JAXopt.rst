Optimization of a Voigt profile using JAXopt
============================================

.. code:: ipython3

    from exojax.spec.lpf import voigt
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import jaxopt

Let’s optimize the Voigt function :math:`V(\nu, \beta, \gamma_L)` using
exojax! :math:`V(\nu, \beta, \gamma_L)` is a convolution of a Gaussian
with a STD of :math:`\beta` and a Lorentian with a gamma parameter of
:math:`\gamma_L`.

.. code:: ipython3

    nu=jnp.linspace(-10,10,100)
    plt.plot(nu, voigt(nu,1.0,2.0)) #beta=1.0, gamma_L=2.0




.. parsed-literal::

    [<matplotlib.lines.Line2D at 0x7fbcef7569a0>]




.. image:: optimize_voigt_JAXopt_files/optimize_voigt_JAXopt_3_1.png


optimization of a simple absorption model
-----------------------------------------

Next, we try to fit a simple absorption model to mock data. The
absorption model is

$ f= e^{-a
V(:raw-latex:`\nu`,:raw-latex:`\beta`,:raw-latex:`\gamma`\_L)}$

.. code:: ipython3

    def absmodel(nu,a,beta,gamma_L):
        return jnp.exp(-a*voigt(nu,beta,gamma_L))

Adding a noise…

.. code:: ipython3

    from numpy.random import normal
    data=absmodel(nu,2.0,1.0,2.0)+normal(0.0,0.01,len(nu))
    plt.plot(nu,data,".")




.. parsed-literal::

    [<matplotlib.lines.Line2D at 0x7fb90d0398b0>]




.. image:: optimize_voigt_JAXopt_files/optimize_voigt_JAXopt_8_1.png


Let’s optimize the multiple parameters

We define the objective function as :math:`obj = |d - f|^2`

.. code:: ipython3

    # loss or objective function
    def objective(params):
        a,beta,gamma_L=params
        f=data-absmodel(nu,a,beta,gamma_L)
        g=jnp.dot(f,f)
        return g


.. code:: ipython3

    # Gradient Descent

.. code:: ipython3

    gd = jaxopt.GradientDescent(fun=objective, maxiter=10)
    res = gd.run(init_params=(1.5,0.7,1.5))
    params, state = res

.. code:: ipython3

    params




.. parsed-literal::

    (DeviceArray(1.9579332, dtype=float32, weak_type=True),
     DeviceArray(1.0382165, dtype=float32, weak_type=True),
     DeviceArray(1.8850585, dtype=float32, weak_type=True))



.. code:: ipython3

    from numpy.random import normal
    model=absmodel(nu,params[0],params[1],params[2])
    plt.plot(nu,model)
    plt.plot(nu,data,".")




.. parsed-literal::

    [<matplotlib.lines.Line2D at 0x7fb90cf3d490>]




.. image:: optimize_voigt_JAXopt_files/optimize_voigt_JAXopt_15_1.png


.. code:: ipython3

    #NCG

.. code:: ipython3

    gd = jaxopt.NonlinearCG(fun=objective, maxiter=100)
    res = gd.run(init_params=(1.5,0.7,1.5))
    params, state = res

.. code:: ipython3

    params




.. parsed-literal::

    (DeviceArray(1.9526778, dtype=float32),
     DeviceArray(1.0492882, dtype=float32),
     DeviceArray(1.8708111, dtype=float32))



.. code:: ipython3

    from numpy.random import normal
    model=absmodel(nu,params[0],params[1],params[2])
    plt.plot(nu,model)
    plt.plot(nu,data,".")




.. parsed-literal::

    [<matplotlib.lines.Line2D at 0x7fb90c0d6eb0>]




.. image:: optimize_voigt_JAXopt_files/optimize_voigt_JAXopt_19_1.png


