import numpy as np

#(a, b, a', b') a,b for temeprature exoponent, a', b' for gamma 
#from https://github.com/hitranonline/planetary-broadeners
nonair_coeff_CO_in_H2 = (np.array([0.64438, 0.49261, -0.0748, 0.0032]),
                         np.array([0.69861, -0.09569, 0.003, 5.7852E-5]),
                         np.array([0.08228, -0.07411, 0.10795, 0.00211]),
                         np.array([-1.0, 1.53458, 0.03054, 6.9468E-5]))

nonair_coeff_CO_in_He = (np.array([0.5393, 0.1286, -0.0129, 0.00175]),
                         np.array([0.3146, -0.0417, 0.00403, -6.589E-6]),
                         np.array([0.0809, 0.3641, -0.04025, 0.00178]),
                         np.array([8.1769, -0.9105, 0.0397, 2.556E-6]))

nonair_coeff_CO_in_CO2 = (np.array([0.70343, -0.10857, 0.00407, 1.112E-4]),
                          np.array([-0.14755, 0.00528, 1.3829E-4, 1.4546E-6]),
                          np.array([0.12106, 0.05433, -0.00851, 6.90673E-4]),
                          np.array([0.63012, -0.07902, 0.006, 1.703E-4]))


def nonair_polynomial(m, a_coeff, b_coeff):
    """nonair polynomial

    Notes:
        value = ((a0 + a1 * x + a2 * x**2 + a3 * x**3) /
        (1 + b1 * x + b2 * x**2 + b3 * x**3 + b4 * x**4))
        where x = m

    Args:
        m (int array): m transition state
        a_coeff (array): nonair coefficient a 
        b_coeff (array): nonair coefficient b 

    Returns:
        float array: nonair value
    """
    numerator = np.power(m[:, np.newaxis], np.arange(len(a_coeff))) @ a_coeff
    denominater = np.power(m[:, np.newaxis],
                           np.arange(len(b_coeff)) + 1) @ b_coeff + 1.0
    return numerator / denominater

def temperature_exponent_nonair(m, coeff):
    """non air temperature exponent

    Args:
        m (int array): m transition state
        coeff (array): nonair coefficient (a,b,a',b') 
    Returns:
        float array: nonair tempearure exponent
    """
    return nonair_polynomial(m, coeff[0], coeff[1])

def gamma_nonair(m, coeff):
    """non air gamma (Lorentian width at reference)

    Args:
        m (int array): m transition state
        coeff (array): nonair coefficient (a,b,a',b') 
    Returns:
        float array: nonair Lorentian width at reference (cm-1)
    """
    return nonair_polynomial(m, coeff[2], coeff[3])
