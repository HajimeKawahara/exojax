import jax.numpy as jnp
import matplotlib.pyplot as plt
from exojax.spec.twostream import contribution_function_lart
from exojax.spec.twostream import solve_twostream_pure_absorption_numpy


def panel_imshow(val1, val2, title1, title2):
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.set_title(title1)
    a = ax.imshow(val1)
    plt.colorbar(a, shrink=0.5)
    ax.set_aspect(0.35 / ax.get_data_ratio())
    ax = fig.add_subplot(212)
    ax.set_title(title2)
    a = ax.imshow(val2)
    ax.set_aspect(0.35 / ax.get_data_ratio())
    plt.colorbar(a, shrink=0.5)
    plt.show()


def comparison_with_pure_absorption(cumThat, Qhat, spectrum, trans_coeff,
                                    scat_coeff, piB):
    cumTpure, Qpure, spectrum_pure = solve_twostream_pure_absorption_numpy(
        trans_coeff, scat_coeff, piB)

    contribution_function = contribution_function_lart(cumThat, Qhat)
    panel_imshow((trans_coeff), (scat_coeff),
                 "Transmission Coefficient $\mathcal{T}$",
                 "Scattering Coefficient $\mathcal{S}$")

    panel_imshow(cumThat, cumTpure, "cumThat", "cumTpure")
    panel_imshow(cumThat / cumTpure, cumThat, "cThat/cTpure", "cThat")
    panel_imshow(Qhat, Qpure, "Qhat", "Qpure")
    panel_imshow(Qhat / Qpure , Qpure, "Qhat/Qpure", "Qpure")
    
    panel_imshow(contribution_function, jnp.log10(contribution_function),
                 'contribution function ($\hat{Q} \prod \hat{T}$)',
                 'log scale')
    plt.plot(spectrum, label="That * Qhat ")
    plt.plot(spectrum_pure, label="Tpure * Qpure", ls="dashed", lw=2)
    plt.legend()
    plt.show()

    return spectrum
