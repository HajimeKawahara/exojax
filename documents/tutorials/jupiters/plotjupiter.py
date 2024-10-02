import numpy as np
import matplotlib.pyplot as plt


def print_wavminmax(wav_obs):
    print(
        "wavelength range used in this analysis=",
        np.min(wav_obs),
        "--",
        np.max(wav_obs),
        "AA",
    )


def plot_opt(nus_obs, spectra, F_samp_init, F_samp):
    fig = plt.figure(figsize=(30, 5))
    ax = fig.add_subplot(111)
    plt.plot(nus_obs, spectra, ".", label="observed spectrum")
    plt.plot(nus_obs, F_samp_init, alpha=0.5, label="init", color="C1", ls="dashed")
    plt.plot(nus_obs, F_samp, alpha=0.5, label="best fit", color="C1", lw=3)
    plt.legend()
    plt.xlim(np.min(nus_obs), np.max(nus_obs))
    plt.savefig("Jupiter_petitIRD.png")
    return plt


def plot_spec1(unmask_nus_obs, unmask_spectra, nus_obs, spectra):
    fig = plt.figure(figsize=(20, 5))
    plt.plot(nus_obs, spectra, ".")
    plt.plot(unmask_nus_obs, unmask_spectra, alpha=0.5)
    return plt


def plot_spec2(nus_obs, spectra, solspec, nus_solar, vperc):
    fig = plt.figure(figsize=(20, 5))
    ax = fig.add_subplot(211)
    plt.plot(nus_obs, spectra, label="masked spectrum")
    plt.plot(nus_solar * (1.0 + vperc), solspec, lw=1, label="Solar")
    plt.xlabel("wavenumber (cm-1)")
    plt.xlim(np.min(nus_obs), np.max(nus_obs))
    plt.ylim(0.0, 0.25)
    plt.legend()
    return plt


def plotPT(art, Tarr):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(Tarr, art.pressure)
    ax.invert_yaxis()
    plt.yscale("log")
    plt.xscale("log")
    return plt


def plot_cloud_structure(Parr, rg_layer, MMRc, fac):
    fig = plt.figure()
    ax = fig.add_subplot(131)
    plt.plot(rg_layer, Parr)
    plt.xlabel("rg (cm)")
    plt.ylabel("pressure (bar)")
    plt.yscale("log")
    ax.invert_yaxis()
    ax = fig.add_subplot(132)
    plt.plot(MMRc, Parr)
    plt.xlabel("condensate MMR")
    plt.yscale("log")
    # plt.xscale("log")
    ax.invert_yaxis()
    ax = fig.add_subplot(133)
    plt.plot(fac * MMRc, Parr)
    plt.xlabel("cloud density g/L")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlim(1e-8, None)
    ax.invert_yaxis()
    return plt


def plot_extinction(nus, sigma_extinction, sigma_scattering, asymmetric_factor):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(311)
    plt.plot(nus, asymmetric_factor, color="black")
    plt.xscale("log")
    plt.ylabel("$g$")
    ax = fig.add_subplot(312)
    plt.plot(
        nus,
        sigma_scattering / sigma_extinction,
        label="single scattering albedo",
        color="black",
    )
    plt.xscale("log")
    plt.ylabel("$\\omega$")
    ax = fig.add_subplot(313)
    plt.plot(nus, sigma_extinction, label="ext", color="black")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("wavenumber (cm-1)")
    plt.ylabel("$\\beta_0$")
    plt.savefig("miefig_high.png")
    return plt


def plot_prediction(wav_obs, spectra, median_mu1, hpdi_mu1):
    plt.rcParams["font.family"] = "FreeSerif"
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111)
    # plt.plot(nus_obs, spectra, ".", label="observed spectrum")
    # plt.plot(
    #    nus_obs, median_mu1, alpha=0.5, label="median prediction", color="C1", ls="dashed"
    # )
    plt.plot(wav_obs, spectra, ".", label="observed spectrum")
    mask = wav_obs < 16220.0

    plt.plot(
        wav_obs[mask],
        median_mu1[mask],
        alpha=0.5,
        lw=2,
        label="median prediction",
        color="black",
    )

    ax.fill_between(
        wav_obs[mask],
        hpdi_mu1[0][mask],
        hpdi_mu1[1][mask],
        alpha=0.3,
        interpolate=True,
        color="C0",
        label="95% area",
    )
    
    plt.plot(
        wav_obs[~mask],
        median_mu1[~mask],
        alpha=0.5,
        lw=2,
        color="black",
    )

    ax.fill_between(
        wav_obs[~mask],
        hpdi_mu1[0][~mask],
        hpdi_mu1[1][~mask],
        alpha=0.3,
        interpolate=True,
        color="C0",
    )
    
    plt.legend(fontsize=16)
    #ax.fill_betweenx(
    #    [-0.02, 0.13],
    #    np.max(wav_obs[mask]),
    #    np.min(wav_obs[~mask]),
    #    alpha=0.2,
    #    interpolate=True,
    #    color="gray",
    #)
    plt.ylim(-0.02, 0.13)
    plt.xlim(np.min(wav_obs), np.max(wav_obs))
    plt.xlabel("wavelength $\AA$", fontsize=18)
    plt.ylabel("normalized spectrum", fontsize=18)
    plt.tick_params(labelsize=18)
    return plt


def plottp(torig, porig, Parr, Tarr):
    plt.plot(torig, porig)
    plt.plot(Tarr, Parr)
    plt.xlim(np.min(Tarr) - 10, np.max(Tarr))
    plt.ylim(np.min(Parr), np.max(Parr))
    plt.gca().invert_yaxis()
    plt.yscale("log")
    return plt
