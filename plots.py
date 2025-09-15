"""Holds various simple and lower-level plotting function in turn utilised by other more complex plotting code or quick visualisations."""

import matplotlib

matplotlib.use("qtagg")
import matplotlib.pyplot as plt
import numpy as np

import catalog
import line_fit as lf
import spectr

"""
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Computer Modern",
        "font.size": 15,
    }
)
"""


def spectra_plot(spectra, axis=None, norm=False, **kwargs):
    """Plots the provided spectra on a specified or newly created axis. With an option to normalise the spectra."""
    if axis is None:
        fig = plt.gcf()
        axis = plt.gca()
    else:
        fig = plt.gcf()
        axis = axis
    if norm:
        spectra = spectr.spect_norm(spectra)
    axis.plot(spectra[0], spectra[1], **kwargs)
    fig.set_layout_engine(layout="tight")
    axis.set_xlabel("Wavelength ($\mu$m)")


def spectras_plot(spectra, axis=None, norm=False, label="_", **kwargs):
    """Plots all spectra in the provided list on a specified or newly created axis. If requested the spectra can be also first individually normalised."""
    if axis is None:
        fig = plt.gcf()
        axis = plt.gca()
    else:
        fig = plt.gcf()
        axis = axis
    if norm:
        spectra = spectr.spects_norm(spectra)
    for i, spectrum in enumerate(spectra):
        axis.plot(
            spectrum[0],
            spectrum[1],
            label=label * int(i + 1 == len(spectra)) + "_" * int(i + 1 != len(spectra)),
            **kwargs,
        )
    fig.set_layout_engine(layout="tight")
    axis.set_xlabel("Wavelength ($\mu$m)")


def histogram_in(
    sources, value, bins=None, range=None, axis=None, label="", norm=False, **kwargs
):
    """Plots a histogram of population of spectra in provided catalogue as a function of specified parameter."""
    if axis is None:
        fig = plt.gcf()
        axis = plt.gca()
    else:
        fig = plt.gcf()
        axis = axis
    hist, bins, _ = catalog.value_bins(
        sources, value, bins=bins, range=range, density=norm
    )
    axis.stairs(hist, bins, label=f"{label}" + f"({hist.sum()})" * (not norm), **kwargs)
    axis.set_xlabel(value)
    axis.legend()
    fig.set_layout_engine(layout="tight")


def plot_values(sources, valx, valy, axis=None, **kwargs):
    """Plots a scatter plot of two specified parameters (as long as they are available) of spectra in the provided catalogue."""
    if axis is None:
        fig = plt.gcf()
        axis = plt.gca()
    else:
        fig = plt.gcf()
        axis = axis
    valsx = []
    valsy = []
    for s in sources:
        if valx and valy in s.keys():
            x = s[valx]
            y = s[valy]
            if x is not None and y is not None:
                valsx.append(x)
                valsy.append(y)
    axis.plot(
        valsx, valsy, ls="", marker="+", c="black", label=f"({len(valsx)})", **kwargs
    )
    axis.set_xlabel(valx)
    axis.set_ylabel(valy)
    axis.legend()
    fig.set_layout_engine(layout="tight")


def plot_fit(spectra, fit, sources=None, axis=None, text=True, save=None, plot=True):
    """Plots a provided spectrum overlayed with provided AstroPy fitting model to a specified or newly created axis. Adding information about the model to the plot if requested."""
    if axis is None:
        fig, axs = plt.subplots()
    else:
        fig = plt.gcf()
        axs = axis
    label = f"({len(sources)})" if sources is not None else ""
    spectras_plot([spectra], axis=axs, label=label, norm=False)
    x = np.linspace(min(spectra[0]), max(spectra[0]), 200)
    axs.plot(x, fit(x), ls=":", c="gray")
    if text:
        plot_fit_text(fit, axis=axs)
    if label:
        axs.legend()
    axs.axhline(y=0, c="gray", ls=":")
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)
    elif plot:
        plt.show()
        plt.close(fig)


def plot_fit_text(fit, axis=None):
    """Extracts information about a provided fitting model from a provided AstroPy object, converts them to text and places the text to appropriate position on a specified or newly created axis."""
    if axis is None:
        fig = plt.gcf()
        axis = plt.gca()
    else:
        fig = plt.gcf()
        axis = axis
    yoff = fit._leaflist[0].yoff.value
    for g in fit._leaflist[1:]:
        mean = g.mean.value
        dens = yoff + g.amplitude.value
        flux = g.flux
        axis.text(
            mean,
            dens,
            f"$\\lambda = {mean:.4f}\\mu m$\n$ \mathrm{{Flux}} = {ftL(flux, 2)}\\mathrm{{W}}/\\mathrm{{m}}^2$",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    return axis


def ftL(num, d=2):
    """Renders the provided number in LaTeX syntax and 'scientific' notation."""
    neg = False
    if num == 0:
        return "0.0"
    elif num < 0:
        neg = True
        num = -num
    mag = int(np.floor(np.log10(num)))
    numd = num * 10 ** (-mag)
    strd = neg * "-" + str(numd)[: 2 + d]
    if mag == 0:
        return strd
    else:
        return strd + f"\\cdot 10^{{{mag}}}"


def spectra_resolution(sources):
    """Plot very simple diagram of spectral element size as a function of redshift for each type of spectra in the passed catalogue."""
    fig = plt.gcf()
    axs = plt.gca()
    gratings = dict()
    examp = dict()
    for s in sources:
        g = s["file"].split("_")[1]
        if g not in gratings.keys():
            gratings[g] = 1
            examp[g] = s
        else:
            gratings[g] += 1
    spectras = [spectr.get_spectrum(examp[s]) for s in examp]
    for g, spectra in zip(gratings, spectras):
        wavs = spectra[0]
        dwav = [wavs[i + 1] - wavs[i] for i in range(len(wavs) - 1)]
        # or L/DeltaL
        wavs = wavs[:-1]
        axs.plot(wavs, np.array(dwav), label=f'{g.split("-")[0]} ({gratings[g]})')
        axs.set_yscale("log")
    axs.legend(loc=1)
    axs.set_xlabel("Wavelength ($\mu$m)")
    axs.set_ylabel("$\Delta$Wavelength ($\mu$m)")
    fig.set_layout_engine(layout="tight")
