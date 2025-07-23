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


def histogram_in(sources, value, bins=None, range=None, axis=None, label="", **kwargs):
    if axis is None:
        fig = plt.gcf()
        axis = plt.gca()
    else:
        fig = plt.gcf()
        axis = axis
    hist, bins, _ = catalog.value_bins(sources, value, bins=bins, range=range)
    axis.stairs(hist, bins, label=f"{label} ({hist.sum()})", **kwargs)
    axis.set_xlabel(value)
    axis.legend()
    fig.set_layout_engine(layout="tight")


def plot_values(sources, valx, valy, axis=None, **kwargs):
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


def plot_linefit(spectrum, line, axis, **kwargs):
    if axis is None:
        fig = plt.gcf()
        axis = plt.gca()
    else:
        fig = plt.gcf()
        axis = axis
    m = lf.fit_line(spectrum, line, **kwargs)
    x = np.linspace(line - m.sigma * 2, line + m.sigma * 2, 200)
    axis.plot(x, m(x), ls=":", c="gray")
    # m.mean, m.sigma, m.amplitude, m.yoff


def spectra_resolution(sources):
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
    spectras = [spectr.get_spectrum_n(examp[s]) for s in examp]
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
