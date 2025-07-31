import matplotlib

matplotlib.use("qtagg")

import csv

import matplotlib.pyplot as plt
import numpy as np

import catalog
import line_fit as lf
import plots
import spectr


def S_S23(sources, **kwargs):
    fsii = fit_lines(sources, [0.6718, 0.6733], [0.6718, 0.6733], **kwargs)
    fsiii = fit_lines(sources, [0.9532, 0.9548], 0.9532, **kwargs)
    # fsiii += fit_lines(sources, [0.9069, 0.9071], 0.9063, **kwargs)
    fhbe = fit_lines(sources, [0.4862], 0.4862, **kwargs)
    S23 = np.log10((fsii + fsiii) / fhbe)
    if np.isfinite(S23):
        return S23, [6.63 + 2.202 * S23 + 1.060 * S23**2]
    else:
        return S23, []


def N_N2O2(sources, **kwargs):
    fnii = fit_lines(sources, [0.6550, 0.6564, 0.6585], 0.6585, **kwargs)
    foii = fit_lines(sources, [0.3727, 0.3729], [0.3727, 0.3729], **kwargs)
    N2O2 = np.log10(fnii / foii)
    if np.isfinite(N2O2):
        return N2O2, [0.52 * N2O2 - 0.65]
    else:
        return N2O2, []


def N_N2(sources, **kwargs):
    fnii = fit_lines(sources, [0.6550, 0.6564, 0.6585], 0.6585, **kwargs)
    fhal = fit_lines(sources, [0.6550, 0.6564, 0.6585], 0.6564, **kwargs)
    N2 = np.log10(fnii / fhal)
    if np.isfinite(N2):
        return N2, [0.62 * N2 - 0.57]
    else:
        return N2, []


def N_N2S2(sources, **kwargs):
    N2S2 = O_N2(sources, **kwargs)[0] - O_S2(sources, **kwargs)[0]
    if np.isfinite(N2S2):
        return N2S2, [0.85 * N2S2 - 1.00]
    else:
        return N2S2, []


def O_N2(sources, **kwargs):
    fnii = fit_lines(sources, [0.6550, 0.6564, 0.6585], 0.6585, **kwargs)
    fhal = fit_lines(sources, [0.6550, 0.6564, 0.6585], 0.6564, **kwargs)
    N2 = np.log10(fnii / fhal)
    if np.isfinite(N2):
        p = [-0.489 - N2, 1.513, -2.554, -5.293, -2.867]
        roots = [
            np.real(v) for v in np.roots(p) + 8.69 if np.isreal(v) and 7.6 < v < 8.9
        ]
        return N2, roots
    else:
        return N2, []


def O_R3(sources, **kwargs):
    foiii = fit_lines(sources, [0.5008], 0.5008, **kwargs)
    fhbe = fit_lines(sources, [0.4862], 0.4862, **kwargs)
    R3 = np.log10(foiii / fhbe)
    if np.isfinite(R3):
        p = [-0.277 - R3, -3.549, -3.593, -0.981]
        roots = [
            np.real(v) for v in np.roots(p) + 8.69 if np.isreal(v) and 7.6 < v < 8.9
        ]
        return R3, roots
    else:
        return R3, []


def O_O3N2(sources, **kwargs):
    O3N2 = O_R3(sources, **kwargs)[0] - O_N2(sources, **kwargs)[0]
    if np.isfinite(O3N2):
        p = [0.281 - O3N2, -4.765, -2.268]
        roots = [
            np.real(v) for v in np.roots(p) + 8.69 if np.isreal(v) and 7.6 < v < 8.9
        ]
        return O3N2, roots
    else:
        return O3N2, []


def O_O3S2(sources, **kwargs):
    O3S2 = O_R3(sources, **kwargs)[0] - O_S2(sources, **kwargs)[0]
    if np.isfinite(O3S2):
        p = [0.191 - O3S2, -4.292, -2.538, 0.053, 0.332]
        roots = [
            np.real(v) for v in np.roots(p) + 8.69 if np.isreal(v) and 7.6 < v < 8.9
        ]
        return O3S2, roots
    else:
        return O3S2, []


def O_S2(sources, **kwargs):
    fsii = fit_lines(sources, [0.6718, 0.6733], [0.6718, 0.6733], **kwargs)
    fhal = fit_lines(sources, [0.6550, 0.6564, 0.6585], 0.6564, **kwargs)
    S2 = np.log10(fsii / fhal)
    if np.isfinite(S2):
        p = [-0.442 - S2, -0.360, -6.271, -8.339, -3.559]
        roots = [
            np.real(v) for v in np.roots(p) + 8.69 if np.isreal(v) and 7.6 < v < 8.9
        ]
        return S2, roots
    else:
        return S2, []


def O_R2(sources, **kwargs):
    foii = fit_lines(sources, [0.3727, 0.3729], [0.3727, 0.3729], **kwargs)
    fhbe = fit_lines(sources, [0.4862], 0.4862, **kwargs)
    R2 = np.log10(foii / fhbe)
    if np.isfinite(R2):
        p = [0.435 - R2, -1.362, -5.655, -4.851, -0.478, 0.736]
        roots = [
            np.real(v) for v in np.roots(p) + 8.69 if np.isreal(v) and 7.6 < v < 8.9
        ]
        return R2, roots
    else:
        return R2, []


def O_O3O2(sources, **kwargs):
    O3O2 = O_R3(sources, **kwargs)[0] - O_R2(sources, **kwargs)[0]
    if np.isfinite(O3O2):
        p = [-0.691 - O3O2, -2.944, -1.308]
        roots = [
            np.real(v) for v in np.roots(p) + 8.69 if np.isreal(v) and 7.6 < v < 8.9
        ]
        return O3O2, roots
    else:
        return O3O2, []


def O_R23(sources, **kwargs):
    foiii = fit_lines(sources, [0.5008], 0.5008, **kwargs)
    foii = fit_lines(sources, [0.3727, 0.3729], [0.3727, 0.3729], **kwargs)
    fhbe = fit_lines(sources, [0.4862], 0.4862, **kwargs)
    R23 = np.log10((foii + foiii) / fhbe)
    if np.isfinite(R23):
        p = [0.527 - R23, -1.569, -1.652, -0.421]
        roots = [
            np.real(v) for v in np.roots(p) + 8.69 if np.isreal(v) and 7.6 < v < 8.9
        ]
        return R23, roots
    else:
        return R23, []


def O_RS23(sources, **kwargs):
    RS23 = np.log10(
        10 ** (O_R3(sources, **kwargs)[0]) + 10 ** (O_S2(sources, **kwargs)[0])
    )
    if np.isfinite(RS23):
        p = [-0.054 - RS23, -2.546, -1.970, 0.082, 0.222]
        roots = [
            np.real(v) for v in np.roots(p) + 8.69 if np.isreal(v) and 7.6 < v < 8.9
        ]
        return RS23, roots
    else:
        return RS23, []


Oxygen = {
    "N2": O_N2,
    "R3": O_R3,
    "O3N2": O_O3N2,
    "O3S2": O_O3S2,
    "S2": O_S2,
    "R2": O_R2,
    "O3O2": O_O3O2,
    "R23": O_R23,
    "RS23": O_RS23,
}
Nitrogen = {"N2O2": N_N2O2, "N2": N_N2, "N2S2": N_N2S2}
Sulphur = {"S23": S_S23}


def fit_lines(
    sources,
    lines,
    mline,
    reso=300,
    base="../Data/Npy/",
    typ="median",
    plot=False,
    **kwargs
):
    grats = [s["grat"] for s in sources]
    grat = max(set(grats), key=grats.count)
    rang, _ = lf.line_range(lines, grat=grat)
    sources = catalog.filter_zranges(sources, [rang])
    spectra, sourn = spectr.resampled_spectra(sources, rang, reso, base=base)
    if len(sourn):
        stack = spectr.combine_spectra(spectra)
        stacc = spectr.stack(stack, sourn, typ=typ)
        fit, x = lf.fit_lines(stacc, lines, grat=grat)
        if type(mline) == list:
            flux = sum([lf.flux_at(fit, l) for l in mline])
        else:
            flux = lf.flux_at(fit, mline)
        if plot:
            plots.plot_fit(stacc, sourn, fit, **kwargs)
        return flux
    else:
        return np.nan


def flux_conv(sources, lines, lind, save=None, axis=None, typ="median"):
    if axis is None:
        fig, axs = plt.subplots()
    else:
        fig = plt.gcf()
        axs = axis
    mx = len(sources)
    n = np.linspace(mx / 20, mx, 200)
    y = []
    x = []
    for i in n:
        sample = np.random.choice(sources, size=int(i), replace=False)
        flux = fit_lines(sample, lines, lines[lind], plot=False, typ=typ)
        y.append(flux)
        x.append(i)
    x, y = zip(*sorted(zip(x, y)))
    axs.plot(x, y)
    axs.set_xlabel("Number of spectra stacked")
    axs.set_ylabel("Flux")
    axs.set_title("Convergence of line flux with stack size")
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)
    else:
        plt.show()
    plt.close(fig)


def abundance_in_z(sources, zrangs, abund=Oxygen, save=None, title=None, yax=None, **kwargs):
    n = int(-(-np.sqrt(len(abund)) // 1))
    fig = plt.figure()
    gs = fig.add_gridspec(n, n, hspace=0, wspace=0)
    axes = gs.subplots(sharex="col", sharey="row")
    if n == 1:
        axs = [axes]
    else:
        axs = axes.flatten()
    valss = []
    for i, (nam, ab) in enumerate(abund.items()):
        for zrang in zrangs:
            sourz = [s for s in sources if zrang[0] < s["z"] < zrang[1]]
            v = ab(sourz, **kwargs)[1]
            if v:
                zmean = [(zrang[1] + zrang[0]) / 2] * len(v)
                zerr = [(zrang[1] - zrang[0]) / 2] * len(v)
                axs[i].plot(zmean, v, ls="", marker="D", c="black")
                axs[i].errorbar(zmean, v, xerr=zerr, ls="", c="black", capsize=5)
            valss += v
        axs[i].set_title(nam, y=0.85)
    minz = min([min(zr) for zr in zrangs])
    maxz = max([max(zr) for zr in zrangs])
    ranz = maxz - minz
    minz = minz - ranz * 0.1
    maxz = maxz + ranz * 0.1
    rana = max(valss) - min(valss)
    mina = min(valss) - rana * 0.1
    maxa = max(valss) + rana * 0.25
    for i in range(n):
        axs[-i - 1].set_xlim(minz, maxz)
        axs[i * n].set_ylim(mina, maxa)
        axs[-i - 1].set_xlabel("$z$")
        axs[i * n].set_ylabel(yax)
    fig.set_size_inches((max(n, 2) + 0.3) * 2.5, (max(n, 2) + 0.4) * 2.5)
    fig.suptitle(title)
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)
    else:
        plt.show()
    plt.close(fig)
