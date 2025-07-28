import matplotlib

matplotlib.use("qtagg")

import csv

import matplotlib.pyplot as plt
import numpy as np

import catalog
import line_fit as lf
import plots
import spectr

colors = ["b", "g", "r", "c", "m", "y"]


def plot_zstack(rangs, resos, norm=False, base="../Data/Npy/", save=None):
    """legacy plot of stacks in two redshift bins separated by z=2.5 ."""
    a = catalog.fetch_json("../catalog_z.json")["sources"]
    # plots.histogram_in(catalog.rm_bad(a), 'z')
    ah = catalog.filter_zranges(a, rangs)
    ahf = catalog.rm_bad(ah)
    ahflz = catalog.value_range(ahf, "z", [0, 2.5])
    ahfhz = catalog.value_range(ahf, "z", [2.5, 20])
    # plots.histogram_in(ahflz, 'z')
    # plots.histogram_in(ahfhz, 'z')
    print(f"In low z bin:\t{len(ahflz)}\nIn high z bin:\t{len(ahfhz)}")

    for sources in [ahflz, ahfhz]:
        fig, axs = plt.subplots()
        for i, rang in enumerate(rangs):
            if type(resos) is list:
                reso = resos[i]
            else:
                reso = resos
            spectra, sourn = spectr.resampled_spectra(sources, rang, reso, base=base)
            assert (
                np.array(spectra).size > 0
            ), "No sources fall within the chosen range."
            stack = spectr.combine_spectra(spectra)

            stacked = spectr.stack(stack, sourn, typ="median")
            plots.spectra_plot(stacked, axis=axs, c="black", label="median", norm=norm)
        fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)
    else:
        plt.show()


def plot_simple(sources, rangs, resos, typ="median", base="../Data/Npy/", save=None):
    """Plots stack for sources covering a specified range."""
    fig, axs = plt.subplots()
    sources = catalog.filter_zranges(sources, rangs)
    for i, rang in enumerate(rangs):
        if type(resos) is list:
            reso = resos[i]
        else:
            reso = resos
        spectra, sourn = spectr.resampled_spectra(sources, rang, reso, base=base)
        assert np.array(spectra).size > 0, "No sources fall within the chosen range."
        stack = spectr.combine_spectra(spectra)

        stacked = spectr.stack(stack, sourn, typ=typ)
        plots.spectra_plot(stacked, axis=axs, c="black", label=str(typ), norm=False)
    axs.axhline(y=0, c="gray", ls=":")
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)
    else:
        plt.show()


def plot_zstacks(
    sources, rangs, zrangs, resos, base="../Data/Npy/", save=None, axis=None, fits=None
):
    """Plots stacks of sources covering specified range in separately in specified redshift bins."""
    if axis is None:
        fig, axs = plt.subplots()
    else:
        fig = plt.gcf()
        axs = axis
    sources = catalog.filter_zranges(sources, rangs)
    ahfr = [catalog.value_range(sources, "z", r) for r in zrangs]
    print("")
    print([len(s) for s in ahfr])
    for k, sours in enumerate(ahfr):
        if len(sours) != 0:
            c = colors[k % len(colors)]
            stacked = []
            for i, rang in enumerate(rangs):
                if type(resos) is list:
                    reso = resos[i]
                else:
                    reso = resos
                spectra, sourn = spectr.resampled_spectra(sours, rang, reso, base=base)
                stack = spectr.combine_spectra(spectra)
                stacc = spectr.stack(stack, sourn, typ="median")
                stacked.append(stacc)
            plots.spectras_plot(
                stacked, axis=axs, c=c, label=f"{zrangs[k]} ({len(sours)})", norm=False
            )
            for i, rang in enumerate(rangs):
                if fits is not None and rang[0] <= min(fits) and max(fits) <= rang[1]:
                    fit, x = lf.fit_lines(stacked[i], fits, grat=sours[0]["grat"])
                    axs.plot(x, fit(x), ls=":", c="gray")
    axs.legend()
    axs.axhline(y=0, c="gray", ls=":")
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)
        plt.close(fig)
    else:
        return axs


def plot_stacks(sources, rang, reso, save=None):
    """Plot results of different stacking methods."""
    sources = catalog.filter_zranges(a, [rang])

    fig, axs = plt.subplots()
    spectra, sourn = spectr.resampled_spectra(sources, rang, reso)
    sp_stack = spectr.combine_spectra(spectra)
    stacked = spectr.stack(sp_stack, sourn, typ="median")
    plots.spectra_plot(stacked, axis=axs, c="black", label="median")
    stacked = spectr.stack(sp_stack, sourn, typ="mean")
    plots.spectra_plot(stacked, axis=axs, label="mean")
    stacked = spectr.stack(sp_stack, sourn, typ="mean", normalise=True)
    plots.spectra_plot(stacked, axis=axs, label="mean norm")
    stacked = spectr.stack(sp_stack, sourn, typ="sn", normalise=True)
    plots.spectra_plot(stacked, axis=axs, label="sn")
    stacked = spectr.stack(sp_stack, sourn, typ="ha", normalise=True)
    plots.spectra_plot(stacked, axis=axs, label="ha")
    axs.legend()
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)
        plt.close(fig)
    else:
        plt.show()


def histograms(sources):
    """Plots histograms of parameters of provided sources."""
    plots.histogram_in(sources, "z", bins=20, range=(0, 14))
    plots.histogram_in(sources, "sn50", bins=20, range=(0, 30))
    plots.histogram_in(sources, "Ha", bins=20, range=(-5, 90))
    plots.histogram_in(sources, "phot_restU", bins=20, range=(-0.5, 3))
    plots.histogram_in(sources, "z_phot", bins=20, range=(0, 14))
    plots.histogram_in(sources, "phot_mass", bins=20, range=(0, 2 * 10**10))


def hist_region(sources, rangs, save=None):
    """Plots coverage of a given region(s) in redshift among given sources"""
    fig, axs = plt.subplots()
    zr = [0, 12]
    aa = catalog.filter_zranges(sources, rangs)
    plots.histogram_in(
        aa, "z", color="black", axis=axs, label=str(rangs), range=zr, bins=20, lw=2.5
    )
    rj = [min([x for r in rangs for x in r]), max([x for r in rangs for x in r])]
    aj = catalog.filter_zranges(sources, [rj])
    plots.histogram_in(
        aj,
        "z",
        color="grey",
        axis=axs,
        label=str([rj]),
        range=zr,
        bins=20,
        lw=2.5,
        ls=":",
    )
    for i, rang in enumerate(rangs):
        al = catalog.filter_zranges(sources, [rang])
        plots.histogram_in(
            al,
            "z",
            color=colors[i % len(colors)],
            axis=axs,
            label=str([rang]),
            range=zr,
            bins=20,
        )
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)
        plt.close(fig)
    else:
        plt.show()


def hist_in_z(sources, value, zrangs, range=None, save=None, norm=False):
    """Plots histogram of a given value in specified redshift bins."""
    fig, axs = plt.subplots()
    for i, zrang in enumerate(zrangs):
        al = [s for s in sources if s["z"] is not None and zrang[0] < s["z"] < zrang[1]]
        plots.histogram_in(
            al,
            value,
            color=colors[i % len(colors)],
            axis=axs,
            label=str(zrang),
            range=range,
            bins=20,
            norm=norm,
        )
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)
        plt.close(fig)
    else:
        plt.show()


def plot_all_lines():
    """Plots large set of various diagnostic plots for lines of interest."""
    a = catalog.fetch_json("../catalog_z.json")["sources"]
    af = catalog.rm_bad(a)
    afp = [s for s in af if s["grat"] == "prism"]
    afm = [s for s in af if s["grat"][-1] == "m" and s["grat"][0] == "g"]
    afh = [s for s in af if s["grat"][-1] == "h" and s["grat"][0] == "g"]
    path0 = "../Plots/lines3/"
    srs = {"medium": afm, "high": afh, "prism": afp}
    sources = {
        "npy": "../Data/Npy/",
        "ppxf": "../Data/Subtracted/",
        "smoo": "../Data/Subtracted_b/",
    }
    lines = {
        "S23": [[0.45, 0.52], [0.63, 0.69], [0.89, 0.95]],
        "N2O2": [[0.35, 0.4], [0.63, 0.69]],
        "N2S2": [[0.63, 0.71]],
        "R3": [[0.45, 0.52]],
        "O3N2": [[0.45, 0.52], [0.63, 0.69]],
        "R2": [[0.35, 0.4], [0.46, 0.52]],
    }
    z = {
        "S23": [[0, 1.5], [1.5, 2.5], [2.5, 4], [4, 20]],
        "N2O2": [[0, 1.5], [1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 20]],
        "N2S2": [[0, 1.5], [1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 20]],
        "R3": [[0, 1.5], [1.5, 2.5], [2.5, 4], [4, 6], [6, 20]],
        "O3N2": [[0, 1.5], [1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 20]],
        "R2": [[0, 1.5], [1.5, 2.5], [2.5, 4], [4, 6], [6, 20]],
    }
    for sr in srs:
        for k in lines:
            hist_region(srs[sr], lines[k], save=path0 + sr + k + "hist.png")
            for da in sources:
                plot_zstacks(
                    srs[sr],
                    lines[k],
                    z[k],
                    300,
                    base=sources[da],
                    save=path0 + sr + k + da + ".png",
                )


def cont_diff(source, plot=False):
    """For a provided single source plots comparison with continuum approximation templates."""
    spectr1 = spectr.get_spectrum_n(source, base="../Data/Continuum/")
    spectr2 = spectr.get_spectrum_n(source, base="../Data/Continuum_b/")
    spectro = spectr.get_spectrum_n(source, base="../Data/Npy/")
    if plot:
        plots.spectras_plot([spectr1, spectr2, spectro])
        plt.show()
    if spectr1 is not None and spectr2 is not None:
        return spectr.relat_diff(spectr1, spectr2)
    else:
        return None


if __name__ == "__main__":
    # plot_zstack([[0.47, 0.69]], 300)
    # histograms()
    plot_all_lines()
    print("")
