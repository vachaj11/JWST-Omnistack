import csv

import matplotlib.pyplot as plt
import numpy as np

import catalog
import plots
import spectr


def plot_zstack(rangs, resos, norm=False):
    a = catalog.fetch_json("../catalog.json")["sources"]
    # plots.histogram_in(catalog.rm_bad(a), 'z')
    ah = catalog.filter_zranges(a, rangs)
    ahf = catalog.rm_bad(ah)
    ahflz = catalog.value_range(ahf, "z", [0, 2.5])
    ahfhz = catalog.value_range(ahf, "z", [2.5, 20])
    # plots.histogram_in(ahflz, 'z')
    # plots.histogram_in(ahfhz, 'z')
    print(f"In low z bin:\t{len(ahflz)}\nIn high z bin:\t{len(ahfhz)}")

    for sources in [ahflz, ahfhz]:
        fig = plt.gcf()
        axs = plt.gca()
        for i, rang in enumerate(rangs):
            if type(resos) is list:
                reso = resos[i]
            else:
                reso = resos
            spectra, sourn = spectr.resampled_spectra(sources, rang, reso)
            stack = spectr.combine_spectra(spectra)

            stacked = spectr.stack(stack, sourn, typ="median")
            plots.spectra_plot(stacked, axis=axs, c="black", label="median", norm=norm)
        plt.show()


def plot_zstacks(rangs, zrangs, resos):
    colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
    fig = plt.gcf()
    axs = plt.gca()
    a = catalog.fetch_json("../catalog.json")["sources"]
    # plots.histogram_in(catalog.rm_bad(a), 'z')
    ah = catalog.filter_zranges(a, rangs)
    ahf = catalog.rm_bad(ah)
    ahfr = [catalog.value_range(ahf, "z", r) for r in zrangs]
    print([len(s) for s in ahfr])
    for k, sources in enumerate(ahfr):
        c = colors[k % len(colors)]
        stacked = []
        for i, rang in enumerate(rangs):
            if type(resos) is list:
                reso = resos[i]
            else:
                reso = resos
            spectra, sourn = spectr.resampled_spectra(sources, rang, reso)
            stack = spectr.combine_spectra(spectra)
            stacked.append(spectr.stack(stack, sourn, typ="median"))
        plots.spectras_plot(
            stacked, axis=axs, c=c, label=f"{zrangs[k]} ({len(sources)})", norm=True
        )
    axs.legend()
    plt.show()


def plot_stacks(rang, reso):
    a = catalog.fetch_json("../catalog.json")["sources"]
    # plots.histogram_in(catalog.rm_bad(a), 'z')
    ah = catalog.filter_zranges(a, [rang])
    sources = catalog.rm_bad(ah)

    fig = plt.gcf()
    axs = plt.gca()
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
    plt.show()


def histograms():
    a = catalog.fetch_json("../catalog.json")["sources"]
    af = catalog.rm_bad(a)
    plots.histogram_in(af, "z", bins=20, range=(0, 14))
    plots.histogram_in(af, "sn50", bins=20, range=(0, 30))
    plots.histogram_in(af, "Ha", bins=20, range=(-5, 90))
    plots.histogram_in(af, "phot_restU", bins=20, range=(-0.5, 3))
    plots.histogram_in(af, "z_phot", bins=20, range=(0, 14))
    plots.histogram_in(af, "phot_mass", bins=20, range=(0, 2 * 10**10))


def add_photometry(sources, pathp):
    dic = catalog.construct_dict(pathp)
    print("here")
    values = [
        "phot_Av",
        "phot_mass",
        "phot_restU",
        "phot_restV",
        "phot_restJ",
        "z_phot",
        "phot_LHa",
        "phot_LOIII",
        "phot_LOII",
    ]
    for i, source in enumerate(sources):
        sid = source["file"]
        g = None
        for s in dic:
            if sid == s["file"].replace("-v4_", "-v3_"):
                g = s
                break
        if g is not None:
            for v in values:
                source[v] = g[v]
        else:
            for v in values:
                source[v] = None
        print(f"\r{i} {type(g)}", end="")
    return sources


if __name__ == "__main__":
    # plot_zstack([[0.47, 0.69]], 300)
    # histograms()
    print("")
