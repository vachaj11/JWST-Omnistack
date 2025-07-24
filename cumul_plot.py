import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

import catalog
import joint
import plots

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Computer Modern",
        "font.size": 15,
    }
)
lines = {
    0.9532: ["$\\mathrm{S}_\\mathrm{III} \\mathrm{\\,\\, 9532\\AA}$", 0.02],
    0.6725: ["$\\mathrm{S}_\\mathrm{II} \\mathrm{\\,\\, 6725\\AA}$", 0.01],
    0.6585: ["$\\mathrm{N}_\\mathrm{II} \\mathrm{\\,\\, 6585\\AA}$", 0.005],
    0.6564: ["$\\mathrm{H}_\\mathrm{\\alpha} \\mathrm{\\,\\, 6564\\AA}$", 0.025],
    0.5008: ["$\\mathrm{O}_\\mathrm{III} \\mathrm{\\,\\, 5008\\AA}$", 0.025],
    0.4862: ["$\\mathrm{H}_\\mathrm{\\beta} \\mathrm{\\,\\, 4862\\AA}$", 0.01],
    0.4686: ["$\\mathrm{He}_\\mathrm{II} \\mathrm{\\,\\, 4686\\AA}$", 0.005],
    0.4341: ["$\\mathrm{H}_\\mathrm{\\gamma} \\mathrm{\\,\\, 4341\\AA}$", 0.01],
    0.4102: ["$\\mathrm{H}_\\mathrm{\\delta} \\mathrm{\\,\\, 4102\\AA}$", 0.02],
    0.3728: ["$\\mathrm{O}_\\mathrm{II} \\mathrm{\\,\\, 3728\\AA}$", 0.02],
    0.1907: ["$\\mathrm{C}_\\mathrm{III} \\mathrm{\\,\\, 1907\\AA}$", 0.02],
    0.1883: ["$\\mathrm{Si}_\\mathrm{III} \\mathrm{\\,\\, 1883\\AA}$", 0.02],
    0.166: ["$\\mathrm{O}_\\mathrm{III} \\mathrm{\\,\\, 1660\\AA}$", 0.02],
    0.164: ["$\\mathrm{He}_\\mathrm{II} \\mathrm{\\,\\, 1640\\AA}$", 0.02],
    0.1216: ["$\\mathrm{Ly}_\\mathrm{\\alpha} \\mathrm{\\,\\, 1216\\AA}$", 0.03],
}

lines_C = {
    0.5007: ["$\\mathrm{O}_\\mathrm{III} \\mathrm{\\,\\, 5007\\AA}$", 0.025],
    0.4959: ["$\\mathrm{O}_\\mathrm{III} \\mathrm{\\,\\, 4959\\AA}$", 0.025],
    0.4862: ["$\\mathrm{H}_\\mathrm{\\beta} \\mathrm{\\,\\, 4862\\AA}$", 0.01],
    0.4363: ["$\\mathrm{O}_\\mathrm{III} \\mathrm{\\,\\, 4363\\AA}$", 0.025],
    0.4341: ["$\\mathrm{H}_\\mathrm{\\gamma} \\mathrm{\\,\\, 4341\\AA}$", 0.01],
    0.4102: ["$\\mathrm{H}_\\mathrm{\\delta} \\mathrm{\\,\\, 4102\\AA}$", 0.02],
    0.3869: ["$\\mathrm{Ne}_\\mathrm{III} \\mathrm{\\,\\, 3869\\AA}$", 0.02],
    0.3727: ["$\\mathrm{O}_\\mathrm{II} \\mathrm{\\,\\, 3727\\AA}$", 0.02],
    0.2465: ["$\\mathrm{Fe}_\\mathrm{III} \\mathrm{\\,\\, 2465\\AA}$", 0.02],
    0.1908: ["$\\mathrm{C}_\\mathrm{III} \\mathrm{\\,\\, 1908\\AA}$", 0.02],
}

ratios = {
    "Sulphur": ("gold", [0, 1, 5]),
    "Nitrogen": ("grey", [1, 2, 3, 9]),
    "Oxygen": ("green", [4, 5, 9]),
}


def plot_lines(sources, lines, title=None, save=None, narrow=1, ratios=None):
    fig, axs = plt.subplots()
    hists = []
    tickp = []
    tickl = []
    for i, l in enumerate(lines):
        sourl = catalog.filter_zranges(
            sources,
            [[l - lines[l][1] / narrow, l + lines[l][1] / narrow]],
            z_shift=True,
        )
        hist, bins, _ = catalog.value_bins(sourl, "z", bins=24, range=[0, 12])
        hist = np.where(~(hist == 0), hist, np.nan)
        points = np.array([bins, [i] * len(bins)]).T.reshape(-1, 1, 2)
        lines[l].append(np.concatenate([points[:-1], points[1:]], axis=1))
        lines[l].append(hist)
        hists.append(hist)
        tickp.append(i)
        tickl.append("$" + lines[l][0] + "$")
    hists = np.array(hists)
    hmax = max(np.nanmax(hists), 1)
    for l in lines:
        lc = LineCollection(lines[l][-2], cmap="Reds", lw=8)
        lc.set_array(lines[l][-1])
        # lc.set_norm(colors.LogNorm(vmin = 1, vmax = hmax))
        lc.set_clim(1, hmax)
        line = axs.add_collection(lc)
    if ratios is not None:
        for i, (nam, (c, itm)) in enumerate(ratios.items()):
            xs = [-0.5 * (i + 1)] * len(itm)
            lin = axs.plot(xs, itm, c=c, mfcalt=c, marker=".", alpha=0.3, markersize=25)
            lin[0].set_clip_on(False)
            axs.text(xs[0], -1, nam, c=c, rotation=-40, rotation_mode="anchor")
    axs.set_xlim(0, 12)
    axs.set_ylim(-0.5, len(lines) - 0.5)
    axs.set_yticks(tickp, labels=tickl)
    axs.set_xticks([i for i in range(13)])
    axs.grid()
    axs.tick_params(
        direction="in",
        grid_linestyle=":",
        grid_linewidth=1,
        grid_color="black",
        grid_alpha=0.5,
    )
    axs.set_xlabel(r"$\mathrm{Redshift}$")
    axs.set_ylabel(r"$\mathrm{Feature}$")
    fig.colorbar(line, ax=axs, label="Number of sources", pad=0.12)
    ax2 = axs.twinx()
    ax2.set_ylim(-0.5, len(lines) - 0.5)
    rlabels = np.char.add(
        np.char.add("(", np.nansum(hists, axis=1).astype(int).astype(str)), ")"
    )
    ax2.set_yticks(tickp, labels=rlabels)
    ax2.tick_params(direction="in")
    axs.set_title(title)
    fig.set_size_inches(10, 6)
    fig.tight_layout()
    if save is not None:
        fig.savefig(save)
        plt.close(fig)
    else:
        plt.show()


def plot_histograms(sources, lines, title=None, save=None, ymax=2600, narrow=1):
    fig = plt.figure()
    gs = fig.add_gridspec(-(-len(lines) // 3), 3, hspace=0, wspace=0)
    axes = gs.subplots(sharex="col", sharey="row")
    axs = axes.flatten()
    for i, l in enumerate(lines):
        lin = lines[l]
        al = catalog.filter_zranges(
            sources, [[l - lin[1] / narrow, l + lin[1] / narrow]]
        )
        plots.histogram_in(
            al,
            "z",
            axis=axs[i],
            range=[0, 12],
            bins=24,
            color="black",
            lw=2,
            label=lin[0] + " ",
        )
    for i in range(3):
        axs[-i - 1].set_xlabel(r"$\mathrm{Redshift}$")
        axs[-i - 1].set_xlim(-1, 13)
    for i in range(-(-len(lines) // 3)):
        axs[i * 3].set_ylabel(r"No. of sources")
        axs[i * 3].set_ylim(0, ymax)
    for ax in axs:
        leg = ax.get_legend()
        if leg is not None:
            txt = leg.texts[0].get_text()
            leg.remove()
            ax.set_title(txt, y=0.75)
    fig.suptitle(title)
    fig.set_size_inches(8, 9)
    fig.tight_layout()
    if save is not None:
        fig.savefig(save)
        plt.close(fig)
    else:
        plt.show()


def plot_stacks(sources, lines, zrang=None, title=None, save=None, ite=0, narrow=1):
    fig = plt.figure()
    gs = fig.add_gridspec(3, 5, hspace=0)
    axs = gs.subplots(sharex="col")
    keys = list(lines.keys())[ite * 5 : (ite + 1) * 5]
    if zrang is None:
        z = [[0, 1.5], [1.5, 3], [3, 4.5], [4.5, 20]]
    else:
        z = zrang
    bases = ["../Data/Npy/", "../Data/Subtracted/", "../Data/Subtracted_b/"]
    for i, l in enumerate(keys):
        lin = lines[l]
        rangs = [[l - lin[1] / narrow, l + lin[1] / narrow]]
        al = catalog.filter_zranges(sources, rangs)
        for b, bas in enumerate(bases):
            fits = [l] if b == 0 else None
            joint.plot_zstacks(al, rangs, z, 300, base=bas, axis=axs[b, i], fits=fits)
            if b != 2:
                axs[b, i].get_legend().remove()
            else:
                axs[b, i].get_legend().set(loc=1)
        axs[0, i].set_title(lin[0])
    axs[0, 0].set_ylabel("Flux\n ($\mu$J)")
    axs[1, 0].set_ylabel("Flux (Ppxf-subtracted)\n ($\mu$J)")
    axs[2, 0].set_ylabel("Flux (Smooth-subtracted)\n ($\mu$J)")
    fig.suptitle(title)
    fig.set_size_inches(19, 10)
    fig.tight_layout()
    if save is not None:
        fig.savefig(save, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_mz(sources, title=None, save=None, **kwargs):
    fig, axs = plt.subplots()
    plots.plot_values(sources, "phot_mass", "z", axis=axs, alpha=0.1, **kwargs)
    axs.set_xlim(10**5, 10**12)
    axs.set_ylim(0, 14.5)
    axs.set_xscale("log")
    axs.set_xlabel("$\\mathrm{M}/\\mathrm{M}_{\\odot}$")
    fig.suptitle(title)
    fig.set_size_inches(6, 5)
    fig.tight_layout()
    if save is not None:
        fig.savefig(save, dpi=150)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    a = catalog.fetch_json("../catalog_z.json")["sources"]
    af = catalog.rm_bad(a)
    afp = [s for s in af if s["grat"] == "prism"]
    afm = [s for s in af if s["grat"][-1] == "m" and s["grat"][0] == "g"]
    afh = [s for s in af if s["grat"][-1] == "h" and s["grat"][0] == "g"]
    path0 = "../Plots/"
    srs = {"medium": afm, "high": afh, "prism": afp}
    sources = {
        "npy": "../Data/Npy/",
        "ppxf": "../Data/Subtracted/",
        "smoo": "../Data/Subtracted_b/",
    }
    for i in range(2):
        plot_stacks(
            afp,
            lines_C,
            ite=i,
            zrang=[[7, 20]],
            save=f"../Plots/linesC/spectr_prism_{i}.png",
            title=f"Stack of lines in prism ({i} of 1)",
        )
        plot_stacks(
            afm,
            lines_C,
            ite=i,
            zrang=[[7, 20]],
            save=f"../Plots/linesC/spectr_medium_{i}.png",
            narrow=3,
            title=f"Stack of lines in medium resolution ({i} of 1)",
        )
        plot_stacks(
            afh,
            lines_C,
            ite=i,
            zrang=[[7, 20]],
            save=f"../Plots/linesC/spectr_high_{i}.png",
            narrow=5,
            title=f"Stack of lines in high resolution ({i} of 1)",
        )
    """
    for i in range(3):
        plot_stacks(
            afp,
            lines,
            ite=i,
            save=f"../Plots/lines4/spectr_prism_{i}.png",
            title=f"Stack of lines in prism ({i} of 2)",
        )
        plot_stacks(
            afm,
            lines,
            ite=i,
            save=f"../Plots/lines4/spectr_medium_{i}.png",
            narrow=3,
            title=f"Stack of lines in medium resolution ({i} of 2)",
        )
        plot_stacks(
            afh,
            lines,
            ite=i,
            save=f"../Plots/lines4/spectr_high_{i}.png",
            narrow=5,
            title=f"Stack of lines in high resolution ({i} of 2)",
        )
    plot_mz(afp, title='Mass vs Redshift for prism', save='../Plots/lines4/mz_prism.png')
    plot_mz(afm, title='Mass vs Redshift for medium resolution', save='../Plots/lines4/mz_medium.png')
    plot_mz(afh, title='Mass vs Redshift for high resolution', save='../Plots/lines4/mz_high.png')
    plot_histograms(
        afp,
        lines,
        title="Coverage of lines in prism",
        save="../Plots/lines4/hist_prism.png",
        ymax=3350,
    )
    plot_histograms(
        afm,
        lines,
        title="Coverage of lines in medium resolution",
        save="../Plots/lines4/hist_medium.png",
        ymax=1700,
        narrow=3,
    )
    plot_histograms(
        afh,
        lines,
        title="Coverage of lines in high resolution",
        save="../Plots/lines4/hist_high.png",
        ymax=630,
        narrow=5,
    )
    plot_lines(
        afp,
        lines,
        title="Coverage of lines in prism",
        save="../Plots/lines4/lines_prism.png",
        ratios = ratios,
    )
    plot_lines(
        afm,
        lines,
        title="Coverage of lines in medium resolution",
        save="../Plots/lines4/lines_medium.png",
        narrow=3,
        ratios = ratios,
    )
    plot_lines(
        afh,
        lines,
        title="Coverage of lines in high resolution",
        save="../Plots/lines4/lines_high.png",
        narrow=5,
        ratios = ratios,
    )
    """
