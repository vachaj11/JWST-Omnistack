import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import load_npz

import abundc as ac
import catalog
import paramite as pr

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Computer Modern",
        "font.size": 15,
    }
)

flatten = lambda l: sum(map(flatten, list(l)), []) if hasattr(l, "__iter__") else [l]


def reconst_comparison(
    xdata, ydata, line, yax=None, xax=None, lim=None, title="_", save=None
):
    xax = yax if xax is None else xax
    yax = xax if yax is None else yax

    fig = plt.figure(figsize=(6, 5.5))
    axs = plt.gca()

    xvals = xdata.get(line)
    yvals = ydata.get(line)
    axs.plot(xvals, yvals, ls="", marker=".", ms=1.5, c="black", alpha=0.3)

    crang = np.nan_to_num(
        flatten(xvals) + flatten(yvals), nan=np.nan, posinf=np.nan, neginf=np.nan
    )
    clims = (min(np.nanmin(crang), 0), np.nanpercentile(crang, 90))
    ranc = clims[1] - clims[0]
    minc = clims[0] - ranc * 0.1 if lim is None else lim[0]
    maxc = clims[1] + ranc * 0.1 if lim is None else lim[1]
    axs.axline((0, 0), (1, 1), c="gray", ls="--", lw=1, alpha=0.7)
    axs.set_xlim(minc, maxc)
    axs.set_ylim(minc, maxc)
    axs.set_xlabel(xax)
    axs.set_ylabel(yax)

    fig.suptitle(title)
    fig.set_layout_engine(layout="tight")

    if save is not None:
        fig.savefig(save)
    else:
        plt.show()
    plt.close(fig)


def get_I_R(sources, nam):
    tup = ac.core_lines[nam]
    M, fl, so = pr.art_fluxes(sources, tup, n_one=200, n_sam=250000)
    I = dict()
    R = dict()
    for n, m in {"OSEM": pr.OSEM, "MART": pr.MART, "FIST": pr.FIST}.items():
        method = (lambda M, f: m(M, f, t_f=1800),)
        R[n] = method(M, fl)
    Ind = ac.indiv_stat(
        (lambda x: ac.core_fit(x, nam, cal_red=0, indiv=False)), sources, calib=None
    )
    for n in tup[0]:
        vls = [s[n] for s in Ind]
        I[n] = vls
    return I, R


def main():
    f = catalog.fetch_json("../catalog_v4.json")["sources"]
    ff = catalog.rm_bad(f)
    ffm = [s for s in ff if s["grat"][0] == "g"]

    I, R = get_I_R(ffm, "H1_a")

    reconst_comparison(
        R["FIST"],
        I,
        "N2_6584A",
        save="../Plots/recon/compar_INDI.pdf",
        title="Comparison of fluxes reconstructed via \\textit{FISTA}\nand from un-stacked measurements",
        xax="\\textit{FISTA} $[\\mathrm{N}_\\mathrm{II}]\\lambda6585$",
        yax="Un-stacked $[\\mathrm{N}_\\mathrm{II}]\\lambda6585$",
        lim=[0, 5 * 10 ** (-16)],
    )
    reconst_comparison(
        R["FIST"],
        R["OSEM"],
        "N2_6584A",
        save="../Plots/recon/compar_OSEM.pdf",
        title="Comparison of fluxes reconstructed via \\textit{FISTA}\nand via \\textit{OSEM}",
        xax="\\textit{FISTA} $[\\mathrm{N}_\\mathrm{II}]\\lambda6585$",
        yax="\\textit{OSEM} $[\\mathrm{N}_\\mathrm{II}]\\lambda6585$",
        lim=[0, 5 * 10 ** (-16)],
    )
    reconst_comparison(
        R["FIST"],
        R["MLEM"],
        "N2_6584A",
        save="../Plots/recon/compar_MLEM.pdf",
        title="Comparison of fluxes reconstructed via \\textit{FISTA}\nand via \\textit{MLEM}",
        xax="\\textit{FISTA} $[\\mathrm{N}_\\mathrm{II}]\\lambda6585$",
        yax="\\textit{MLEM} $[\\mathrm{N}_\\mathrm{II}]\\lambda6585$",
        lim=[0, 5 * 10 ** (-16)],
    )
    reconst_comparison(
        R["FIST"],
        R["MART"],
        "N2_6584A",
        save="../Plots/recon/compar_MART.pdf",
        title="Comparison of fluxes reconstructed via \\textit{FISTA}\nand via \\textit{MART}",
        xax="\\textit{FISTA} $[\\mathrm{N}_\\mathrm{II}]\\lambda6585$",
        yax="\\textit{MART} $[\\mathrm{N}_\\mathrm{II}]\\lambda6585$",
        lim=[0, 5 * 10 ** (-16)],
    )
    """
    M = load_npz('../M_ne.npz')
    F = np.load('../F_ne.npy', allow_pickle=True).item()
    t_f=60
    RT = dict()
    #RT['PART'] = pr.PART(M, F, t_f=t_f)
    #RT['MART'] = pr.MART(M, F, t_f=t_f)
    #RT['MLEM'] = pr.MLEM(M, F, t_f=t_f)
    RT['OSEM'] = pr.OSEM(M, F, t_f=t_f)
    RT['FIST'] = pr.FIST(M, F, t_f=t_f)

    reconst_comparison(
        R['FIST'], 
        RT['FIST'], 
        'N2_6584A',
        save="../Plots/recon/compar_FIST_speed.pdf",
        title="Comparison of fluxes reconstructed via \\textit{FISTA}\nand via \\textit{FISTA}",
        xax="\\textit{FISTA} $[\\mathrm{N}_\\mathrm{II}]\\lambda6585$",
        yax="\\textit{FISTA} $[\\mathrm{N}_\\mathrm{II}]\\lambda6585$",
        lim=[0,5*10**(-16)]
    )    
    reconst_comparison(
        R['FIST'], 
        RT['OSEM'], 
        'N2_6584A',
        save="../Plots/recon/compar_OSEM_speed.pdf",
        title="Comparison of fluxes reconstructed via \\textit{FISTA}\nand via \\textit{OSEM}",
        xax="\\textit{FISTA} $[\\mathrm{N}_\\mathrm{II}]\\lambda6585$",
        yax="\\textit{OSEM} $[\\mathrm{N}_\\mathrm{II}]\\lambda6585$",
        lim=[0,5*10**(-16)]
    )
    reconst_comparison(
        R['FIST'], 
        RT['MLEM'], 
        'N2_6584A',
        save="../Plots/recon/compar_MLEM_speed.pdf",
        title="Comparison of fluxes reconstructed via \\textit{FISTA}\nand via \\textit{MLEM}",
        xax="\\textit{FISTA} $[\\mathrm{N}_\\mathrm{II}]\\lambda6585$",
        yax="\\textit{MLEM} $[\\mathrm{N}_\\mathrm{II}]\\lambda6585$",
        lim=[0,5*10**(-16)]
    )
    reconst_comparison(
        R['FIST'], 
        RT['MART'], 
        'N2_6584A',
        save="../Plots/recon/compar_MART_speed.pdf",
        title="Comparison of fluxes reconstructed via \\textit{FISTA}\nand via \\textit{MART}",
        xax="\\textit{FISTA} $[\\mathrm{N}_\\mathrm{II}]\\lambda6585$",
        yax="\\textit{MART} $[\\mathrm{N}_\\mathrm{II}]\\lambda6585$",
        lim=[0,5*10**(-16)]
    )
    """


if __name__ == "__main__":
    main()
