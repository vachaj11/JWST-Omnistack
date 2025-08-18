import matplotlib

matplotlib.use("qtagg")

import csv
import io
import time
from multiprocessing import Manager, Process, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pyneb as pn

import catalog
import line_fit as lf
import plots
import spectr

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Computer Modern",
        "font.size": 15,
    }
)


def S_S23(sources, **kwargs):
    fsii = fit_lines(sources, [0.6718, 0.6733], [0.6718, 0.6733], **kwargs)
    fsiii = fit_lines(sources, [0.9532, 0.9548], 0.9532, **kwargs)
    # fsiii += fit_lines(sources, [0.9069, 0.9071], 0.9063, **kwargs)
    fhbe = fit_lines(sources, [0.4862], 0.4862, **kwargs)
    S23 = np.log10((fsii + fsiii) / fhbe)
    if np.isfinite(S23):
        return [S23], [6.63 + 2.202 * S23 + 1.060 * S23**2]
    else:
        return [S23], []


def N_N2O2(sources, **kwargs):
    fnii = fit_lines(sources, [0.6550, 0.6564, 0.6585], 0.6585, **kwargs)
    foii = fit_lines(sources, [0.3727, 0.3729], [0.3727, 0.3729], **kwargs)
    N2O2 = np.log10(fnii / foii)
    if np.isfinite(N2O2):
        return [N2O2], [0.52 * N2O2 - 0.65]
    else:
        return [N2O2], []


def N_N2(sources, **kwargs):
    fnii = fit_lines(sources, [0.6550, 0.6564, 0.6585], 0.6585, **kwargs)
    fhal = fit_lines(sources, [0.6550, 0.6564, 0.6585], 0.6564, **kwargs)
    N2 = np.log10(fnii / fhal)
    if np.isfinite(N2):
        return [N2], [0.62 * N2 - 0.57]
    else:
        return [N2], []


def N_N2S2(sources, **kwargs):
    N2S2 = O_N2(sources, **kwargs)[0][0] - O_S2(sources, **kwargs)[0][0]
    if np.isfinite(N2S2):
        return [N2S2], [0.85 * N2S2 - 1.00]
    else:
        return [N2S2], []


def O_N2(sources, **kwargs):
    fnii = fit_lines(sources, [0.6550, 0.6564, 0.6585], 0.6585, **kwargs)
    fhal = fit_lines(sources, [0.6550, 0.6564, 0.6585], 0.6564, **kwargs)
    N2 = np.log10(fnii / fhal)
    if np.isfinite(N2):
        p = [-0.489 - N2, 1.513, -2.554, -5.293, -2.867]
        roots = [
            np.real(v) for v in np.roots(p) + 8.69 if np.isreal(v) and 7.6 < v < 8.9
        ]
        return [N2], roots
    else:
        return [N2], []


def O_R3(sources, **kwargs):
    foiii = fit_lines(sources, [0.5008], 0.5008, **kwargs)
    fhbe = fit_lines(sources, [0.4862], 0.4862, **kwargs)
    R3 = np.log10(foiii / fhbe)
    if np.isfinite(R3):
        p = [-0.277 - R3, -3.549, -3.593, -0.981]
        roots = [
            np.real(v) for v in np.roots(p) + 8.69 if np.isreal(v) and 7.6 < v < 8.9
        ]
        return [R3], roots
    else:
        return [R3], []


def O_O3N2(sources, **kwargs):
    O3N2 = O_R3(sources, **kwargs)[0][0] - O_N2(sources, **kwargs)[0][0]
    if np.isfinite(O3N2):
        p = [0.281 - O3N2, -4.765, -2.268]
        roots = [
            np.real(v) for v in np.roots(p) + 8.69 if np.isreal(v) and 7.6 < v < 8.9
        ]
        return [O3N2], roots
    else:
        return [O3N2], []


def O_O3S2(sources, **kwargs):
    O3S2 = O_R3(sources, **kwargs)[0][0] - O_S2(sources, **kwargs)[0][0]
    if np.isfinite(O3S2):
        p = [0.191 - O3S2, -4.292, -2.538, 0.053, 0.332]
        roots = [
            np.real(v) for v in np.roots(p) + 8.69 if np.isreal(v) and 7.6 < v < 8.9
        ]
        return [O3S2], roots
    else:
        return [O3S2], []


def O_S2(sources, **kwargs):
    fsii = fit_lines(sources, [0.6718, 0.6733], [0.6718, 0.6733], **kwargs)
    fhal = fit_lines(sources, [0.6550, 0.6564, 0.6585], 0.6564, **kwargs)
    S2 = np.log10(fsii / fhal)
    if np.isfinite(S2):
        p = [-0.442 - S2, -0.360, -6.271, -8.339, -3.559]
        roots = [
            np.real(v) for v in np.roots(p) + 8.69 if np.isreal(v) and 7.6 < v < 8.9
        ]
        return [S2], roots
    else:
        return [S2], []


def O_R2(sources, **kwargs):
    foii = fit_lines(sources, [0.3727, 0.3729], [0.3727, 0.3729], **kwargs)
    fhbe = fit_lines(sources, [0.4862], 0.4862, **kwargs)
    R2 = np.log10(foii / fhbe)
    if np.isfinite(R2):
        p = [0.435 - R2, -1.362, -5.655, -4.851, -0.478, 0.736]
        roots = [
            np.real(v) for v in np.roots(p) + 8.69 if np.isreal(v) and 7.6 < v < 8.9
        ]
        return [R2], roots
    else:
        return [R2], []


def O_O3O2(sources, **kwargs):
    O3O2 = O_R3(sources, **kwargs)[0][0] - O_R2(sources, **kwargs)[0][0]
    if np.isfinite(O3O2):
        p = [-0.691 - O3O2, -2.944, -1.308]
        roots = [
            np.real(v) for v in np.roots(p) + 8.69 if np.isreal(v) and 7.6 < v < 8.9
        ]
        return [O3O2], roots
    else:
        return [O3O2], []


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
        return [R23], roots
    else:
        return [R23], []


def O_RS32(sources, **kwargs):
    RS32 = np.log10(
        10 ** (O_R3(sources, **kwargs)[0][0]) + 10 ** (O_S2(sources, **kwargs)[0][0])
    )
    if np.isfinite(RS32):
        p = [-0.054 - RS32, -2.546, -1.970, 0.082, 0.222]
        roots = [
            np.real(v) for v in np.roots(p) + 8.69 if np.isreal(v) and 7.6 < v < 8.9
        ]
        return [RS32], roots
    else:
        return [RS32], []


def tem_den(fluxec):
    diagn = dict()
    if (fluxec["O3_4959A"] > 0 or fluxec["O3_5007A"] > 0) and fluxec["O3_4363A"] > 0:
        diagn["[OIII] 4363/5007+"] = fluxec["O3_4363A"] / (
            fluxec["O3_4959A"] + fluxec["O3_5007A"]
        )
    if (fluxec["O2_3726A"] > 0 or fluxec["O2_3729A"] > 0) and (
        fluxec["O2_7319A"] > 0 or fluxec["O2_7330A"] > 0
    ):
        diagn["[OII] 3727/7325"] = (fluxec["O2_3726A"] + fluxec["O2_3729A"]) / (
            fluxec["O2_7319A"] + fluxec["O2_7330A"]
        )
    if (fluxec["S2_6716A"] > 0 or fluxec["S2_6731A"] > 0) and (
        fluxec["S2_4069A"] > 0 or fluxec["S2_4076A"] > 0
    ):
        diagn["[SII] 4072+/6720+"] = (fluxec["S2_4069A"] + fluxec["S2_4076A"]) / (
            fluxec["S2_6716A"] + fluxec["S2_6731A"]
        )
    if fluxec["S2_6716A"] > 0 and fluxec["S2_6731A"] > 0:
        diagn["[SII] 6731/6716"] = fluxec["S2_6731A"] / fluxec["S2_6716A"]
    if (fluxec["S3_9531A"] > 0 or fluxec["S3_9069A"] > 0) and fluxec["S3_6312A"] > 0:
        diagn["[SIII] 6312/9200+"] = fluxec["S3_6312A"] / (
            fluxec["S3_9531A"] + fluxec["S3_9069A"]
        )
    if (fluxec["N2_6548A"] > 0 or fluxec["N2_6584A"] > 0) and fluxec["N2_5755A"] > 0:
        diagn["[NII] 5755/6584+"] = fluxec["N2_5755A"] / (
            fluxec["N2_6548A"] + fluxec["N2_6584A"]
        )

    ks = list(diagn.keys())
    diags = pn.Diagnostics()
    for k in ks:
        if k == "[OII] 3727/7325":
            diags.addDiag(
                "[OII] 3727/7325",
                (
                    "O2",
                    "(L(3726)+L(3729))/(L(7319)+L(7330))",
                    "RMS([E(3726)*L(3726)/(L(3726)+L(3729)), E(3729)*L(3729)/(L(3726)+L(3729)),E(7319)*L(7319)/(L(7319)+L(7330)),E(7330)*L(7330)/(L(7319)+L(7330))])",
                ),
            )
        else:
            diags.addDiag(k)
    tns = []
    dens = ["[SII] 4072+/6720+", "[SII] 6731/6716"]
    pairs = [(j, k) for j in ks for k in dens if k in ks and j not in dens]
    if not pairs and "[SII] 6731/6716" not in ks:
        diags.addDiag("[SII] 6731/6716")
        diagn["[SII] 6731/6716"] = 1
        pairs = [(j, "[SII] 6731/6716") for j in ks]
        # print("No SII lines could be used for density determination!")
    elif not pairs and "[SII] 6731/6716" in ks:
        diags.addDiag(
            "[OII] 3727/7325",
            (
                "O2",
                "(L(3726)+L(3729))/(L(7319)+L(7330))",
                "RMS([E(3726)*L(3726)/(L(3726)+L(3729)), E(3729)*L(3729)/(L(3726)+L(3729)),E(7319)*L(7319)/(L(7319)+L(7330)),E(7330)*L(7330)/(L(7319)+L(7330))])",
            ),
        )
        diagn["[OII] 3727/7325"] = 10
        pairs = [("[OII] 3727/7325", "[SII] 6731/6716")]
        # print("No non-SII lines could be used for density determination!")
    for p in pairs:
        t, n = diags.getCrossTemDen(p[0], p[1], diagn[p[0]], diagn[p[1]])
        if np.isfinite(t) and np.isfinite(n):
            tns.append((t, n))
    if tns:
        return np.nanmedian(tns, axis=0)
    else:
        return (15000, 500)


def abunds(flx, tem, den, N_over_O=False):
    ions = dict()
    abun = dict()

    O3 = pn.Atom("O", 3)
    O3ab = 0
    if flx["O3_4959A"] > 0 or flx["O3_5007A"] > 0:
        O3ab += O3.getIonAbundance(
            int_ratio=flx["O3_4959A"] + flx["O3_5007A"],
            tem=tem,
            den=den,
            to_eval="L(5007)+L(4959)",
            Hbeta=flx["H1r_4861A"],
        )
    ions["O3"] = O3ab

    O2 = pn.Atom("O", 2)
    O2ab = 0
    O2co = 0
    if flx["O2_3726A"] > 0 or flx["O2_3729A"] > 0:
        O2ab += O2.getIonAbundance(
            int_ratio=flx["O2_3726A"] + flx["O2_3729A"],
            tem=tem,
            den=den,
            to_eval="L(3726)+L(3729)",
            Hbeta=flx["H1r_4861A"],
        )
        O2co += 1
    if flx["O2_7319A"] > 0 or flx["O2_7330A"] > 0:
        O2ab += O2.getIonAbundance(
            int_ratio=flx["O2_7319A"] + flx["O2_7330A"],
            tem=tem,
            den=den,
            to_eval="L(7319)+L(7330)",
            Hbeta=flx["H1r_4861A"],
        )
        O2co += 1
    ions["O2"] = O2ab / O2co if O2co > 0 else 0

    S2 = pn.Atom("S", 2)
    S2ab = 0
    if flx["S2_6716A"] > 0 or flx["S2_6731A"] > 0:
        S2ab += S2.getIonAbundance(
            int_ratio=flx["S2_6731A"] + flx["S2_6716A"],
            tem=tem,
            den=den,
            to_eval="L(6731)+L(6716)",
            Hbeta=flx["H1r_4861A"],
        )
    ions["S2"] = S2ab

    S3 = pn.Atom("S", 3)
    S3ab = 0
    S3co = 0
    if flx["S3_9531A"] > 0 and flx["S3_9069A"] > 0:
        S3ab += S3.getIonAbundance(
            int_ratio=flx["S3_9531A"] + flx["S3_9069A"],
            tem=tem,
            den=den,
            to_eval="L(9531)+L(9069)",
            Hbeta=flx["H1r_4861A"],
        )
        S3co += 1
    if flx["S3_6312A"] > 0:
        S3ab += S3.getIonAbundance(
            int_ratio=flx["S3_6312A"],
            tem=tem,
            den=den,
            to_eval="L(6312)",
            Hbeta=flx["H1r_4861A"],
        )
        S3co += 1
    ions["S3"] = S3ab / S3co if S3co > 0 else 0

    N2 = pn.Atom("N", 2)
    N2ab = 0
    if flx["N2_6548A"] > 0 or flx["N2_6584A"] > 0:
        N2ab += N2.getIonAbundance(
            int_ratio=flx["N2_6548A"] + flx["N2_6584A"],
            tem=tem,
            den=den,
            to_eval="L(6548)+L(6584)",
            Hbeta=flx["H1r_4861A"],
        )
    ions["N2"] = N2ab

    b = lambda x: bool(x) and np.isfinite(x)
    icf = pn.ICF()
    if b(ions["O2"]) or b(ions["O3"]):
        abun["O"] = 12 + np.log10(
            icf.getElemAbundance(ions, icf_list="direct_O.23")["direct_O.23"]
        )
    else:
        abun["O"] = np.nan

    icf.addICF(
        "PMo_47_3",
        "S",
        '(abun["S2"]+abun["S3"])',
        '(1-(abun["O3"]/(abun["O2"]+abun["O3"]))**3.27)**(-1/3.27)',
    )
    if (b(ions["S2"]) or b(ions["S3"])) and (b(ions["O2"]) or b(ions["O3"])):
        abun["S"] = 12 + np.log10(
            icf.getElemAbundance(ions, icf_list="PMo_47_3")["PMo_47_3"]
        )
    else:
        abun["S"] = np.nan

    if b(ions["N2"]) and b(ions["O2"]):
        if not N_over_O:
            abun["N"] = 12 + np.log10(
                icf.getElemAbundance(ions, icf_list="TPP77_14")["TPP77_14"]
            )
        else:
            abun["N"] = np.log10(ions["N2"] / ions["O2"])
    else:
        abun["N"] = np.nan
    return abun


def abundances(sources, cal_red=None, N_over_O=False, **kwargs):
    lines = {
        "O3_4959A": (0.4959, [0.4959], 8),
        "O3_5007A": (0.5007, [0.5007], 8),
        "O3_4363A": (0.4363, [0.4364, 0.4372], 7),
        "O2_3726A": (0.3726, [0.3726, 0.3729], 7),
        "O2_3729A": (0.3729, [0.3726, 0.3729], 7),
        "O2_7319A": (0.7320, [0.7320, 0.7331], 8),
        "O2_7330A": (0.7331, [0.7320, 0.7331], 8),
        "H1r_4861A": (0.4862, [0.4862], 8),
        "H1r_6563A": (0.6564, [0.6550, 0.6564, 0.6585], 8),
        "H1r_4341A": (0.4341, [0.4341, 0.4364], 8),
        "H1r_4102A": (0.4102, [0.4102], 5),
        "S2_6716A": (0.6718, [0.6718, 0.6732], 7),
        "S2_6731A": (0.6732, [0.6718, 0.6732], 7),
        "S2_4069A": (0.4070, [0.4070, 0.4078], 3),
        "S2_4076A": (0.4078, [0.4070, 0.4078], 3),
        "S3_9531A": (0.9533, [0.9533], 8),
        "S3_9069A": (0.9071, [0.9071], 8),
        "S3_6312A": (0.6314, [0.6314], 2),
        "N2_6548A": (0.6548, [0.6548, 0.6565], 8),
        "N2_6584A": (0.6585, [0.6565, 0.6585], 8),
        "N2_5755A": (0.5757, [0.5757], 4),
    }
    if cal_red is None:
        cHbeta = red_const(sources)
    else:
        cHbeta = cal_red
    redcorr = pn.RedCorr(cHbeta=cHbeta, law="CCM89")
    lins = dict()
    fluxes = dict()
    fluxec = dict()
    for l in lines:
        flux = fit_lines(
            sources, lines[l][1], lines[l][0], dwidth=lines[l][2], cal_red=0, **kwargs
        )
        fluxes[l] = flux
        if flux > 0:
            wav = float(l.split("_")[1].split("A")[0])
            fluxec[l] = flux * redcorr.getCorrHb(wav)
        else:
            fluxec[l] = 0.0
    tem, den = tem_den(fluxec)
    ions = []
    abun = abunds(fluxec, tem, den, N_over_O=N_over_O)
    return abun


O_Dir = lambda s, **kwargs: [[abundances(s, **kwargs)["O"]]] * 2
S_Dir = lambda s, **kwargs: [[abundances(s, **kwargs)["S"]]] * 2
N_Dir = lambda s, **kwargs: [[abundances(s, N_over_O=True, **kwargs)["N"]]] * 2

Oxygen = {
    "N2": O_N2,
    "R3": O_R3,
    "O3N2": O_O3N2,
    "O3S2": O_O3S2,
    "S2": O_S2,
    "R2": O_R2,
    "O3O2": O_O3O2,
    "R23": O_R23,
    "RS32": O_RS32,
}
Nitrogen = {
    "N2O2": N_N2O2,
    "N2": N_N2,
    "N2S2": N_N2S2,
}
Sulphur = {"S23": S_S23}
Names = {
    "N2": "$\\mathrm{log}([\\mathrm{N}_\\mathrm{II}]\\lambda 6584/\\mathrm{H}_\\alpha)$",
    "R3": "$\\mathrm{log}([\\mathrm{O}_\\mathrm{III}]\\lambda 5007/\\mathrm{H}_\\beta)$",
    "O3N2": "$\\mathrm{R}3-\\mathrm{N}2$",
    "O3S2": "$\\mathrm{R}3-\\mathrm{S}2$",
    "S2": "$\\mathrm{log}([\\mathrm{S}_\\mathrm{II}]\\lambda\\lambda 6717, 6731/\\mathrm{H}_\\alpha)$",
    "R2": "$\\mathrm{log}([\\mathrm{O}_\\mathrm{II}]\\lambda 3727/\\mathrm{H}_\\beta)$",
    "O3O2": "$\\mathrm{log}([\\mathrm{O}_\\mathrm{III}]\\lambda 5007/[\\mathrm{O}_\\mathrm{II}]\\lambda 3727)$",
    "R23": "$\\mathrm{log}(([\\mathrm{O}_\\mathrm{II}]\\lambda 3727+[\\mathrm{O}_\\mathrm{III}]\\lambda 5007)/\\mathrm{H}_\\beta)$",
    "RS32": "$\\mathrm{log}(10^\\mathrm{R3}+10^\\mathrm{S2})$",
    "N2O2": "$\\mathrm{log}([\\mathrm{N}_\\mathrm{II}]\\lambda 6585/[\\mathrm{O}_\\mathrm{II}]\\lambda 3727)$",
    "N2": "$\\mathrm{log}([\\mathrm{N}_\\mathrm{II}]\\lambda 6585/[\\mathrm{H}_\\alpha)$",
    "N2S2": "$\\mathrm{log}([\\mathrm{N}_\\mathrm{II}]\\lambda 6585/[\\mathrm{S}_\\mathrm{II}]\\lambda\\lambda 6717, 6731)$",
    "S23": "$\\mathrm{log}(([\\mathrm{S}_\\mathrm{II}]\\lambda\\lambda 6716, 6731 + [\\mathrm{S}_\\mathrm{III}]\\lambda 9532)/\\mathrm{H}_\\beta)$",
}


def fit_lines(
    sources,
    lines,
    mline,
    reso=300,
    base="../Data/Npy/",
    typ="median",
    plot=False,
    cal_red=None,
    dwidth=8,
    manual=False,
    **kwargs,
):
    if cal_red is None:
        cal_red = red_const(sources)
    grats = [s["grat"] for s in sources]
    grat = max(set(grats), key=grats.count)
    rang, _ = lf.line_range(lines, grat=grat)
    sources = catalog.filter_zranges(sources, [rang])
    spectra, sourn = spectr.resampled_spectra(sources, rang, reso, base=base)
    if len(sourn):
        stack = spectr.combine_spectra(spectra)
        stacc = spectr.stack(stack, sourn, typ=typ)
        fit, x, stacc = lf.fit_lines(
            stacc, lines, grat=grat, dwidth=dwidth, manual=manual, mline=mline
        )
        if type(mline) == list:
            flux = sum([redd(lf.flux_at(fit, l), l, sources, cal_red) for l in mline])
        else:
            flux = redd(lf.flux_at(fit, mline), mline, sources, cal_red)
        if plot and not manual:
            plots.plot_fit(stacc, fit, sources=sourn, **kwargs)
        return flux
    else:
        return np.nan


def red_const(sources, T=12000):

    lT = np.log10(T)
    wavs = {
        6563.0: 10.35 - 3.254 * lT + 0.3457 * lT**2,
        4340.0: 0.0254 + 0.1922 * lT - 0.0204 * lT**2,
        4102.0: -0.07132 + 0.1436 * lT - 0.0153 * lT**2,
    }
    flub = fit_lines(sources, [0.4862], 0.4862, cal_red=0)
    flux = {
        6563.0: fit_lines(sources, [0.6550, 0.6564, 0.6585], 0.6564, cal_red=0) / flub,
        4340.0: fit_lines(sources, [0.4341, 0.4364], 0.4341, cal_red=0) / flub,
        4102.0: fit_lines(sources, [0.4102], 0.4102, cal_red=0) / flub,
    }
    consts = []
    for k in wavs:
        if flux[k] > 0:
            rc = pn.RedCorr(law="CCM89")
            rc.setCorr(obs_over_theo=flux[k] / wavs[k], wave1=k, wave2=4862.0)
            consts.append(rc.cHbeta)
    # print("cHbeta values: " + str(consts))
    return consts[0]


def redd(flux, line, sources, cal_red):
    if cal_red is None:
        cal_red = red_const(sources)
    else:
        rc = pn.RedCorr(law="CCM89", cHbeta=cal_red)
        return flux * rc.getCorrHb(line * 10**4)


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


def iprocess(funct, srs, ind, va, zs, vs, it, **kwargs):
    itl = 0
    for sr in srs:
        vi = funct([sr], **kwargs)[ind]
        if vi:
            zs.append([sr["z"]] * len(vi))
            vs.append([v for v in vi])
            va.append(vi)
            it.value += 1


def bprocess(funct, srs, ind, vis, **kwargs):
    for sr in srs:
        vi = np.flip(funct(sr, **kwargs)[ind])
        vis.append(vi)


def boots_stat(funct, sources, ite=100, calib=True, manual=True, **kwargs):
    ind = 1 if calib else 0
    vs = np.flip(funct(sources, manual=manual, **kwargs)[ind])
    vals = np.full((ite, len(vs)), np.nan)
    manag = Manager()
    vis = manag.list()
    i = 0
    nos = 5
    proc = int(cpu_count() * 1.5)
    active = []
    while i < ite or len(active) > 0:
        pr_no = -((i - ite) // nos)
        if proc > len(active) and i < ite:
            for l in range(min(pr_no, proc - len(active))):
                print(f"\r\033[KBootstrapping {i} out of {ite}.", end="")
                noi = min(nos, ite - i)
                rsourcs = [
                    np.random.choice(sources, size=len(sources)) for l in range(noi)
                ]
                args = (funct, rsourcs, ind, vis)
                t = Process(target=bprocess, args=args, kwargs=kwargs)
                t.start()
                active.append(t)
                i += noi
        for t in active:
            if not t.is_alive():
                t.terminate()
                active.remove(t)
        time.sleep(0.1)
    for i, vi in enumerate(vis):
        vals[i, : vi.shape[0]] = vi[: vals.shape[1]]
    vals = np.nan_to_num(vals, nan=np.nan, posinf=np.nan, neginf=np.nan)
    medians = np.nanmedian(vals, axis=0)
    err33 = np.nanpercentile(vals, 33, axis=0)
    err67 = np.nanpercentile(vals, 67, axis=0)
    return vs, medians, [medians - err33, err67 - medians]


def indiv_stat(funct, sources, no=None, calib=True, **kwargs):
    ind = 1 if calib else 0
    no = len(sources) if no is None else no
    srcs = np.random.choice(sources, size=len(sources), replace=False)
    manag = Manager()
    va = manag.list()
    zs = manag.list()
    vs = manag.list()
    i = 0
    nos = 50
    it = manag.Value(int, 0)
    proc = cpu_count() * 2
    active = []
    while it.value < no and i < len(srcs) or len(active) > 0:
        pr_no = -((-min(len(srcs) - i, no - it.value)) // nos)
        if proc > len(active) and (it.value < no and i < len(srcs)):
            for l in range(min(pr_no, proc - len(active))):
                print(f"\r\033[KCaclulating {i} our of {no} points.", end="")
                sr = srcs[i : i + nos]
                args = (funct, sr, ind, va, zs, vs, it)
                t = Process(target=iprocess, args=args, kwargs=kwargs)
                t.start()
                active.append(t)
                i += len(sr)
        for t in active:
            if not t.is_alive():
                t.terminate()
                active.remove(t)
        time.sleep(0.1)
    va = list(va)
    zs = sum(list(zs), [])
    vs = sum(list(vs), [])
    if len(va):
        vals = np.full((len(va), max([len(v) for v in va])), np.nan)
        for l in range(len(va)):
            vi = np.flip(va[l])
            vals[l, : vi.shape[0]] = vi[: vals.shape[1]]
        vals = np.nan_to_num(vals, nan=np.nan, posinf=np.nan, neginf=np.nan)
        medians = np.nanmedian(vals, axis=0)
        err33 = np.nanpercentile(vals, 33, axis=0)
        err67 = np.nanpercentile(vals, 67, axis=0)
    else:
        medians, err33, err67 = [np.array([np.nan])] * 3
    print(f"Needed {i} out of {len(srcs)} for {it.value} results.")
    return (zs, vs), (
        medians,
        np.nan_to_num([medians - err33, err67 - medians], nan=0.0),
    )


flatten = lambda l: sum(map(flatten, list(l)), []) if hasattr(l, "__iter__") else [l]


def abundance_in_z(
    sources, zrangs, abund=Oxygen, save=None, title=None, yax=None, indiv=True, **kwargs
):
    n = int(-(-np.sqrt(len(abund)) // 1))
    fig = plt.figure()
    gs = fig.add_gridspec(n, n, hspace=0, wspace=0)
    axes = gs.subplots(sharex="col", sharey="row")
    yrang = []
    if n == 1:
        axs = [axes]
    else:
        axs = axes.flatten()
    valss = []
    for i, (nam, ab) in enumerate(abund.items()):
        for zrang in zrangs:
            sourz = [s for s in sources if zrang[0] < s["z"] < zrang[1]]
            cal_red = red_const(sourz)
            if indiv:
                ind, _ = indiv_stat(ab, sourz, cal_red=cal_red, **kwargs)
                if ind[0]:
                    axs[i].plot(
                        ind[0],
                        ind[1],
                        ls="",
                        marker=".",
                        c="gray",
                        alpha=0.15,
                        markersize=2,
                    )
            v, m, st = boots_stat(ab, sourz, cal_red=None, **kwargs)
            if v.size:
                zmean = [(zrang[1] + zrang[0]) / 2] * len(v)
                zerr = [(zrang[1] - zrang[0]) / 2] * len(v)
                axs[i].plot(zmean, v, ls="", marker="D", c="black")
                axs[i].errorbar(zmean, v, xerr=zerr, ls="", c="black", capsize=5)
                axs[i].errorbar(zmean, m, yerr=st, ls="", c="black", capsize=5)
                yrang += [np.nanmin(m - st[0]), np.nanmax(m + st[1]), v]
            valss = np.concatenate([valss, v])
        axs[i].set_title(nam, y=0.85)
    for i in range(len(axs) - len(abund)):
        axs[len(abund) + i].tick_params(axis="y", left=False, labelleft=False)
    minz = min([min(zr) for zr in zrangs])
    maxz = max([max(zr) for zr in zrangs])
    ranz = maxz - minz
    minz = minz - ranz * 0.1
    maxz = maxz + ranz * 0.1
    yrang = np.nan_to_num(flatten(yrang), nan=np.nan, posinf=np.nan, neginf=np.nan)
    ylims = (np.nanmin(yrang), np.nanmax(yrang))
    rana = ylims[1] - ylims[0]
    mina = ylims[0] - rana * 0.1
    maxa = ylims[1] + rana * 0.25
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


def ratios_in_z(
    sources, zrangs, abund=Oxygen, save=None, title=None, indiv=True, **kwargs
):
    n = int(-(-np.sqrt(len(abund)) // 1))
    fig = plt.figure()
    gs = fig.add_gridspec(n, n, hspace=0)
    axes = gs.subplots(sharex="col")
    if n == 1:
        axs = [axes]
    else:
        axs = axes.flatten()
    valss = []
    for i, (nam, ab) in enumerate(abund.items()):
        yrang = []
        for zrang in zrangs:
            sourz = [s for s in sources if zrang[0] < s["z"] < zrang[1]]
            cal_red = red_const(sourz)
            if indiv:
                ind, sta = indiv_stat(ab, sourz, calib=False, cal_red=cal_red, **kwargs)
                if ind[0]:
                    zmean = [(zrang[1] + zrang[0]) / 2] * len(sta[0])
                    zerr = [(zrang[1] - zrang[0]) / 2] * len(sta[0])
                    axs[i].plot(
                        ind[0],
                        ind[1],
                        ls="",
                        marker=".",
                        c="gray",
                        alpha=0.15,
                        markersize=2,
                    )
                    axs[i].errorbar(
                        zmean,
                        sta[0],
                        yerr=sta[1],
                        xerr=zerr,
                        ls="",
                        c="gainsboro",
                        capsize=5,
                    )
                    yrang += [
                        np.nanmin(sta[0] - sta[1][0]),
                        np.nanmax(sta[0] + sta[1][1]),
                    ]
            v, m, st = boots_stat(ab, sourz, calib=False, cal_red=None, **kwargs)
            if v.size:
                zmean = [(zrang[1] + zrang[0]) / 2] * len(v)
                zerr = [(zrang[1] - zrang[0]) / 2] * len(v)
                axs[i].plot(zmean, v, ls="", marker="D", c="black")
                axs[i].errorbar(zmean, v, xerr=zerr, ls="", c="black", capsize=5)
                axs[i].errorbar(zmean, m, yerr=st, ls="", c="black", capsize=5)
                axs[i].set_ylabel(Names[nam], fontsize=11)
                axs[i].yaxis.set_tick_params(labelsize=11)
                yrang += [np.nanmin(m - st[0]), np.nanmax(m + st[1]), v]
            valss = np.concatenate([valss, v])
        yrang = np.nan_to_num(flatten(yrang), nan=np.nan, posinf=np.nan, neginf=np.nan)
        ylims = (np.nanmin(yrang), np.nanmax(yrang))
        axs[i].set_title(nam, y=0.85)
        axs[i].set_ylim(
            ylims[0] - 0.1 * (ylims[1] - ylims[0]),
            ylims[1] + 0.25 * (ylims[1] - ylims[0]),
        )
    for i in range(len(axs) - len(abund)):
        axs[len(abund) + i].tick_params(axis="y", left=False, labelleft=False)
    minz = min([min(zr) for zr in zrangs])
    maxz = max([max(zr) for zr in zrangs])
    ranz = maxz - minz
    minz = minz - ranz * 0.1
    maxz = maxz + ranz * 0.1
    for i in range(n):
        axs[-i - 1].set_xlim(minz, maxz)
        axs[-i - 1].set_xlabel("$z$")
    fig.set_size_inches((max(n, 2) + 0.3) * 2.5, (max(n, 2) + 0.4) * 2.5)
    fig.suptitle(title)
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)
    else:
        plt.show()
    plt.close(fig)


def abundance_calib(
    sources,
    xmetr=O_Dir,
    abund=Oxygen,
    binval="z",
    bins=10,
    save=None,
    title=None,
    xax=None,
    indiv=False,
    **kwargs,
):
    n = int(-(-np.sqrt(len(abund)) // 1))
    fig = plt.figure()
    gs = fig.add_gridspec(n, n, hspace=0)
    axes = gs.subplots(sharex="col")
    xrang = []
    if n == 1:
        axs = [axes]
    else:
        axs = axes.flatten()

    sbins = catalog.inbins(sources, binval, nbin=bins)
    xbins = []
    for sb in sbins:
        xbins.append((sb, boots_stat(xmetr, sb, **kwargs)))
    valss = []
    for i, (nam, ab) in enumerate(abund.items()):
        yrang = []
        for sb in xbins:
            sourz = sb[0]
            cal_red = red_const(sourz)
            if indiv:
                ind, _ = indiv_stat(ab, sourz, calib=False, cal_red=cal_red, **kwargs)
                if ind[0]:
                    axs[i].plot(
                        ind[0],
                        ind[1],
                        ls="",
                        marker=".",
                        c="gray",
                        alpha=0.15,
                        markersize=2,
                    )
            v, m, st = boots_stat(ab, sourz, calib=False, cal_red=None, **kwargs)
            if v.size:
                zv = [sb[1][0][0]] * len(v)
                zm = [sb[1][1][0]] * len(v)
                zerr = [[sb[1][2][0][0]] * len(v), [sb[1][2][1][0]] * len(v)]
                axs[i].plot(zv, v, ls="", marker="D", c="black")
                axs[i].errorbar(zm, m, xerr=zerr, yerr=st, ls="", c="black", capsize=5)
                axs[i].set_ylabel(Names[nam], fontsize=11)
                axs[i].yaxis.set_tick_params(labelsize=11)
                yrang += [np.nanmin(m - st[0]), np.nanmax(m + st[1])]
                xrang += [zm[0] - zerr[0][0], zm[0] + zerr[1][0]]
            valss = np.concatenate([valss, v])
        yrang = np.nan_to_num(flatten(yrang), nan=np.nan, posinf=np.nan, neginf=np.nan)
        ylims = (np.nanmin(yrang), np.nanmax(yrang))
        axs[i].set_title(nam, y=0.85)
        axs[i].set_ylim(
            ylims[0] - 0.1 * (ylims[1] - ylims[0]),
            ylims[1] + 0.25 * (ylims[1] - ylims[0]),
        )
    for i in range(len(axs) - len(abund)):
        axs[len(abund) + i].tick_params(axis="y", left=False, labelleft=False)

    xrang = np.nan_to_num(flatten(xrang), nan=np.nan, posinf=np.nan, neginf=np.nan)
    xlims = (np.nanmin(xrang), np.nanmax(xrang))
    ranx = xlims[1] - xlims[0]
    minx = xlims[0] - ranx * 0.1
    maxx = xlims[1] + ranx * 0.1
    for i in range(n):
        axs[-i - 1].set_xlim(minx, maxx)
        axs[-i - 1].set_xlabel(xax)
    fig.set_size_inches((max(n, 2) + 0.3) * 2.5, (max(n, 2) + 0.4) * 2.5)
    fig.suptitle(title)
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)
    else:
        plt.show()
    plt.close(fig)


def abundance_compar(
    sources,
    xmetr=O_Dir,
    abund=Oxygen,
    binval="z",
    bins=10,
    save=None,
    title=None,
    yax=None,
    xax=None,
    indiv=False,
    **kwargs,
):
    n = int(-(-np.sqrt(len(abund)) // 1))
    xax = yax if xax is None else xax
    yax = xax if yax is None else yax
    fig = plt.figure()
    gs = fig.add_gridspec(n, n, hspace=0, wspace=0)
    axes = gs.subplots(sharex="col", sharey="row")
    yrang = []
    xrang = []
    if n == 1:
        axs = [axes]
    else:
        axs = axes.flatten()

    sbins = catalog.inbins(sources, binval, nbin=bins)
    xbins = []
    for sb in sbins:
        xbins.append((sb, boots_stat(xmetr, sb, **kwargs)))
    valss = []
    for i, (nam, ab) in enumerate(abund.items()):
        for sb in xbins:
            sourz = sb[0]
            cal_red = red_const(sourz)
            if indiv:
                ind, _ = indiv_stat(ab, sourz, cal_red=cal_red, **kwargs)
                if ind[0]:
                    axs[i].plot(
                        ind[0],
                        ind[1],
                        ls="",
                        marker=".",
                        c="gray",
                        alpha=0.15,
                        markersize=2,
                    )
            v, m, st = boots_stat(ab, sourz, cal_red=None, **kwargs)
            """
            print('\n=====')
            print(sb[1][0], sb[1][1])
            print((v, m))
            print('=====')
            """
            if v.size:
                zv = [sb[1][0][0]] * len(v)
                zm = [sb[1][1][0]] * len(v)
                zerr = [[sb[1][2][0][0]] * len(v), [sb[1][2][1][0]] * len(v)]
                axs[i].plot(zv, v, ls="", marker="D", c="black")
                axs[i].errorbar(zm, m, xerr=zerr, yerr=st, ls="", c="black", capsize=5)
                yrang += [np.nanmin(m - st[0]), np.nanmax(m + st[1])]
                xrang += [zm[0] - zerr[0][0], zm[0] + zerr[1][0]]
            valss = np.concatenate([valss, v])
        axs[i].set_title(nam, y=0.85)
    for i in range(len(axs) - len(abund)):
        axs[len(abund) + i].tick_params(axis="y", left=False, labelleft=False)

    yrang = np.nan_to_num(flatten(yrang), nan=np.nan, posinf=np.nan, neginf=np.nan)
    ylims = (np.nanmin(yrang), np.nanmax(yrang))
    rana = ylims[1] - ylims[0]
    mina = ylims[0] - rana * 0.1
    maxa = ylims[1] + rana * 0.25
    xrang = np.nan_to_num(flatten(xrang), nan=np.nan, posinf=np.nan, neginf=np.nan)
    xlims = (np.nanmin(xrang), np.nanmax(xrang))
    ranx = xlims[1] - xlims[0]
    minx = xlims[0] - ranx * 0.1
    maxx = xlims[1] + ranx * 0.1
    for i in range(n):
        axs[-i - 1].set_xlim(minx, maxx)
        axs[i * n].set_ylim(mina, maxa)
        axs[-i - 1].set_xlabel(xax)
        axs[i * n].set_ylabel(yax)
    maxc = max(maxx, maxa)
    minc = min(minx, mina)
    for ax in axs:
        ax.axline((0, 0), (1, 1), c="gray", ls="--", alpha=0.7)
        ax.set_xlim(minc, maxc)
        ax.set_ylim(minc, maxc)
    fig.set_size_inches((max(n, 2) + 0.3) * 2.5, (max(n, 2) + 0.4) * 2.5)
    fig.suptitle(title)
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    f = catalog.fetch_json("../catalog_z.json")["sources"]
    ff = catalog.rm_bad(f)
    ffm = [s for s in ff if s["grat"][0] == "g" and s["grat"][-1] == "m"]
    """
    abundance_in_z(
        ffm,
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6.5], [6.5, 8], [8, 12]],
        abund=Sulphur,
        title="Sulphur abundance in medium resolution\n via different calibrations",
        yax="$12+\\mathrm{log}(\\mathrm{S}/\\mathrm{H})$",
        save="../Plots/abund/sulphur_cal.png",
    )
    
    abundance_in_z(
        ffm,
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6.5], [6.5, 8], [8, 12]],
        abund=Nitrogen,
        title="Nitrogen abundance in medium resolution\n via different calibrations",
        yax="$\\mathrm{log}(\\mathrm{N}/\\mathrm{O})$",
        save="../Plots/abund/nitrogen_cal.png",
    )
    abundance_in_z(
        ffm,
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6.5], [6.5, 8], [8, 12]],
        abund=Oxygen,
        title="Oxygen abundance in medium resolution via different calibrations",
        yax="$12+\\mathrm{log}(\\mathrm{O}/\\mathrm{H})$",
        save="../Plots/abund/oxygen_cal.png",
    )

    ratios_in_z(
        ffm,
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6.5], [6.5, 8], [8, 12]],
        abund=Sulphur,
        title="Line fluxes for sulphur abundance calibration\n in medium resolution",
        save="../Plots/abund/sulphur_flu.png",
    )
    ratios_in_z(
        ffm,
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6.5], [6.5, 8], [8, 12]],
        abund=Nitrogen,
        title="Line fluxes for nitrogen abundance calibration\n in medium resolution",
        save="../Plots/abund/nitrogen_flu.png",
    )
    ratios_in_z(
        ffm,
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6.5], [6.5, 8], [8, 12]],
        abund=Oxygen,
        title="Line fluxes for oxygen abundance calibration in medium resolution",
        save="../Plots/abund/oxygen_flu.png",
    )
    """
    abundance_compar(
        ffm,
        xmetr=S_Dir,
        abund=Sulphur,
        binval="z",
        bins=10,
        save="../Plots/abund/sulphur_com.png",
        title="Sulphur abundance in medium resolution\n via direct method and strong lines",
        yax="$12+\\mathrm{log}(\\mathrm{S}/\\mathrm{H})$",
    )

    abundance_compar(
        ffm,
        xmetr=N_Dir,
        abund=Nitrogen,
        binval="z",
        bins=10,
        save="../Plots/abund/nitrogen_com.png",
        title="Nitrogen abundance in medium resolution\n via direct method and strong lines",
        yax="$\\mathrm{log}(\\mathrm{N}/\\mathrm{O})$",
    )
    abundance_compar(
        ffm,
        xmetr=O_Dir,
        abund=Oxygen,
        binval="z",
        bins=10,
        save="../Plots/abund/oxygen_com.png",
        title="Oxygen abundance in medium resolution\n via direct method and strong lines",
        yax="$12+\\mathrm{log}(\\mathrm{O}/\\mathrm{H})$",
    )

    abundance_calib(
        ffm,
        xmetr=S_Dir,
        abund=Sulphur,
        binval="z",
        bins=10,
        save="../Plots/abund/sulphur_clf.png",
        title="Sulphur abundance in medium resolution\n via compared to broad line calibrations",
        xax="$12+\\mathrm{log}(\\mathrm{S}/\\mathrm{H})$",
    )

    abundance_calib(
        ffm,
        xmetr=N_Dir,
        abund=Nitrogen,
        binval="z",
        bins=10,
        save="../Plots/abund/nitrogen_clf.png",
        title="Nitrogen abundance in medium resolution\n via compared to broad line calibrations",
        xax="$\\mathrm{log}(\\mathrm{N}/\\mathrm{O})$",
    )
    abundance_calib(
        ffm,
        xmetr=O_Dir,
        abund=Oxygen,
        binval="z",
        bins=10,
        save="../Plots/abund/oxygen_clf.png",
        title="Oxygen abundance in medium resolution\n via compared to broad line calibrations",
        xax="$12+\\mathrm{log}(\\mathrm{O}/\\mathrm{H})$",
    )
    """
    abundance_in_z(
        ffm,
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6.5], [6.5, 8], [8, 12]],
        abund={"S Direct": S_Dir},
        title="Sulphur abundance in medium resolution\n via direct method",
        yax="$12+\\mathrm{log}(\\mathrm{S}/\\mathrm{H})$",
        save="../Plots/abund/sulphur_dir.png",
        indiv=False,
    )
    
    abundance_in_z(
        ffm,
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6.5], [6.5, 8], [8, 12]],
        abund={"N Direct": N_Dir},
        title="Nitrogen abundance in medium resolution\n via direct method",
        yax="$12+\\mathrm{log}(\\mathrm{N}/\\mathrm{H})$",
        save="../Plots/abund/nitrogen_dir.png",
        indiv = False,
    )
    abundance_in_z(
        ffm,
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6.5], [6.5, 8], [8, 12]],
        abund={"O Direct": O_Dir},
        title="Oxygen abundance in medium resolution\n via direct method",
        yax="$12+\\mathrm{log}(\\mathrm{O}/\\mathrm{H})$",
        save="../Plots/abund/oxygen_dir.png",
        indiv = False,
    )
    """
