import time
from multiprocessing import Manager, Process, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pyneb as pn
from astropy.cosmology import LambdaCDM

import catalog
import line_fit as lf
import spectr

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Computer Modern",
        "font.size": 15,
    }
)

# cosmological parameters from Planck 2018
H0 = 67.49
Om0 = 0.315
Od0 = 0.6847

# astropy cosmology
cosm = LambdaCDM(H0, Om0, Od0)

## Universal extinction correction class. Instantiated outside other functions for added speed
rc = pn.RedCorr(law="CCM89")


def S_S23(sources, new=False, **kwargs):
    if len(sources) == 1 and "rec_S_S23"+"_n"*new in sources[0].keys():
        return sources[0]["rec_S_S23"+"_n"*new]
    fsii = fit_lines(
        sources, [0.6718, 0.6732], {"S2_6716A": 0.6717, "S2_6731A": 0.6732}, **kwargs
    )
    fsii = sum(fsii.values())
    fsiii = fit_lines(sources, [0.9533, 0.9551], {"S3_9531A": 0.9532}, **kwargs)
    fsiii = sum(fsiii.values())
    # fsiii += fit_lines(sources, [0.9069, 0.9071], 0.9063, **kwargs)
    fhbe = fit_lines(sources, [0.4862], {"H1r_4861A": 0.4862}, **kwargs)
    fhbe = sum(fhbe.values())
    S23 = np.log10((fsii + fsiii) / fhbe)
    if np.isfinite(S23):
        return [S23], [6.63 + 2.202 * S23 + 1.060 * S23**2]
    else:
        return [S23], []


def N_N2O2(sources, new=True, **kwargs):
    if len(sources) == 1 and "rec_N_N2O2"+"_n"*new in sources[0].keys():
        return sources[0]["rec_N_N2O2"+"_n"*new]
    fnii = fit_lines(sources, [0.6550, 0.6564, 0.6585], {"N2_6584A": 0.6585}, **kwargs)
    fnii = sum(fnii.values())
    foii = fit_lines(
        sources, [0.3726, 0.3729], {"O2_3726A": 0.3726, "O2_3729A": 0.3729}, **kwargs
    )
    foii = sum(foii.values())
    N2O2 = np.log10(fnii / foii)
    if np.isfinite(N2O2):
        if new:
            return [N2O2], [0.69 * N2O2 - 0.65]
        else:
            return [N2O2], [0.52 * N2O2 - 0.65]
    else:
        return [N2O2], []


def N_N2(sources, new=False, **kwargs):
    if len(sources) == 1 and "rec_N_N2"+"_n"*new in sources[0].keys():
        return sources[0]["rec_N_N2"+"_n"*new]
    ff = fit_lines(
        sources,
        [0.6550, 0.6564, 0.6585],
        {"N2_6584A": 0.6585, "H1r_6563A": 0.6564},
        **kwargs,
    )
    fnii = ff["N2_6584A"]
    fhal = ff["H1r_6563A"]
    N2 = np.log10(fnii / fhal)
    if np.isfinite(N2):
        return [N2], [0.62 * N2 - 0.57]
    else:
        return [N2], []


def N_N2S2(sources, new = True, **kwargs):
    if len(sources) == 1 and "rec_N_N2S2"+"_n"*new in sources[0].keys():
        return sources[0]["rec_N_N2S2"+"_n"*new]
    N2S2 = O_N2(sources, **kwargs)[0][0] - O_S2(sources, **kwargs)[0][0]
    if np.isfinite(N2S2):
        if new:
            return [N2S2], [1.12 * N2S2 - 0.93]
        else:
            return [N2S2], [0.85 * N2S2 - 1.00]
    else:
        return [N2S2], []


def O_N2(sources, new=True, **kwargs):
    if len(sources) == 1 and "rec_O_N2"+"_n"*new in sources[0].keys():
        return sources[0]["rec_O_N2"+"_n"*new]
    ff = fit_lines(
        sources,
        [0.6550, 0.6564, 0.6585],
        {"N2_6584A": 0.6585, "H1r_6563A": 0.6564},
        **kwargs,
    )
    fnii = ff["N2_6584A"]
    fhal = ff["H1r_6563A"]
    N2 = np.log10(fnii / fhal)
    if np.isfinite(N2):
        if new:
            p = [-1.356 - N2, 1.532]
            roots = [
                np.real(v) for v in np.roots(p) + 8.0 if np.isreal(v) and 7.8 < v < 8.6
            ]
            return [N2], roots
        else:
            p = [-0.489 - N2, 1.513, -2.554, -5.293, -2.867]
            roots = [
                np.real(v) for v in np.roots(p) + 8.69 if np.isreal(v) and 7.6 < v < 8.9
            ]
            return [N2], roots
    else:
        return [N2], []


def O_R3(sources, new = True, **kwargs):
    if len(sources) == 1 and "rec_O_R3"+"_n"*new in sources[0].keys():
        return sources[0]["rec_O_R3"+"_n"*new]
    foiii = fit_lines(sources, [0.5007], {"O3_5007A": 0.5007}, **kwargs)
    foiii = sum(foiii.values())
    fhbe = fit_lines(sources, [0.4862], {"H1r_4861A": 0.4862}, **kwargs)
    fhbe = sum(fhbe.values())
    R3 = np.log10(foiii / fhbe)
    if np.isfinite(R3):
        if new:
            p = [0.852 - R3, -0.162, -1.149, -0.553]
            roots = [
                np.real(v) for v in np.roots(p) + 8.0 if np.isreal(v) and 7.3 < v < 8.6
            ]
            return [R3], roots
        else:
            p = [-0.277 - R3, -3.549, -3.593, -0.981]
            roots = [
                np.real(v) for v in np.roots(p) + 8.69 if np.isreal(v) and 7.6 < v < 8.9
            ]
            return [R3], roots
    else:
        return [R3], []


def O_O3N2(sources, new=True, **kwargs):
    if len(sources) == 1 and "rec_O_O3N2"+"_n"*new in sources[0].keys():
        return sources[0]["rec_O_O3N2"+"_n"*new]
    O3N2 = O_R3(sources, **kwargs)[0][0] - O_N2(sources, **kwargs)[0][0]
    if np.isfinite(O3N2):
        if new:
            p = [2.294 - O3N2, -1.411, -3.077]
            roots = [
                np.real(v) for v in np.roots(p) + 8.0 if np.isreal(v) and 7.8 < v < 8.6
            ]
            return [O3N2], roots
        else:
            p = [0.281 - O3N2, -4.765, -2.268]
            roots = [
                np.real(v) for v in np.roots(p) + 8.69 if np.isreal(v) and 7.6 < v < 8.9
            ]
            return [O3N2], roots
    else:
        return [O3N2], []


def O_O3S2(sources, new=True, **kwargs):
    if len(sources) == 1 and "rec_O_O3S2"+"_n"*new in sources[0].keys():
        return sources[0]["rec_O_O3S2"+"_n"*new]
    O3S2 = O_R3(sources, **kwargs)[0][0] - O_S2(sources, **kwargs)[0][0]
    if np.isfinite(O3S2):
        if new:
            p = [1.997 - O3S2, -1.981]
            roots = [
                np.real(v) for v in np.roots(p) + 8.0 if np.isreal(v) and 7.9 < v < 8.6
            ]
            return [O3S2], roots
        else:
            p = [0.191 - O3S2, -4.292, -2.538, 0.053, 0.332]
            roots = [
                np.real(v) for v in np.roots(p) + 8.69 if np.isreal(v) and 7.6 < v < 8.9
            ]
            return [O3S2], roots
    else:
        return [O3S2], []


def O_S2(sources, new=True, **kwargs):
    if len(sources) == 1 and "rec_O_S2"+"_n"*new in sources[0].keys():
        return sources[0]["rec_O_S2"+"_n"*new]
    fsii = fit_lines(
        sources, [0.6718, 0.6732], {"S2_6716A": 0.6717, "S2_6731A": 0.6732}, **kwargs
    )
    fsii = sum(fsii.values())
    fhal = fit_lines(sources, [0.6550, 0.6564, 0.6585], {"H1r_6563A": 0.6564}, **kwargs)
    fhal = sum(fhal.values())
    S2 = np.log10(fsii / fhal)
    if np.isfinite(S2):
        if new:
            p = [-1.139 - S2, 0.723]
            roots = [
                np.real(v) for v in np.roots(p) + 8.0 if np.isreal(v) and 7.9 < v < 8.6
            ]
            return [S2], roots
        else:
            p = [-0.442 - S2, -0.360, -6.271, -8.339, -3.559]
            roots = [
                np.real(v) for v in np.roots(p) + 8.69 if np.isreal(v) and 7.6 < v < 8.9
            ]
            return [S2], roots
    else:
        return [S2], []


def O_R2(sources, new=True, **kwargs):
    if len(sources) == 1 and "rec_O_R2"+"_n"*new in sources[0].keys():
        return sources[0]["rec_O_R2"+"_n"*new]
    foii = fit_lines(
        sources, [0.3726, 0.3729], {"O2_3726A": 0.3726, "O2_3729A": 0.3729}, **kwargs
    )
    foii = sum(foii.values())
    fhbe = fit_lines(sources, [0.4862], {"H1r_4861A": 0.4862}, **kwargs)
    fhbe = sum(fhbe.values())
    R2 = np.log10(foii / fhbe)
    if np.isfinite(R2):
        if new:
            p = [0.172 - R2, 0.954, -0.832]
            roots = [
                np.real(v) for v in np.roots(p) + 8.0 if np.isreal(v) and 7.3 < v < 8.6
            ]
            return [R2], roots
        else:
            p = [0.435 - R2, -1.362, -5.655, -4.851, -0.478, 0.736]
            roots = [
                np.real(v) for v in np.roots(p) + 8.69 if np.isreal(v) and 7.6 < v < 8.9
            ]
            return [R2], roots
    else:
        return [R2], []


def O_O3O2(sources, new=True, **kwargs):
    if len(sources) == 1 and "rec_O_O3O2"+"_n"*new in sources[0].keys():
        return sources[0]["rec_O_O3O2"+"_n"*new]
    O3O2 = O_R3(sources, **kwargs)[0][0] - O_R2(sources, **kwargs)[0][0]
    if np.isfinite(O3O2):
        if new:
            p = [0.697 - O3O2, -1.245, -0.869]
            roots = [
                np.real(v) for v in np.roots(p) + 8.0 if np.isreal(v) and 7.3 < v < 8.6
            ]
            return [O3O2], roots
        else:
            p = [-0.691 - O3O2, -2.944, -1.308]
            roots = [
                np.real(v) for v in np.roots(p) + 8.69 if np.isreal(v) and 7.6 < v < 8.9
            ]
            return [O3O2], roots
    else:
        return [O3O2], []


def O_R23(sources, new=True, **kwargs):
    if len(sources) == 1 and "rec_O_R23"+"_n"*new in sources[0].keys():
        return sources[0]["rec_O_R23"+"_n"*new]
    foiii = fit_lines(sources, [0.5007], {"O3_5007A": 0.5007}, **kwargs)
    foiii = sum(foiii.values())
    foii = fit_lines(
        sources, [0.3726, 0.3729], {"O2_3726A": 0.3726, "O2_3729A": 0.3729}, **kwargs
    )
    foii = sum(foii.values())
    fhbe = fit_lines(sources, [0.4862], {"H1r_4861A": 0.4862}, **kwargs)
    fhbe = sum(fhbe.values())
    R23 = np.log10((foii + foiii) / fhbe)
    if np.isfinite(R23):
        if new:
            p = [0.998 - R23, 0.053, -0.141, -0.493,-0.774]
            roots = [
                np.real(v) for v in np.roots(p) + 8.0 if np.isreal(v) and 7.3 < v < 8.6
            ]
            return [R23], roots
        else:
            p = [0.527 - R23, -1.569, -1.652, -0.421]
            roots = [
                np.real(v) for v in np.roots(p) + 8.69 if np.isreal(v) and 7.6 < v < 8.9
            ]
            return [R23], roots
    else:
        return [R23], []


def O_RS32(sources, new=False, **kwargs):
    if len(sources) == 1 and "rec_O_RS32"+"_n"*new in sources[0].keys():
        return sources[0]["rec_O_RS32"+"_n"*new]
    or3 = O_R3(sources, **kwargs)[0][0]
    os2 = O_S2(sources, **kwargs)[0][0]
    RS32 = np.log10(
        10 ** (or3) + 10 ** (os2)
    )
    if np.isfinite(RS32):
        p = [-0.054 - RS32, -2.546, -1.970, 0.082, 0.222]
        roots = [
            np.real(v) for v in np.roots(p) + 8.69 if np.isreal(v) and 7.6 < v < 8.9
        ]
        return [RS32], roots
    else:
        return [RS32], []
        
def O_O3S2(sources, new=True, **kwargs):
    if len(sources) == 1 and "rec_O_O3S2"+"_n"*new in sources[0].keys():
        return sources[0]["rec_O_O3S2"+"_n"*new]
    or3 = O_R3(sources, **kwargs)[0][0]
    os2 = O_S2(sources, **kwargs)[0][0]
    O3S2 = or3-os2
    if np.isfinite(O3S2):
        p = [1.997 - O3S2, -1.981]
        roots = [
            np.real(v) for v in np.roots(p) + 8.0 if np.isreal(v) and 7.9 < v < 8.6
        ]
        return [O3S2], roots
    else:
        return [O3S2], []

def SFR(O_ab, H_a, z):
    """This will only work if slit loss correction is done."""
    Z = 0.014*10**(O_ab-8.69)
    C = -40.26+0.89*np.log(Z)+0.14*np.log(Z)**2
    dist = cosm.luminosity_distance(z).to('m').value
    L_a = H_a*4*np.pi*dist**2*10**7
    M_0 = 1
    SFR = M_0*10**C*L_a
    return SFR

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


def abunds(flx, tem, den, N_over_O=True):
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


def mark_agn(sources):
    bpt = lambda x: 0.61 / (x - 0.47) + 1.19 if x < 0.47 else -np.inf
    for s in sources:
        if not catalog.rm_bad([s], agn=True) or s["grat"] == "prism":
            s["BPT_pos"] = 1
            s["N2"] = None
            s["R3"] = None
            continue
        N2 = O_N2([s])[0][0]
        R3 = O_R3([s])[0][0]
        if np.isfinite(N2) and np.isfinite(R3):
            if R3 > bpt(N2):
                s["BPT_pos"] = 2.5
            else:
                s["BPT_pos"] = 0
            s["N2"] = N2
            s["R3"] = R3

        elif np.isfinite(N2):
            if N2 > 0:
                s["BPT_pos"] = 2.2
            else:
                s["BPT_pos"] = 0.2
            s["N2"] = N2
            s["R3"] = None
        elif np.isfinite(R3):
            if R3 > 0.9:
                s["BPT_pos"] = 2.3
            else:
                s["BPT_pos"] = 0.3
            s["N2"] = None
            s["R3"] = R3
        else:
            s["BPT_pos"] = 1.5
            s["N2"] = None
            s["R3"] = None
    return sources


def abundances(sources, cal_red=None, N_over_O=True, **kwargs):
    if cal_red is None:
        cHbeta = red_const(sources)
    else:
        cHbeta = cal_red
    rc.cHbeta = cHbeta
    lins = dict()
    fluxes = dict()
    fluxec = dict()
    for l in core_lines:
        fluxs = fit_lines(sources, l[1], l[0], dwidth=l[2], cal_red=0, **kwargs)
        for f in fluxs:
            fluxes[f] = fluxs[f]
    for l, flux in fluxes.items():
        if flux > 0:
            wav = float(l.split("_")[1].split("A")[0])
            fluxec[l] = flux * rc.getCorrHb(wav)
        else:
            fluxec[l] = 0.0
    tem, den = tem_den(fluxec)
    ions = []
    abun = abunds(fluxec, tem, den, N_over_O=N_over_O)
    return abun


def S_Dir(sources, new=False, **kwargs):
    if len(sources) == 1 and "rec_S_Dir"+"_n"*new in sources[0].keys():
        return sources[0]["rec_S_Dir"+"_n"*new]
    return [[abundances(sources, **kwargs)["S"]]] * 2


def N_Dir(sources, new=False, **kwargs):
    if len(sources) == 1 and "rec_N_Dir"+"_n"*new in sources[0].keys():
        return sources[0]["rec_N_Dir"+"_n"*new]
    return [[abundances(sources, N_over_O=True, **kwargs)["N"]]] * 2


def O_Dir(sources, new=False, **kwargs):
    if len(sources) == 1 and "rec_O_Dir"+"_n"*new in sources[0].keys():
        return sources[0]["rec_O_Dir"+"_n"*new]
    return [[abundances(sources, **kwargs)["O"]]] * 2


Oxygen = {
    "N2": lambda *args, **kwags: O_N2(*args, **kwargs, new=False),
    "R3": lambda *args, **kwags: O_R3(*args, **kwargs, new=False),
    "O3N2": lambda *args, **kwags: O_O3N2(*args, **kwargs, new=False),
    "O3S2": lambda *args, **kwags: O_O3S2(*args, **kwargs, new=False),
    "S2": lambda *args, **kwags: O_S2(*args, **kwargs, new=False),
    "R2": lambda *args, **kwags: O_R2(*args, **kwargs, new=False),
    "O3O2": lambda *args, **kwags: O_O3O2(*args, **kwargs, new=False),
    "R23": lambda *args, **kwags: O_R23(*args, **kwargs, new=False),
    "RS32": lambda *args, **kwags: O_RS32(*args, **kwargs, new=False),
}
Nitrogen = {
    "N2O2": lambda *args, **kwags: N_N2O2(*args, **kwargs, new=False),
    "N2": lambda *args, **kwags: N_N2(*args, **kwargs, new=False),
    "N2S2": lambda *args, **kwags: N_N2S2(*args, **kwargs, new=False),
}
Sulphur = {"S23": lambda *args, **kwags: S_S23(*args, **kwargs, new=False)}
Oxygen_new = {
    "R3": lambda *args, **kwags: O_R3(*args, **kwargs, new=True),
    "R2": lambda *args, **kwags: O_R2(*args, **kwargs, new=True),
    "R23": lambda *args, **kwags: O_R23(*args, **kwargs, new=True),
    "O3O2": lambda *args, **kwags: O_O3O2(*args, **kwargs, new=True),
    "N2": lambda *args, **kwags: O_N2(*args, **kwargs, new=True),
    "O3N2": lambda *args, **kwags: O_O3N2(*args, **kwargs, new=True),
    "O3S2": lambda *args, **kwags: O_O3S2(*args, **kwargs, new=True),
    "S2": lambda *args, **kwags: O_S2(*args, **kwargs, new=True),
    "O3S2": lambda *args, **kwags: O_O3S2(*args, **kwargs, new=True),
}
Nitrogen_new = {
    "N2O2": lambda *args, **kwags: N_N2O2(*args, **kwargs, new=True),
    "N2": lambda *args, **kwags: N_N2(*args, **kwargs, new=True),
    "N2S2": lambda *args, **kwags: N_N2S2(*args, **kwargs, new=True),
}
Sulphur_new = {"S23": lambda *args, **kwags: S_S23(*args, **kwargs, new=False)}
Names = {
    "N2": "$\\mathrm{log}([\\mathrm{N}_\\mathrm{II}]\\lambda 6584/\\mathrm{H}_\\alpha)$",
    "R3": "$\\mathrm{log}([\\mathrm{O}_\\mathrm{III}]\\lambda 5007/\\mathrm{H}_\\beta)$",
    "O3N2": "$\\mathrm{R}3-\\mathrm{N2}$",
    "O3S2": "$\\mathrm{R}3-\\mathrm{S2}$",
    "S2": "$\\mathrm{log}([\\mathrm{S}_\\mathrm{II}]\\lambda\\lambda 6717, 6731/\\mathrm{H}_\\alpha)$",
    "R2": "$\\mathrm{log}([\\mathrm{O}_\\mathrm{II}]\\lambda 3727/\\mathrm{H}_\\beta)$",
    "O3O2": "$\\mathrm{log}([\\mathrm{O}_\\mathrm{III}]\\lambda 5007/[\\mathrm{O}_\\mathrm{II}]\\lambda 3727)$",
    "R23": "$\\mathrm{log}(([\\mathrm{O}_\\mathrm{II}]\\lambda 3727+[\\mathrm{O}_\\mathrm{III}]\\lambda 5007)/\\mathrm{H}_\\beta)$",
    "RS32": "$\\mathrm{log}(10^\\mathrm{R3}+10^\\mathrm{S2})$",
    "O3S2": "$\\mathrm{R3}-\\mathrm{S2}$",
    "N2O2": "$\\mathrm{log}([\\mathrm{N}_\\mathrm{II}]\\lambda 6585/[\\mathrm{O}_\\mathrm{II}]\\lambda 3727)$",
    "N2": "$\\mathrm{log}([\\mathrm{N}_\\mathrm{II}]\\lambda 6585/[\\mathrm{H}_\\alpha)$",
    "N2S2": "$\\mathrm{log}([\\mathrm{N}_\\mathrm{II}]\\lambda 6585/[\\mathrm{S}_\\mathrm{II}]\\lambda\\lambda 6717, 6731)$",
    "S23": "$\\mathrm{log}(([\\mathrm{S}_\\mathrm{II}]\\lambda\\lambda 6716, 6731 + [\\mathrm{S}_\\mathrm{III}]\\lambda 9532)/\\mathrm{H}_\\beta)$",
}
core_lines = (
    ({"O2_3726A": 0.3726, "O2_3729A": 0.3729}, [0.3726, 0.3729], 7),
    ({"S2_4069A": 0.4070, "S2_4076A": 0.4076}, [0.4070, 0.4076], 2.7),
    ({"H1r_4102A": 0.4102}, [0.4102], 4),
    ({"H1r_4341A": 0.4341}, [0.4341, 0.4364], 7),
    ({"O3_4363A": 0.4363}, [0.4364, 0.4372], 5),
    ({"H1r_4861A": 0.4862}, [0.4862], 8),
    ({"O3_4959A": 0.4959}, [0.4959], 8),
    ({"O3_5007A": 0.5007}, [0.5007], 8),
    ({"N2_5755A": 0.5756}, [0.5756, 0.5750], 3),
    ({"S3_6312A": 0.6313}, [0.6302, 0.6312], 3),
    (
        {"H1r_6563A": 0.6564, "N2_6548A": 0.6548, "N2_6584A": 0.6585},
        [0.6550, 0.6564, 0.6585],
        6,
    ),
    ({"S2_6716A": 0.6717, "S2_6731A": 0.6732}, [0.6718, 0.6732], 7),
    ({"O2_7319A": 0.7320, "O2_7330A": 0.7331}, [0.7320, 0.7331], 8),
    ({"S3_9069A": 0.9070}, [0.9071], 8),
    ({"S3_9531A": 0.9532}, [0.9533, 0.9551], 8),
)


def fit_lines(
    sources,
    lines,
    mline,
    reso=300,
    base="../Data/Npy/",
    typ="median",
    cal_red=None,
    dwidth=8,
    manual=False,
    indiv=True,
    R=850,
    **kwargs,
):
    if not any(sources):
        return lf.flux_nan(mline)
    if cal_red is None:
        cal_red = red_const(sources, indiv=indiv)
    if len(sources) == 1 and indiv:
        return indiv_extract(sources[0], mline, cal_red)
    grats = [s["grat"] for s in sources]
    grat = max(set(grats), key=grats.count) if R is None else R
    rang, _ = lf.line_range(lines, grat=grat)
    sources = catalog.filter_zranges(sources, [rang])
    spectra, sourn = spectr.resampled_spectra(sources, rang, reso, base=base, degrade=R)
    if len(sourn):
        stack = spectr.combine_spectra(spectra)
        stacc = spectr.stack(stack, sourn, typ=typ)
        fit, x, stacc = lf.fit_lines(
            stacc, lines, grat=grat, dwidth=dwidth, manual=manual, mline=mline
        )
        flux = lf.flux_extract(
            fit, mline, grat=grat, red=lambda f, l: redd(f, l, sources, cal_red)
        )
        return flux
    else:
        return lf.flux_nan(mline)


def indiv_extract(source, mline, cal_red):
    outd = True
    rca = lambda f, l: redd(f, l, [source], cal_red)
    if type(mline) is not dict:
        mline = {"0": mline}
        outd = False
    fluxes = dict()
    for k, l in mline.items():
        if f"rec_{k}" in source.keys():
            fluxes[k] = rca(source[f"rec_{k}"], l)
        else:
            fluxes[k] = np.nan
    if outd:
        return fluxes
    else:
        return next(f for f in fluxes.values())


def red_const(sources, T=12000, **kwargs):

    lT = np.log10(T)
    wavs = {
        6563.0: 10.35 - 3.254 * lT + 0.3457 * lT**2,
        4340.0: 0.0254 + 0.1922 * lT - 0.0204 * lT**2,
        4102.0: -0.07132 + 0.1436 * lT - 0.0153 * lT**2,
    }
    flub = sum(
        fit_lines(
            sources, [0.4862], {"H1r_4861A": 0.4862}, cal_red=0, **kwargs
        ).values()
    )
    if flub:
        flux = {
            6563.0: sum(
                fit_lines(
                    sources,
                    [0.6550, 0.6564, 0.6585],
                    {"H1r_6563A": 0.6564},
                    cal_red=0,
                    **kwargs,
                ).values()
            )
            / flub,
            4340.0: sum(
                fit_lines(
                    sources,
                    [0.4341, 0.4364],
                    {"H1r_4341A": 0.4341},
                    cal_red=0,
                    **kwargs,
                ).values()
            )
            / flub,
            4102.0: sum(
                fit_lines(
                    sources, [0.4102], {"H1r_4102A": 0.4102}, cal_red=0, **kwargs
                ).values()
            )
            / flub,
        }
        consts = []
        for k in wavs:
            if np.isfinite(flux[k]) and flux[k] > 0:
                rc.setCorr(obs_over_theo=flux[k] / wavs[k], wave1=k, wave2=4862.0)
                consts.append(rc.cHbeta)
        # print("cHbeta values: " + str(consts))
        return consts[0] if consts else 0.0
    else:
        return 0.0


def redd(flux, line, sources, cal_red):
    if cal_red is None:
        cal_red = red_const(sources)
    if cal_red == 0.0:
        return flux
    else:
        rc.cHbeta = cal_red
        return flux * rc.getCorrHb(line * 10**4)


def iprocess(funct, srs, calib, zs, vs, it, val, sind, **kwargs):
    if calib is None:
        for i, sr in enumerate(srs):
            vi = funct([sr], **kwargs)
            vs[sind + i] = vi
            zs[sind + i] = vi
            it.value += 1
    else:
        ind = 1 if calib else 0
        for i, sr in enumerate(srs):
            if (v := sr.get(val)) is not None:
                vi = funct([sr], **kwargs)[ind]
                zs[sind + i] = [sr[val]] * len(vi)
                vs[sind + i] = vi
                if vi:
                    it.value += 1
            else:
                zs[sind + i] = []
                vs[sind + i] = []


def bprocess(funct, srs, ind, vis, **kwargs):
    for sr in srs:
        vi = np.flip(funct(sr, **kwargs)[ind])
        vis.append(vi)


def boots_stat(funct, sources, ite=200, calib=True, manual=False, **kwargs):
    ind = 1 if calib else 0
    vs = np.flip(funct(sources, manual=manual, indiv=False, **kwargs)[ind])
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
                kwargs["indiv"] = False
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


def indiv_stat(funct, sources, val="z", calib=True, **kwargs):
    no = len(sources)
    manag = Manager()
    va = manag.dict()
    zs = manag.dict()
    vs = manag.dict()
    i = 0
    nos = 50
    it = manag.Value(int, 0)
    proc = cpu_count() * 2
    active = []
    while it.value < no and i < len(sources) or len(active) > 0:
        pr_no = -((-min(len(sources) - i, no - it.value)) // nos)
        if proc > len(active) and (it.value < no and i < len(sources)):
            for l in range(min(pr_no, proc - len(active))):
                print(f"\r\033[KCalculating {i} out of {no} points. ", end="")
                sr = sources[i : i + nos]
                args = (funct, sr, calib, zs, vs, it, val, i)
                t = Process(target=iprocess, args=args, kwargs=kwargs)
                t.start()
                active.append(t)
                i += len(sr)
        for t in active:
            if not t.is_alive():
                t.terminate()
                active.remove(t)
        time.sleep(0.1)
    vs = list(zip(*sorted(vs.items(), key=lambda x: x[0])))[1]
    zs = list(zip(*sorted(zs.items(), key=lambda x: x[0])))[1]
    if calib is not None:
        if sum(list(vs), []):
            vals = np.full((len(vs), max([len(v) for v in vs])), np.nan)
            for l in range(len(vs)):
                vi = np.flip(vs[l])
                vals[l, : vi.shape[0]] = vi[: vals.shape[1]]
            vals = np.nan_to_num(vals, nan=np.nan, posinf=np.nan, neginf=np.nan)
            medians = np.nanmedian(vals, axis=0)
            err33 = np.nanpercentile(vals, 33, axis=0)
            err67 = np.nanpercentile(vals, 67, axis=0)
        else:
            medians, err33, err67 = [np.array([np.nan])] * 3
        print(f"Needed {i} out of {len(sources)} for {it.value} results.")
        return (zs, vs), (
            medians,
            np.nan_to_num([medians - err33, err67 - medians], nan=0.0),
        )
    else:
        return vs
