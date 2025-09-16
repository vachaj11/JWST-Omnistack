"""Holds methods for line fitting and subsequent automatised abundance calculations.

Attributes:
    glob_hash (dict): Dictionary storing all calculated results by hash of respective calculation inputs, so that the calculations don't have to be repeated.
    glob_seed (int): Value serving as a global seed, generated at random, to get a different, but still deterministic Generator on each instantiation of the module.
    H0 (float): Assumed value of Hubble parameter
    Om0 (float): Assumed cosmological density of matter
    Od0 (float): Assumed cosmological density of dark energy
    cosm (astropy.cosmology.LambdaCDM): AstroPy cosmology instance used throughout module for cosmological calculations
    rc (pyneb.RedCorr): PyNeb galaxy extintion profile used for calculations of dust attenuation 
    Oxygen (dict): Dictionary holding legacy methods for Oxygen abundance calculations via strong lines.
    Nitrogen (dict): Dictionary holding legacy methods for Nitrogen abundance calculations via strong lines.
    Sulphur (dict): Dictionary holding legacy methods for Sulphur abundance calculations via strong lines.
    Oxygen_new (dict): Dictionary holding updated methods for Oxygen abundance calculations via strong lines.
    Nitrogen_new (dict): Dictionary holding updated methods for Nitrogen abundance calculations via strong lines.
    Sulphur_new (dict): Dictionary holding updated methods for Sulphur abundance calculations via strong lines.
    names (dict): Dictionary holding definitions typeset in Latex for all used strong-line abundance calibrations.
    core_lines (dict): Dictionary holding definitions of spectral line positions in various regions of interest. 
"""

import time
from hashlib import sha1
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


glob_hash = dict()

glob_seed = np.random.randint(0, 2**32)

H0 = 67.49
Om0 = 0.315
Od0 = 0.6847
cosm = LambdaCDM(H0, Om0, Od0)

rc = pn.RedCorr(law="CCM89")

Oxygen = {
    "R2": lambda *args, **kwargs: O_R2(*args, **kwargs, new=False),
    "R3": lambda *args, **kwargs: O_R3(*args, **kwargs, new=False),
    "R23": lambda *args, **kwargs: O_R23(*args, **kwargs, new=False),
    "O3O2": lambda *args, **kwargs: O_O3O2(*args, **kwargs, new=False),
    "RS32": lambda *args, **kwargs: O_RS32(*args, **kwargs, new=False),
    "N2": lambda *args, **kwargs: O_N2(*args, **kwargs, new=False),
    "O3N2": lambda *args, **kwargs: O_O3N2(*args, **kwargs, new=False),
    "S2": lambda *args, **kwargs: O_S2(*args, **kwargs, new=False),
    "O3S2": lambda *args, **kwargs: O_O3S2(*args, **kwargs, new=False),
}
Nitrogen = {
    "N2": lambda *args, **kwargs: N_N2(*args, **kwargs, new=False),
    "N2O2": lambda *args, **kwargs: N_N2O2(*args, **kwargs, new=False),
    "N2S2": lambda *args, **kwargs: N_N2S2(*args, **kwargs, new=False),
}
Sulphur = {"S23": lambda *args, **kwargs: S_S23(*args, **kwargs, new=False)}
Oxygen_new = {
    "R2": lambda *args, **kwargs: O_R2(*args, **kwargs, new=True),
    "R3": lambda *args, **kwargs: O_R3(*args, **kwargs, new=True),
    "R23": lambda *args, **kwargs: O_R23(*args, **kwargs, new=True),
    "O3O2": lambda *args, **kwargs: O_O3O2(*args, **kwargs, new=True),
    "$\\hat{R}$": lambda *args, **kwargs: O_Rh(*args, **kwargs, new=True),
    "N2": lambda *args, **kwargs: O_N2(*args, **kwargs, new=True),
    "O3N2": lambda *args, **kwargs: O_O3N2(*args, **kwargs, new=True),
    "S2": lambda *args, **kwargs: O_S2(*args, **kwargs, new=True),
    "O3S2": lambda *args, **kwargs: O_O3S2(*args, **kwargs, new=True),
}
Nitrogen_new = {
    "N2": lambda *args, **kwargs: N_N2(*args, **kwargs, new=True),
    "N2O2": lambda *args, **kwargs: N_N2O2(*args, **kwargs, new=True),
    "N2S2": lambda *args, **kwargs: N_N2S2(*args, **kwargs, new=True),
}
Sulphur_new = {"S23": lambda *args, **kwargs: S_S23(*args, **kwargs, new=True)}
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
core_lines = {
    "O2_37": ({"O2_3726A": 0.3726, "O2_3729A": 0.3729}, [0.3726, 0.3729], 7),
    "S2_40": ({"S2_4069A": 0.4070, "S2_4076A": 0.4076}, [0.4070, 0.4076], 2.7),
    "H1_d": ({"H1r_4102A": 0.4102}, [0.4102], 4),
    "H1_g": ({"H1r_4341A": 0.4341}, [0.4341, 0.4364], 7),
    "O3_43": ({"O3_4363A": 0.4363}, [0.4364, 0.4372], 5),
    "H1_b": ({"H1r_4861A": 0.4862}, [0.4862], 8),
    "O3_49": ({"O3_4959A": 0.4959}, [0.4959], 8),
    "O3_50": ({"O3_5007A": 0.5007}, [0.5007], 8),
    "N2_57": ({"N2_5755A": 0.5756}, [0.5756, 0.5750], 3),
    "S3_63": ({"S3_6312A": 0.6313}, [0.6302, 0.6312], 3),
    "H1_a": (
        {"H1r_6563A": 0.6564, "N2_6548A": 0.6548, "N2_6584A": 0.6585},
        [0.6550, 0.6564, 0.6585],
        6,
    ),
    "S2_67": ({"S2_6716A": 0.6717, "S2_6731A": 0.6732}, [0.6718, 0.6732], 7),
    "O2_73": ({"O2_7319A": 0.7320, "O2_7330A": 0.7331}, [0.7320, 0.7331], 8),
    "S3_90": ({"S3_9069A": 0.9070}, [0.9071], 8),
    "S3_95": ({"S3_9531A": 0.9532}, [0.9533, 0.9551], 8),
}


def S_S23(
    sources, new=False, cal_red=None, temp=True, rec=False, constr=False, **kwargs
):
    """Calculates Sulphur abundance via the strong-line S23 calibration for stack of spectra in passed catalogue."""
    if len(sources) == 1 and "rec_S_S23" + "_n" * new in sources[0].keys() and not rec:
        return sources[0]["rec_S_S23" + "_n" * new]
    T, n, flx = tem_den_red(
        sources, temp=temp, lines={"S2_67", "S3_95", "H1_b"}, cal_red=cal_red, **kwargs
    )
    fsii = sum(flx["S2_67"].values())
    fsiii = sum(flx["S3_95"].values())
    fhbe = sum(flx["H1_b"].values())
    S23 = np.log10((fsii + fsiii) / fhbe)
    if np.isfinite(S23) and (-1.05 < S23 < 0.3 or not constr):
        return [S23], []
    else:
        return [S23], []


def N_N2(
    sources, new=False, cal_red=None, temp=True, rec=False, constr=False, **kwargs
):
    """Calculates Nitrogen abundance via the strong-line N2 calibration for stack of spectra in passed catalogue."""
    if len(sources) == 1 and "rec_N_N2" + "_n" * new in sources[0].keys() and not rec:
        return sources[0]["rec_N_N2" + "_n" * new]
    T, n, flx = tem_den_red(
        sources, temp=temp, lines={"H1_a"}, cal_red=cal_red, **kwargs
    )
    fnii = flx["H1_a"]["N2_6584A"]
    fhal = flx["H1_a"]["H1r_6563A"]
    N2 = np.log10(fnii / fhal)
    if np.isfinite(N2) and (-1.8 < N2 < -0.2 or not constr):
        return [N2], [0.62 * N2 - 0.57]
    else:
        return [N2], []


def N_N2O2(
    sources, new=True, cal_red=None, temp=True, rec=False, constr=False, **kwargs
):
    """Calculates Nitrogen abundance via the strong-line N2O2 calibration for stack of spectra in passed catalogue."""
    if len(sources) == 1 and "rec_N_N2O2" + "_n" * new in sources[0].keys() and not rec:
        return sources[0]["rec_N_N2O2" + "_n" * new]
    T, n, flx = tem_den_red(
        sources, temp=temp, lines={"H1_a", "O2_37"}, cal_red=cal_red, **kwargs
    )
    fnii = flx["H1_a"]["N2_6584A"]
    foii = sum(flx["O2_37"].values())
    N2O2 = np.log10(fnii / foii)
    if np.isfinite(N2O2):
        if new and (-1.5 < N2O2 < 0.25 or not constr):
            return [N2O2], [0.69 * N2O2 - 0.65]
        elif not new and (-2 < N2O2 < 0 or not constr):
            return [N2O2], [0.52 * N2O2 - 0.65]
        else:
            return [N2O2], []
    else:
        return [N2O2], []


def N_N2S2(sources, new=True, rec=False, constr=False, **kwargs):
    """Calculates Nitrogen abundance via the strong-line N2S2 calibration for stack of spectra in passed catalogue."""
    if len(sources) == 1 and "rec_N_N2S2" + "_n" * new in sources[0].keys() and not rec:
        return sources[0]["rec_N_N2S2" + "_n" * new]
    N2S2 = (
        O_N2(sources, rec=rec, **kwargs)[0][0] - O_S2(sources, rec=rec, **kwargs)[0][0]
    )
    if np.isfinite(N2S2):
        if new and (-0.6 < N2S2 < 0.3 or not constr):
            return [N2S2], [1.12 * N2S2 - 0.93]
        elif not new and (-0.8 < N2S2 < 0.5 or not constr):
            return [N2S2], [0.85 * N2S2 - 1.00]
        else:
            return [N2S2], []
    else:
        return [N2S2], []


def O_R2(sources, new=True, cal_red=None, temp=True, rec=False, **kwargs):
    """Calculates Oxygen abundance via the strong-line R2 calibration for stack of spectra in passed catalogue."""
    if len(sources) == 1 and "rec_O_R2" + "_n" * new in sources[0].keys() and not rec:
        return sources[0]["rec_O_R2" + "_n" * new]
    T, n, flx = tem_den_red(
        sources, temp=temp, lines={"H1_b", "O2_37"}, cal_red=cal_red, **kwargs
    )
    foii = sum(flx["O2_37"].values())
    fhbe = sum(flx["H1_b"].values())
    R2 = np.log10(foii / fhbe)
    if np.isfinite(R2):
        if new:
            p = [0.172 - R2, 0.954, -0.832]
            roots = [
                np.real(v)
                for v in np.roots(np.flip(p)) + 8.0
                if np.isreal(v) and 7.3 < v < 8.6
            ]
            return [R2], sorted(roots)
        else:
            p = [0.435 - R2, -1.362, -5.655, -4.851, -0.478, 0.736]
            roots = [
                np.real(v)
                for v in np.roots(np.flip(p)) + 8.69
                if np.isreal(v) and 7.6 < v < 8.9
            ]
            return [R2], sorted(roots)
    else:
        return [R2], []


def O_R3(sources, new=True, cal_red=None, temp=True, rec=False, **kwargs):
    """Calculates Oxygen abundance via the strong-line R3 calibration for stack of spectra in passed catalogue."""
    if len(sources) == 1 and "rec_O_R3" + "_n" * new in sources[0].keys() and not rec:
        return sources[0]["rec_O_R3" + "_n" * new]
    T, n, flx = tem_den_red(
        sources, temp=temp, lines={"H1_b", "O3_50"}, cal_red=cal_red, **kwargs
    )
    foiii = sum(flx["O3_50"].values())
    fhbe = sum(flx["H1_b"].values())
    R3 = np.log10(foiii / fhbe)
    if np.isfinite(R3):
        if new:
            p = [0.852 - R3, -0.162, -1.149, -0.553]
            roots = [
                np.real(v)
                for v in np.roots(np.flip(p)) + 8.0
                if np.isreal(v) and 7.3 < v < 8.6
            ]
            return [R3], sorted(roots)
        else:
            p = [-0.277 - R3, -3.549, -3.593, -0.981]
            roots = [
                np.real(v)
                for v in np.roots(np.flip(p)) + 8.69
                if np.isreal(v) and 7.6 < v < 8.9
            ]
            return [R3], sorted(roots)
    else:
        return [R3], []


def O_O3O2(sources, new=True, rec=False, **kwargs):
    """Calculates Oxygen abundance via the strong-line O3O2 calibration for stack of spectra in passed catalogue."""
    if len(sources) == 1 and "rec_O_O3O2" + "_n" * new in sources[0].keys() and not rec:
        return sources[0]["rec_O_O3O2" + "_n" * new]
    O3O2 = (
        O_R3(sources, rec=rec, **kwargs)[0][0] - O_R2(sources, rec=rec, **kwargs)[0][0]
    )
    if np.isfinite(O3O2):
        if new:
            p = [0.697 - O3O2, -1.245, -0.869]
            roots = [
                np.real(v)
                for v in np.roots(np.flip(p)) + 8.0
                if np.isreal(v) and 7.3 < v < 8.6
            ]
            return [O3O2], sorted(roots)
        else:
            p = [-0.691 - O3O2, -2.944, -1.308]
            roots = [
                np.real(v)
                for v in np.roots(np.flip(p)) + 8.69
                if np.isreal(v) and 7.6 < v < 8.9
            ]
            return [O3O2], sorted(roots)
    else:
        return [O3O2], []


def O_R23(sources, new=True, cal_red=None, temp=True, rec=False, **kwargs):
    """Calculates Oxygen abundance via the strong-line R23 calibration for stack of spectra in passed catalogue."""
    if len(sources) == 1 and "rec_O_R23" + "_n" * new in sources[0].keys() and not rec:
        return sources[0]["rec_O_R23" + "_n" * new]
    T, n, flx = tem_den_red(
        sources, temp=temp, lines={"H1_b", "O3_50", "O2_37"}, cal_red=cal_red, **kwargs
    )
    foiii = sum(flx["O3_50"].values())
    foii = sum(flx["O2_37"].values())
    fhbe = sum(flx["H1_b"].values())
    R23 = np.log10((foii + foiii) / fhbe)
    if np.isfinite(R23):
        if new:
            p = [0.998 - R23, 0.053, -0.141, -0.493, -0.774]
            roots = [
                np.real(v)
                for v in np.roots(np.flip(p)) + 8.0
                if np.isreal(v) and 7.3 < v < 8.6
            ]
            return [R23], sorted(roots)
        else:
            p = [0.527 - R23, -1.569, -1.652, -0.421]
            roots = [
                np.real(v)
                for v in np.roots(np.flip(p)) + 8.69
                if np.isreal(v) and 7.6 < v < 8.9
            ]
            return [R23], sorted(roots)
    else:
        return [R23], []


def O_Rh(sources, new=True, rec=False, **kwargs):
    """Calculates Oxygen abundance via the strong-line Rh calibration for stack of spectra in passed catalogue."""
    if len(sources) == 1 and "rec_O_Rh" + "_n" * new in sources[0].keys() and not rec:
        return sources[0]["rec_O_Rh" + "_n" * new]
    or3 = O_R3(sources, rec=rec, **kwargs)[0][0]
    or2 = O_R2(sources, rec=rec, **kwargs)[0][0]
    Rh = 0.47 * or2 + 0.88 * or3
    if np.isfinite(Rh):
        p = [0.779 - Rh, 0.263, -0.849, -0.493]
        roots = [
            np.real(v)
            for v in np.roots(np.flip(p)) + 8.0
            if np.isreal(v) and 7.3 < v < 8.6
        ]
        return [Rh], list(reversed(sorted(roots)))
    else:
        return [Rh], []


def O_N2(sources, new=True, cal_red=None, temp=True, rec=False, **kwargs):
    """Calculates Oxygen abundance via the strong-line N2 calibration for stack of spectra in passed catalogue."""
    if len(sources) == 1 and "rec_O_N2" + "_n" * new in sources[0].keys() and not rec:
        return sources[0]["rec_O_N2" + "_n" * new]
    T, n, flx = tem_den_red(
        sources, temp=temp, lines={"H1_a"}, cal_red=cal_red, **kwargs
    )
    fnii = flx["H1_a"]["N2_6584A"]
    fhal = flx["H1_a"]["H1r_6563A"]
    N2 = np.log10(fnii / fhal)
    if np.isfinite(N2):
        if new:
            p = [-1.356 - N2, 1.532]
            roots = [
                np.real(v)
                for v in np.roots(np.flip(p)) + 8.0
                if np.isreal(v) and 7.8 < v < 8.6
            ]
            return [N2], sorted(roots)
        else:
            p = [-0.489 - N2, 1.513, -2.554, -5.293, -2.867]
            roots = [
                np.real(v)
                for v in np.roots(np.flip(p)) + 8.69
                if np.isreal(v) and 7.6 < v < 8.9
            ]
            return [N2], sorted(roots)
    else:
        return [N2], []


def O_O3N2(sources, new=True, rec=False, **kwargs):
    """Calculates Oxygen abundance via the strong-line O3N2 calibration for stack of spectra in passed catalogue."""
    if len(sources) == 1 and "rec_O_O3N2" + "_n" * new in sources[0].keys() and not rec:
        return sources[0]["rec_O_O3N2" + "_n" * new]
    O3N2 = (
        O_R3(sources, rec=rec, **kwargs)[0][0] - O_N2(sources, rec=rec, **kwargs)[0][0]
    )
    if np.isfinite(O3N2):
        if new:
            p = [2.294 - O3N2, -1.411, -3.077]
            roots = [
                np.real(v)
                for v in np.roots(np.flip(p)) + 8.0
                if np.isreal(v) and 7.8 < v < 8.6
            ]
            return [O3N2], sorted(roots)
        else:
            p = [0.281 - O3N2, -4.765, -2.268]
            roots = [
                np.real(v)
                for v in np.roots(np.flip(p)) + 8.69
                if np.isreal(v) and 7.6 < v < 8.9
            ]
            return [O3N2], sorted(roots)
    else:
        return [O3N2], []


def O_S2(sources, new=True, cal_red=None, temp=True, rec=False, **kwargs):
    """Calculates Oxygen abundance via the strong-line S2 calibration for stack of spectra in passed catalogue."""
    if len(sources) == 1 and "rec_O_S2" + "_n" * new in sources[0].keys() and not rec:
        return sources[0]["rec_O_S2" + "_n" * new]
    T, n, flx = tem_den_red(
        sources, temp=temp, lines={"H1_a", "S2_67"}, cal_red=cal_red, **kwargs
    )
    fsii = sum(flx["S2_67"].values())
    fhal = flx["H1_a"]["H1r_6563A"]
    S2 = np.log10(fsii / fhal)
    if np.isfinite(S2):
        if new:
            p = [-1.139 - S2, 0.723]
            roots = [
                np.real(v)
                for v in np.roots(np.flip(p)) + 8.0
                if np.isreal(v) and 7.9 < v < 8.6
            ]
            return [S2], sorted(roots)
        else:
            p = [-0.442 - S2, -0.360, -6.271, -8.339, -3.559]
            roots = [
                np.real(v)
                for v in np.roots(np.flip(p)) + 8.69
                if np.isreal(v) and 7.6 < v < 8.9
            ]
            return [S2], sorted(roots)
    else:
        return [S2], []


def O_O3S2(sources, new=True, rec=False, **kwargs):
    """Calculates Oxygen abundance via the strong-line O3S2 calibration for stack of spectra in passed catalogue."""
    if len(sources) == 1 and "rec_O_O3S2" + "_n" * new in sources[0].keys() and not rec:
        return sources[0]["rec_O_O3S2" + "_n" * new]
    O3S2 = (
        O_R3(sources, rec=rec, **kwargs)[0][0] - O_S2(sources, rec=rec, **kwargs)[0][0]
    )
    if np.isfinite(O3S2):
        if new:
            p = [1.997 - O3S2, -1.981]
            roots = [
                np.real(v)
                for v in np.roots(np.flip(p)) + 8.0
                if np.isreal(v) and 7.9 < v < 8.6
            ]
            return [O3S2], sorted(roots)
        else:
            p = [0.191 - O3S2, -4.292, -2.538, 0.053, 0.332]
            roots = [
                np.real(v)
                for v in np.roots(np.flip(p)) + 8.69
                if np.isreal(v) and 7.6 < v < 8.9
            ]
            return [O3S2], sorted(roots)
    else:
        return [O3S2], []


def O_RS32(sources, new=False, rec=False, **kwargs):
    """Calculates Oxygen abundance via the strong-line RS32 calibration for stack of spectra in passed catalogue."""
    if len(sources) == 1 and "rec_O_RS32" + "_n" * new in sources[0].keys() and not rec:
        return sources[0]["rec_O_RS32" + "_n" * new]
    or3 = O_R3(sources, rec=rec, **kwargs)[0][0]
    os2 = O_S2(sources, rec=rec, **kwargs)[0][0]
    RS32 = np.log10(10 ** (or3) + 10 ** (os2))
    if np.isfinite(RS32):
        p = [-0.054 - RS32, -2.546, -1.970, 0.082, 0.222]
        roots = [
            np.real(v)
            for v in np.roots(np.flip(p)) + 8.69
            if np.isreal(v) and 7.6 < v < 8.9
        ]
        return [RS32], sorted(roots)
    else:
        return [RS32], []


def S_Dir(sources, new=True, rec=False, **kwargs):
    """Calculates Sulphur abundance via the direct method for stack of spectra in passed catalogue."""
    if len(sources) == 1 and "rec_S_Dir" + "_n" * new in sources[0].keys() and not rec:
        return sources[0]["rec_S_Dir" + "_n" * new]
    return [[abundances(sources, **kwargs)["S"]]] * 2


def N_Dir(sources, new=True, rec=False, **kwargs):
    """Calculates Nitrogen abundance via the direct method for stack of spectra in passed catalogue."""
    if len(sources) == 1 and "rec_N_Dir" + "_n" * new in sources[0].keys() and not rec:
        return sources[0]["rec_N_Dir" + "_n" * new]
    return [[abundances(sources, N_over_O=True, **kwargs)["N"]]] * 2


def O_Dir(sources, new=True, rec=False, **kwargs):
    """Calculates Oxygen abundance via the direct method for stack of spectra in passed catalogue."""
    if len(sources) == 1 and "rec_O_Dir" + "_n" * new in sources[0].keys() and not rec:
        return sources[0]["rec_O_Dir" + "_n" * new]
    return [[abundances(sources, **kwargs)["O"]]] * 2


def SFR(O_ab, H_a, z):
    """
    From passed Oxygen abundance, H_alpha luminosity and redshift calculates SFR estimate of the source.

    This provides meaningful results only if the spectrum is calibrated for stil loss.
    """
    if O_ab is not None and H_a is not None and z is not None:
        Z = 0.014 * 10 ** (O_ab - 8.69)
        C = -40.26 + 0.89 * np.log(Z) + 0.14 * np.log(Z) ** 2
        dist = cosm.luminosity_distance(z).to("m").value
        L_a = H_a * 4 * np.pi * dist**2 * 10**7
        M_0 = 1
        SFR = M_0 * 10**C * L_a
        return SFR
    else:
        return None


def abundances(sources, cal_red=None, N_over_O=True, hsh=True, **kwargs):
    """Central method for calculating abundances via the direct method for stack of spectra in passed catalogue."""
    hsh = hsh if len(sources) > 1 else False
    if hsh:
        hax = hashed(sources, "abundances", cal_red, N_over_O, **kwargs)
        if hax in glob_hash:
            return glob_hash[hax]
    tem, den, fluxc = tem_den_red(sources, **kwargs)
    ions = []
    abun = abunds(fluxc, tem, den, N_over_O=N_over_O)
    if hsh:
        glob_hash[hax] = abun
    return abun


def get_core_fluxes(sources, ilines=None, temp=True, cal_red=0, **kwargs):
    """Calculates line fluxes for stack of spectra in provided catalog. The fluxes to be calculated are chosen based on passed specifications."""
    nlines = set(core_lines.keys() if ilines is None else ilines)
    if temp:
        upd = {
            "O3_49",
            "O3_50",
            "O3_43",
            "O2_37",
            "O2_73",
            "S2_67",
            "S2_40",
            "S3_95",
            "S3_90",
            "S3_63",
            "H1_a",
            "N2_57",
        }
    else:
        upd = {"H1_a", "H1_b"}
    nlines = nlines.union(upd)
    lines = {s: core_lines[s] for s in nlines}
    fluxec = dict()
    for l in lines:
        fluxs = core_fit(sources, l, cal_red=cal_red, **kwargs)
        fluxec[l] = fluxs
    return fluxec, lines


def tem_den_red(sources, lines=None, temp=True, cal_red=None, **kwargs):
    """For stack of spectra in provided catalog calculates relevant line fluxes as well as electron density and temperature."""
    flx, finfo = get_core_fluxes(sources, ilines=lines, temp=temp, **kwargs)
    z = np.mean([v for s in sources if (v := s.get("z")) is not None])
    t = {k: 10000 for k in ["TO3", "TO2", "TS3", "TS2", "TN2"]}
    n = 40 * (1 + z) ** 1.5
    if not temp:
        cHbeta = red_const(flx, t=t, n=n) if cal_red is None else cal_red
        sflx = {l: flx[l] for l in lines}
        sfinfo = {l: finfo[l] for l in lines}
        return t, n, fredd(sflx, sfinfo, cHbeta)
    else:
        dt = 10000
        dn = 10000
        ite = 0
        while (dt > 100 or dn > 10) and ite < 10:
            cHbeta = red_const(flx, t=t, n=n)
            flc = fredd(flx, finfo, cHbeta)
            tn, nn = tem_den(flc, t0=t, n0=n)
            dt = np.nanmean([abs(tn[v] - t[v]) for v in tn])
            dn = abs(nn - n)
            t = tn
            n = nn
            ite += 1
        # print(f"\r\033[KElectron values: n:{n}, {t}.\n")
        # print(f"\r\033[KFluxes: {flx}.")
        return t, n, flc


def tem_den(fluxes, n0=400, t0={k: 10000 for k in ["TO3", "TO2", "TS3", "TS2", "TN2"]}):
    """From the specified line fluxes and initial guesses calculates density and temperatures (for different ions) of electron gas."""
    fluxec = dict()
    for v in fluxes.values():
        fluxec.update(v)

    if not np.isnan(t0["TO3"]):
        T0 = t0["TO3"]
    elif not np.isnan(t0["TS3"]):
        T0 = t0["TS3"]
    else:
        T0 = np.nanmean(t0.values())

    v1 = 0.0 if (v := fluxec.get("S2_6716A")) is None else v
    v2 = 0.0 if (v := fluxec.get("S2_6731A")) is None else v
    v3 = 0.0 if (v := fluxec.get("O2_3726A")) is None else v
    v4 = 0.0 if (v := fluxec.get("O2_3729A")) is None else v
    if v1 and v2 and v1 != v2:
        S2 = pn.Atom("S", 2)
        n = S2.getTemDen(v1 / v2, tem=T0, to_eval="L(6716)/L(6731)")
    elif v3 and v4 and v3 != v4:
        O2 = pn.Atom("O", 2)
        n = O2.getTemDen(v3 / v4, tem=T0, to_eval="L(3726)/L(3729)")
    else:
        n = n0

    match n:
        case _ if n < 20:
            n = 20
        case _ if n > 3000:
            n = 3000
        case _ if not np.isfinite(n):
            n = n0
        case _:
            pass

    t = {"TO3": np.nan, "TO2": np.nan, "TS3": np.nan, "TS2": np.nan, "TN2": np.nan}

    v1 = 0.0 if (v := fluxec.get("O2_7319A")) is None else v
    v2 = 0.0 if (v := fluxec.get("O2_7330A")) is None else v
    v3 = 0.0 if (v := fluxec.get("O2_3726A")) is None else v
    v4 = 0.0 if (v := fluxec.get("O2_3729A")) is None else v
    if (v1 or v2) and (v3 or v4):
        v1s = v1 if v1 == v2 else v1 + v2
        v3s = v3 if v3 == v4 else v3 + v4
        O2 = pn.Atom("O", 2)
        tl = O2.getTemDen(
            v1s / v3s, den=n, to_eval="(L(7319)+L(7330))/(L(3726)+L(3729))"
        )
        t["TO2"] = tl

    v1 = 0.0 if (v := fluxec.get("O3_4363A")) is None else v
    v2 = 0.0 if (v := fluxec.get("O3_5007A")) is None else v
    v3 = 0.0 if (v := fluxec.get("O3_4959A")) is None else v
    if v1 and (v2 or v3):
        v2 = v2 if v2 else v3 * 3
        v3 = v3 if v3 else v2 / 3
        v2s = v2 + v3
        O3 = pn.Atom("O", 3)
        tl = O3.getTemDen(v1 / v2s, den=n, to_eval="L(4363)/(L(5007)+L(4959))")
        t["TO3"] = tl

    v1 = 0.0 if (v := fluxec.get("S2_4069A")) is None else v
    v2 = 0.0 if (v := fluxec.get("S2_4076A")) is None else v
    v3 = 0.0 if (v := fluxec.get("S2_6716A")) is None else v
    v4 = 0.0 if (v := fluxec.get("S2_6731A")) is None else v
    if (v1 or v2) and (v3 or v4):
        v1 = v1 if v1 else v2 * 3
        v2 = v2 if v2 else v1 / 3
        v1s = v1 if v1 == v2 else v1 + v2
        v3s = v3 if v3 == v4 else v3 + v4
        S2 = pn.Atom("S", 2)
        tl = S2.getTemDen(
            v1s / v3s, den=n, to_eval="(L(4069)+L(4076))/(L(6716)+L(6731))"
        )
        t["TS2"] = tl

    v1 = 0.0 if (v := fluxec.get("S3_6312A")) is None else v
    v2 = 0.0 if (v := fluxec.get("S3_9531A")) is None else v
    v3 = 0.0 if (v := fluxec.get("S3_9069A")) is None else v
    if v1 and (v2 or v3):
        v2 = v2 if v2 else v3 * 2.44
        v3 = v3 if v3 else v2 / 2.44
        v2s = v2 + v3
        S3 = pn.Atom("S", 3)
        tl = S3.getTemDen(v1 / v2s, den=n, to_eval="L(6312)/(L(9531)+L(9069))")
        t["TS3"] = tl

    v1 = 0.0 if (v := fluxec.get("N2_5755A")) is None else v
    v2 = 0.0 if (v := fluxec.get("N2_6548A")) is None else v
    v3 = 0.0 if (v := fluxec.get("N2_6584A")) is None else v
    if v1 and (v2 or v3):
        v2 = v2 if v2 else v3 / 2.9
        v3 = v3 if v3 else v2 * 2.9
        v2s = v2 + v3
        N2 = pn.Atom("N", 2)
        tl = N2.getTemDen(v1 / v2s, den=n, to_eval="L(5755)/(L(6548)+L(6584))")
        t["TN2"] = tl

    to = dict()
    while set(to.values()) != set(t.values()):
        to = t.copy()
        if np.isnan(t["TO2"]) and not np.isnan(t["TO3"]) and not np.isnan(n):
            t["TO2"] = (
                10**4
                * (1.2 + 0.002 * n + 4.2 / n)
                / (10**4 / t["TO3"] + 0.08 + 0.003 * n + 2.5 / n)
            )
        if np.isnan(t["TS3"]) and not np.isnan(t["TO3"]):
            t["TS3"] = 1.19 * t["TO3"] - 3200
        if np.isnan(t["TN2"]) and not np.isnan(t["TO3"]):
            t["TN2"] = 1.85 / (1 / t["TO3"] + 0.72 / 10**4)
        if np.isnan(t["TO3"]) and not np.isnan(t["TO2"]):
            t["TO3"] = 1 / (2 / t["TO2"] - 0.8 / 10**4)
        if np.isnan(t["TS2"]) and not np.isnan(t["TO2"]):
            t["TS2"] = 0.71 * t["TO2"] + 1200
        if np.isnan(t["TO3"]) and not np.isnan(t["TS3"]):
            t["TO3"] = (t["TS3"] + 3200) / 1.19
        if np.isnan(t["TO2"]) and not np.isnan(t["TS2"]):
            t["TO2"] = (t["TS2"] - 1200) / 0.71
        if np.isnan(t["TO3"]) and not np.isnan(t["TN2"]):
            t["TO3"] = 1 / (1.85 / t["TN2"] - 0.72 / 10**4)

    for v in t:
        match t[v]:
            case _ if t[v] < 2000:
                t[v] = 2000
            case _ if t[v] > 40000:
                t[v] = 40000
            case _ if not np.isfinite(t[v]):
                t[v] = t0[v]
            case _:
                pass
    return t, n


def abunds(fluxes, t, den, N_over_O=True):
    """From specified line fluxes and electron density and temperatures calculates Sulphur, Nitrogen and Oxygen abundances via direct method/atomic modelling."""
    flx = dict()
    for v in fluxes.values():
        flx.update(v)
    ions = dict()
    abun = dict()

    vb = 0.0 if (v := flx.get("H1r_4861A")) is None else v

    if not np.isnan(t["TO3"]):
        thO = t["TO3"]
    elif not np.isnan(t["TS3"]):
        thO = t["TS3"]
    else:
        thO = np.nanmean(t.values())

    O3ab = 0
    v1 = 0.0 if (v := flx.get("O3_5007A")) is None else v
    v2 = 0.0 if (v := flx.get("O3_4959A")) is None else v
    if (v1 or v2) and vb:
        v1 = v1 if v1 else v2 * 3
        v2 = v2 if v2 else v1 / 3
        O3 = pn.Atom("O", 3)
        O3ab += O3.getIonAbundance(
            int_ratio=v1 + v2,
            tem=thO,
            den=den,
            to_eval="L(5007)+L(4959)",
            Hbeta=vb,
        )
    ions["O3"] = O3ab

    if not np.isnan(t["TO2"]):
        tlO = t["TO2"]
    elif not np.isnan(t["TN2"]):
        tlO = t["TN2"]
    elif not np.isnan(t["TS2"]):
        tlO = t["TS2"]
    else:
        tlO = np.nanmean(t.values())

    O2ab = 0
    O2co = 0
    v1 = 0.0 if (v := flx.get("O2_3726A")) is None else v
    v2 = 0.0 if (v := flx.get("O2_3729A")) is None else v
    if (v1 or v2) and vb:
        O2 = pn.Atom("O", 2)
        O2ab += O2.getIonAbundance(
            int_ratio=v1 + v2,
            tem=tlO,
            den=den,
            to_eval="L(3726)+L(3729)",
            Hbeta=vb,
        )
        O2co += 1
    v1 = 0.0 if (v := flx.get("O2_7319A")) is None else v
    v2 = 0.0 if (v := flx.get("O2_7330A")) is None else v
    if (v1 or v2) and vb:
        O2 = pn.Atom("O", 2)
        O2ab += O2.getIonAbundance(
            int_ratio=v1 + v2,
            tem=tlO,
            den=den,
            to_eval="L(7319)+L(7330)",
            Hbeta=vb,
        )
        O2co += 1
    ions["O2"] = O2ab / O2co if O2co > 0 else 0

    if not np.isnan(t["TS2"]):
        tlS = t["TS2"]
    elif not np.isnan(t["TO2"]):
        tlS = t["TO2"]
    elif not np.isnan(t["TN2"]):
        tlS = t["TN2"]
    else:
        tlS = np.nanmean(t.values())

    S2ab = 0
    v1 = 0.0 if (v := flx.get("S2_6716A")) is None else v
    v2 = 0.0 if (v := flx.get("S2_6731A")) is None else v
    if (v1 or v2) and vb:
        S2 = pn.Atom("S", 2)
        S2ab += S2.getIonAbundance(
            int_ratio=v1 + v2,
            tem=tlS,
            den=den,
            to_eval="L(6731)+L(6716)",
            Hbeta=vb,
        )
    ions["S2"] = S2ab

    if not np.isnan(t["TS3"]):
        tmS = t["TS3"]
    elif not np.isnan(t["TO3"]):
        tmS = t["TO3"]
    else:
        tmS = np.nanmean(t.values())

    S3ab = 0
    S3co = 0
    v1 = 0.0 if (v := flx.get("S3_9531A")) is None else v
    v2 = 0.0 if (v := flx.get("S3_9069A")) is None else v
    if (v1 or v2) and vb:
        v1 = v1 if v1 else v2 * 2.44
        v2 = v2 if v2 else v1 / 2.44
        S3 = pn.Atom("S", 3)
        S3ab += S3.getIonAbundance(
            int_ratio=v1 + v2,
            tem=tmS,
            den=den,
            to_eval="L(9531)+L(9069)",
            Hbeta=vb,
        )
        S3co += 1
    v1 = 0.0 if (v := flx.get("S3_6312A")) is None else v
    if v1 and vb:
        S3 = pn.Atom("S", 3)
        S3ab += S3.getIonAbundance(
            int_ratio=v1,
            tem=tmS,
            den=den,
            to_eval="L(6312)",
            Hbeta=vb,
        )
        S3co += 1
    ions["S3"] = S3ab / S3co if S3co > 0 else 0

    if not np.isnan(t["TN2"]):
        tlN = t["TN2"]
    elif not np.isnan(t["TO2"]):
        tlN = t["TO2"]
    elif not np.isnan(t["TS2"]):
        tlN = t["TS2"]
    else:
        tlN = np.nanmean(t.values())

    N2ab = 0
    v1 = 0.0 if (v := flx.get("N2_6548A")) is None else v
    v2 = 0.0 if (v := flx.get("N2_6584A")) is None else v
    if (v1 or v2) and vb:
        v1 = v1 if v1 else v2 / 2.9
        v2 = v2 if v2 else v1 * 2.9
        N2 = pn.Atom("N", 2)
        N2ab += N2.getIonAbundance(
            int_ratio=v1 + v2,
            tem=tlN,
            den=den,
            to_eval="L(6548)+L(6584)",
            Hbeta=vb,
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


def red_const(
    flxs, t={k: 10000 for k in ["TO3", "TO2", "TS3", "TS2", "TN2"]}, n=500, **kwargs
):
    """Calculates an estimate of the extintion cHbeta value from provided line fluxes and electron gas density and temperatures."""
    if not np.isnan(t["TO3"]):
        T = t["TO3"]
    elif not np.isnan(t["TS3"]):
        T = t["TS3"]
    else:
        T = np.nanmean(t.values())

    lT = np.log10(T)
    wavs = {
        6563: 10.35 - 3.254 * lT + 0.3457 * lT**2,
        4340: 0.0254 + 0.1922 * lT - 0.0204 * lT**2,
        4102: -0.07132 + 0.1436 * lT - 0.0153 * lT**2,
    }
    if (f := flxs.get("H1_b")) is not None:
        flub = sum(f.values())
    else:
        return 0.0
    if flub:
        flux = dict()
        flux[6563] = f["H1r_6563A"] if (f := flxs.get("H1_a")) is not None else 0.0
        flux[4340] = sum(f.values()) if (f := flxs.get("H1_g")) is not None else 0.0
        flux[4102] = sum(f.values()) if (f := flxs.get("H1_d")) is not None else 0.0
        for f in flux:
            flux[f] /= flub
        consts = []
        for k in wavs:
            if np.isfinite(flux[k]) and flux[k] > 0:
                rc.setCorr(obs_over_theo=flux[k] / wavs[k], wave1=k, wave2=4862)
                consts.append(rc.cHbeta)
        # print("cHbeta values: " + str(consts))
        return consts[0] if consts else 0.0
    else:
        return 0.0


def fredd(fluxes, flinfo, cal_red):
    """Calibrates provided value for gas attenuation via specified cHbeta value."""
    fluxca = dict()
    rc.cHbeta = cal_red
    for f in fluxes:
        fluxli = dict()
        for n in fluxes[f]:
            corr = rc.getCorrHb(flinfo[f][0][n] * 10**4)
            fluxli[n] = fluxes[f][n] * corr
        fluxca[f] = fluxli
    return fluxca


def core_fit(sources, name, cal_red=None, **kwargs):
    """Convenience function to obtain line fluxes of named region from stack of spectra in passed catalogue."""
    l = core_lines[name]
    return fit_lines(sources, l[1], l[0], dwidth=l[2], cal_red=cal_red, **kwargs)


def indiv_extract(source, mline):
    """Extract requested line fluxes stored in provided catalogue entry.

    This function imposes threshold >10^(-23.5) on the extracted fluxes, otherwise returns them as nan. This will work correctly only if the stored values have units W/m^2.
    """
    outd = True
    if type(mline) is not dict:
        mline = {"0": mline}
        outd = False
    fluxes = dict()
    for k, l in mline.items():
        if (n := f"rec_{k}") in source.keys():
            if (v := source[n]) is not None and np.isfinite(v) and v > 10 ** (-23.5):
                fluxes[k] = source[f"rec_{k}"]
            else:
                fluxes[k] = np.nan
        else:
            fluxes[k] = np.nan
    if outd:
        return fluxes
    else:
        return next(f for f in fluxes.values())


def fit_lines(
    sources,
    lines,
    mline,
    reso=300,
    base="../Data/Npy_v4/",
    typ="median",
    dwidth=8,
    manual=False,
    indiv=True,
    R=850,
    hsh=True,
    **kwargs,
):
    """Stack spectra of sources in passed catalog in specified region, fit the region with Gaussian profiles and extract resulting fluxes."""
    hsh = hsh if len(sources) > 1 else False
    if hsh:
        args = (lines, mline, reso, base, typ, dwidth, R)
        hax = hashed(sources, "fit_lines", *args, **kwargs)
        if hax in glob_hash:
            return glob_hash[hax]
    if not any(sources):
        return lf.flux_nan(mline)
    if len(sources) == 1 and indiv:
        return indiv_extract(sources[0], mline)
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
        flux = lf.flux_extract(fit, mline, grat=grat)
        out = flux
    else:
        out = lf.flux_nan(mline)
    if hsh:
        glob_hash[hax] = out
    return out


def iprocess(funct, srs, calib, zs, vs, it, val, sind, glob, **kwargs):
    """Single process function calculating specified method for each spectra in the passed catalogue."""
    if glob is not None:
        global glob_hash
        glob_hash = glob
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
    # if glob is not None:
    #    glob.update(glob_hash)


def bprocess(funct, srs, ind, vis, glob, **kwargs):
    """Single process function calculating specified method for each catalog in passed list."""
    if glob is not None:
        global glob_hash
        glob_hash = glob
    for sr in srs:
        vi = np.flip(funct(sr, **kwargs)[ind])
        vis.append(vi)
    # if glob is not None:
    #    glob.update(glob_hash)


def boots_stat(
    funct,
    sources,
    ite=200,
    calib=True,
    manual=False,
    hsh=True,
    seed=glob_seed,
    **kwargs,
):
    """Calculate a specified function for a provided catalogue and by subsampling the catalogue estimate related bootstrapping statistics and errors."""
    if hsh:
        global glob_hash
        hax = hashed(sources, "boots_stat", funct, ite, calib, **kwargs)
        if hax in glob_hash:
            return glob_hash[hax]
    ind = 1 if calib else 0
    r = np.random.RandomState(seed)
    vs = np.flip(funct(sources, manual=manual, indiv=False, **kwargs)[ind])
    vals = np.full((ite, len(vs)), np.nan)
    manag = Manager()
    vis = manag.list()
    glob = manag.dict(glob_hash) if hsh else None
    i = 0
    nos = 5
    proc = int(cpu_count() * 1.5)
    active = []
    while i < ite or len(active) > 0:
        pr_no = -((i - ite) // nos)
        if proc > len(active) and i < ite:
            for l in range(min(pr_no, proc - len(active))):
                print(
                    f"\r\033[KBootstrapping {i} out of {ite}. (Buffer len.: {len(glob_hash)})",
                    end="",
                )
                noi = min(nos, ite - i)
                rsourcs = [r.choice(sources, size=len(sources)) for l in range(noi)]
                args = (funct, rsourcs, ind, vis, glob)
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
    if hsh:
        glob_hash = glob._getvalue()
    for i, vi in enumerate(vis):
        vals[i, : vi.shape[0]] = vi[: vals.shape[1]]
    vals = np.nan_to_num(vals, nan=np.nan, posinf=np.nan, neginf=np.nan)
    medians = np.nanmedian(vals, axis=0)
    err33 = np.nanpercentile(vals, 33, axis=0)
    err67 = np.nanpercentile(vals, 67, axis=0)
    out = vs, medians, [medians - err33, err67 - medians]
    if hsh:
        glob_hash[hax] = out
    return out


def indiv_stat(funct, sources, val="z", calib=True, hsh=False, **kwargs):
    """Calculate specified function for each spectra in the passed catalogue. And obtain the statistics related to the set of resulting values."""
    if hsh:
        global glob_hash
        hax = hashed(sources, "indiv_stat", funct, calib, val, **kwargs)
        if hax in glob_hash:
            return glob_hash[hax]
    no = len(sources)
    manag = Manager()
    va = manag.dict()
    zs = manag.dict()
    vs = manag.dict()
    glob = manag.dict(glob_hash) if hsh else None
    i = 0
    nos = 50
    it = manag.Value(int, 0)
    proc = cpu_count() * 2
    active = []
    while it.value < no and i < len(sources) or len(active) > 0:
        pr_no = -((-min(len(sources) - i, no - it.value)) // nos)
        if proc > len(active) and (it.value < no and i < len(sources)):
            for l in range(min(pr_no, proc - len(active))):
                print(
                    f"\r\033[KCalculating {i} out of {no} points. (Buffer len.: {len(glob_hash)})",
                    end="",
                )
                sr = sources[i : i + nos]
                args = (funct, sr, calib, zs, vs, it, val, i, glob)
                t = Process(target=iprocess, args=args, kwargs=kwargs)
                t.start()
                active.append(t)
                i += len(sr)
        for t in active:
            if not t.is_alive():
                t.terminate()
                active.remove(t)
        time.sleep(0.1)
    if hsh:
        glob_hash = glob._getvalue()
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
        print(f"\nNeeded {i} out of {len(sources)} for {it.value} results.")
        out = (zs, vs), (
            medians,
            np.nan_to_num([medians - err33, err67 - medians], nan=0.0),
        )
    else:
        out = vs
    if hsh:
        glob_hash[hax] = out
    return out


def hashed(sources, *args, **kwargs):
    """Calculate sha1 hash of the provided catalogue, jointly with any other arguments or keyword arguments."""
    hsh = sha1()
    # keys = ['srcid', 'ra', 'dec', 'root', 'file', 'grat','grat_orig', 'comment', 'comment_old']
    keys = ["root", "file", "grat"]
    for s in sources:
        for k in keys:
            hsh.update((k + repr(s.get(k))).encode())
    for ar in args:
        hsh.update(repr(ar).encode())
    for i in kwargs.items():
        hsh.update(repr(i).encode())
    return hsh.hexdigest()


def mark_agn(sources):
    """For each source in the catalogue calculate N2 and R3 calibrations and using BPT diagram note whether they should be flagged as potential AGN candidates."""
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


def tem_den_legacy(fluxec):
    """Legacy method for calculating estimations of electron gas density and temperature from passed fluxes."""
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
