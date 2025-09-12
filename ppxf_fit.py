"""Holds methods allowing for ppxf fitting, various estimations of spectra continuum and continuum subtraction."""

import matplotlib

matplotlib.use("qtagg")

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

from multiprocessing import Manager, Process, cpu_count
from pathlib import Path
from urllib import request

import matplotlib.pyplot as plt
import numpy as np
import ppxf.ppxf_util as util
import ppxf.sps_util as lib
from astropy.convolution import Gaussian1DKernel, convolve
from ppxf.ppxf import ppxf

import catalog
import spectr


def ppxf_fit(sources, spectra, silent=True, plot=False):
    """Despite the list formatisation, will function reliably only with single source/spectra passed."""
    if type(sources) is not list:
        sources = [sources]
    if type(spectra) is not list or len(spectra[0]) != 2:
        spectra = [spectra]
    if len(sources) != len(spectra):
        sources = sources[: len(spectra)]

    old_stdout = sys.__stdout__
    if silent:
        sys.stdout = open(os.devnull, "w")
    galaxy_s = np.array([])
    lam_s = np.array([])
    velscale_s = np.array([])
    mask_s = np.array([], dtype=bool)
    noise_s = np.array([])
    FWHM_gal_s = np.array([])
    for so, sp in zip(sources, spectra):
        lam, galaxy = sp[0] * 10**4, sp[1]
        match so["grat"]:
            case int() | float():
                R = R
            case "prism":
                R = 100
            case x if x[-1:] == "h":
                R = 2700
            case x if x[-1:] == "m":
                R = 1000
            case _:
                R = len(lam)
        z = so["z"]
        vl = velscale_s[-1] if velscale_s.size else None
        galaxy, ln_lam, velscale = util.log_rebin(lam, galaxy, velscale=vl)
        lam = np.exp(ln_lam) / (1 + z)
        mask = np.isfinite(galaxy)
        noise = np.full_like(galaxy, np.abs(np.nanmean(galaxy) / so["sn50"]))
        FWHM_gal = np.full_like(galaxy, np.sqrt(lam.min() * lam.max()) / R / (1 + z))
        velscale = np.full_like(galaxy, velscale)

        galaxy_s = np.concatenate([galaxy_s, galaxy])
        lam_s = np.concatenate([lam_s, lam])
        velscale_s = np.concatenate([velscale_s, velscale])
        mask_s = np.concatenate([mask_s, mask])
        noise_s = np.concatenate([noise_s, noise])
        FWHM_gal_s = np.concatenate([FWHM_gal_s, FWHM_gal])
    sps_name = "fsps"
    if sps_name in ["fsps", "galaxev"]:
        w = lam_s < 7400
        galaxy_s = galaxy_s[w]
        lam_s = lam_s[w]
        velscale_s = velscale_s[w]
        mask_s = mask_s[w]
        noise_s = noise_s[w]
        FWHM_gal_s = FWHM_gal_s[w]

    ppxf_dir = Path(lib.__file__).parent
    basename = f"spectra_{sps_name}_9.0.npz"
    filename = ppxf_dir / "sps_models" / basename
    if not filename.is_file():
        url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/" + basename
        request.urlretrieve(url, filename)
    sps = lib.sps_lib(filename, np.mean(velscale_s))

    reg_dim = sps.templates.shape[1:]
    stars_templates = sps.templates.reshape(sps.templates.shape[0], -1)

    gas_templates = np.array([[] for i in range(stars_templates.shape[0])])
    gas_names = np.array([])
    line_wave = np.array([])
    for laml, fluxl, FWHM_gall in spectr.useful_sp_parts((lam_s, galaxy_s, FWHM_gal_s)):
        lam_range_gal = [np.min(laml), np.max(laml)]
        g_templates, g_names, l_wave = util.emission_lines(
            sps.ln_lam_temp, lam_range_gal, np.mean(FWHM_gall), tie_balmer=1
        )
        gas_templates = np.concatenate([gas_templates, g_templates], axis=1)
        gas_names = np.concatenate([gas_names, g_names])
        line_wave = np.concatenate([line_wave, l_wave])
    templates = np.column_stack([stars_templates, gas_templates])

    start = [1200.0, 200.0]
    n_stars = stars_templates.shape[1]
    n_gas = len(gas_names)
    component = [0] * n_stars + [1] * n_gas
    gas_component = np.array(component) > 0
    mome = int(bool(n_stars)) + int(bool(n_gas))
    if mome == 1:
        moments = 2
        starts = start
    else:
        moments = [2, 2]
        starts = [start, start]
    gas_redenning = 0
    if gas_templates.size == 0:
        gas_names = None
        gas_component = None
        gas_redenning = None

    galaxy_s = np.where(np.isfinite(galaxy_s), galaxy_s, 10**5)
    pp = ppxf(
        templates,
        galaxy_s,
        noise_s,
        np.mean(velscale_s),
        starts,
        moments=moments,
        degree=-1,
        mdegree=-1,
        lam=lam_s,
        lam_temp=sps.lam_temp,
        reg_dim=reg_dim,
        component=component,
        gas_component=gas_component,
        reddening=0,
        gas_reddening=gas_redenning,
        gas_names=gas_names,
        mask=mask_s,
    )
    light_weights = pp.weights if gas_component is None else pp.weights[~gas_component]
    light_weights = light_weights.reshape(reg_dim)
    light_weights /= light_weights.sum()
    sps.mean_age_metal(light_weights)
    sps.mass_to_light(light_weights)
    if plot:
        plt.figure(figsize=(15, 5))
        pp.plot()
        plt.title(f"pPXF fit with {sps_name} SPS templates")
        plt.figure(figsize=(10, 3))
        sps.plot(light_weights)
        plt.title("Light Fraction")
        plt.tight_layout()
        plt.show()
    sys.stdout = old_stdout
    return pp


def ppxf_fitting_single(n=0):
    a = catalog.fetch_json("../catalog.json")["sources"]
    af = catalog.rm_bad(a)
    # af = [s for s in a if s['sn50']>4]
    # af = [s for s in af if s['grat']!='prism']
    source = af[n]
    ppxf_process(source, silent=False, plot=True)


def ppxf_fitting_multi(sources, start_in=0, **kwargs):
    a = sources
    manag = Manager()
    proc = cpu_count()
    active = []
    latest = start_in - 1
    while latest < len(a) - 1 or len(active) > 0:
        pr_no = min(proc, len(a) - 1 - latest)
        if len(active) < pr_no:
            for i in range(pr_no - len(active)):
                latest += 1
                args = (a[latest],)
                kwargs = {"silent": True, **kwargs}
                t = Process(target=ppxf_process, args=args, kwargs=kwargs)
                t.start()
                print(f"\rRunning spectra {latest} \tout of {len(a)-1}", end="")
                active.append(t)
        for t in active:
            if not t.is_alive():
                t.terminate()
                active.remove(t)
        time.sleep(0.1)


def ppxf_process(
    source, bi="../Data/Npy_v4/", bo="../Continuum_v4/", bs=None, **kwargs
):
    spectrum = spectr.get_spectrum(source, base=bi)
    try:
        pp = ppxf_fit(source, spectrum, **kwargs)
        z = source["z"]
        wav = pp.lam / 10**4 * (1 + z)
        bestfit = pp.bestfit
        if pp.gas_bestfit is not None:
            bestfit = bestfit - pp.gas_bestfit
        """
        bestf = spectr.resample([wav, bestfit], spectrum[0])
        spectn = [spectrum[0], spectrum[1] - bestf[1]]
        error = pp.error[0][0]*np.sqrt(pp.chi2) 
        v = pp.sol[0][0] 
        c = 299792.458
        z = (1 + source['z'])*np.exp(vpec/c) - 1
        dz = (1 + z)*error/c
        if np.abs(z-source['z']) > np.abs(dz):
            print('Redshift fit disagreeing with original.')
            print(f'New {z}, old {source["z"]}, chi2 {pp.chi2}')
        """
    except:
        print(f'Spectrum of {source["srcid"]} not subtracted.')
        return None
    if bo:
        base = bo
        spectr.save_npy(source, [wav, bestfit], base=base)
    if bs:
        continuum(source, bi=bo, bo=bs)
    return pp


def continuum(
    source,
    nconv=5,
    plot=False,
    bi="../Data/Continuum/",
    bo="../Data/Subtracted/",
    convolv=True,
):
    spectrum = spectr.get_spectrum(source)
    continuu = spectr.get_spectrum(source, base=bi)
    if spectrum is None or continuu is None:
        print(f'\r\033[KSpectrum not available for {source["srcid"]}', end="")
        return None
    if convolv:
        match source["grat"]:
            case "prism":
                R = 100
            case x if x[-1:] == "h":
                R = 2700
            case x if x[-1:] == "m":
                R = 1000
            case _:
                R = len(spectrum[0])
        FWHM_conv = np.sqrt(min(spectrum[0]) * max(spectrum[0])) / R * nconv
        convc = sp_conv(continuu, FWHM_conv)
        resac = spectr.resample(convc, spectrum[0])
    else:
        resac = spectr.resample(continuu, spectrum[0])
    subtc = [spectrum[0], spectrum[1] - resac[1]]
    if plot:
        plt.plot(spectrum[0], spectrum[1])
        plt.plot(resac[0], resac[1])
        # plt.plot(subtc[0],subtc[1])
        plt.show()
    if bo:
        spectr.save_npy(source, subtc, base=bo)
    return subtc


def smooth_to_cont(
    source,
    nconv=40,
    plot=False,
    bi="../Data/Npy_v4/",
    bo="../Data/Continuum_v4_b/",
    bs=None,
):
    spectrum = spectr.get_spectrum(source)
    if spectrum is None:
        print(f'Spectrum not available for {source["srcid"]}')
        return None
    match source["grat"]:
        case "prism":
            R = 100
        case x if x[-1:] == "h":
            R = 2700
        case x if x[-1:] == "m" and x[:1] == "g":
            R = 1000
        case _:
            R = len(spectrum[0])
    FWHM_conv = np.sqrt(min(spectrum[0]) * max(spectrum[0])) / R
    convc = sp_conv(spectrum, FWHM_conv * nconv)
    resac = spectr.resample(convc, spectrum[0])
    clip = spectr.clipping([spectrum[0], spectrum[1] - resac[1]], sup=3, sdo=3, ite=2)
    spcl = [spectrum[0], spectrum[1] * np.isfinite(clip[1])]
    conv = sp_conv(spcl, FWHM_conv * nconv)
    resa = spectr.resample(conv, spectrum[0])

    if plot:
        plt.plot(spectrum[0], spectrum[1])
        plt.plot(resac[0], resac[1])
        plt.plot(resa[0], resa[1])
        plt.show()
    if bo:
        spectr.save_npy(source, resa, base=bo)
    if bs:
        continuum(source, bi=bo, bo=bs, convolv=False)
    return resa


def sp_conv(spectrum, fwhm):
    rastr = len(spectrum[0]) * 3
    rang = [min(spectrum[0]), max(spectrum[0])]
    fwno = fwhm / (rang[1] - rang[0]) * rastr
    unifw = spectr.wave_sp(rang[0], rang[1], rastr)
    unifc = spectr.resample(spectrum, unifw)
    convc = conv1D(unifc[1], fwno)
    return [unifc[0], convc]


def conv1D(array, fwhm):
    sig = fwhm / (2 * np.sqrt(2 * np.log(2)))
    ker = Gaussian1DKernel(sig)
    convolved = convolve(array, ker, boundary="extend")
    return convolved


if __name__ == "__main__":
    a = catalog.fetch_json("../catalog.json")["sources"]
    ppxf_fitting_multi(a, start_in=0)
