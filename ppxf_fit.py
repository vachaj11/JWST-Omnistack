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


def ppxf_fit(source, spectrum, silent=True, plot=False):
    lam, galaxy = spectrum[0] * 10**4, spectrum[1]
    old_stdout = sys.stdout
    if silent:
        sys.stdout = open(os.devnull, "w")

    match source["grat"]:
        case "prism":
            R = 100
        case x if x[-1:] == "h":
            R = 2700
        case x if x[-1:] == "m":
            R = 1000
        case _:
            R = len(lam)

    galaxy, ln_lam, velscale = util.log_rebin(lam, galaxy)
    lam = np.exp(ln_lam)
    mask = np.isfinite(galaxy)

    FWHM_gal = np.sqrt(lam.min() * lam.max()) / R

    z = source["z"]
    lam /= 1 + z
    FWHM_gal /= 1 + z
    galaxy = galaxy

    noise = np.full_like(galaxy, np.abs(np.nanmean(galaxy) / source["sn50"]))

    sps_name = "fsps"
    ppxf_dir = Path(lib.__file__).parent
    basename = f"spectra_{sps_name}_9.0.npz"
    filename = ppxf_dir / "sps_models" / basename
    if not filename.is_file():
        url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/" + basename
        request.urlretrieve(url, filename)
    sps = lib.sps_lib(filename, velscale)

    reg_dim = sps.templates.shape[1:]
    stars_templates = sps.templates.reshape(sps.templates.shape[0], -1)

    gas_templates = np.array([[] for i in range(stars_templates.shape[0])])
    gas_names = np.array([])
    line_wave = np.array([])
    for laml, fluxl in spectr.useful_sp_parts((lam, galaxy)):
        lam_range_gal = [np.min(laml), np.max(laml)]
        g_templates, g_names, l_wave = util.emission_lines(
            sps.ln_lam_temp, lam_range_gal, FWHM_gal, tie_balmer=1
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

    galaxy = np.where(np.isfinite(galaxy), galaxy, 10**5)
    pp = ppxf(
        templates,
        galaxy,
        noise,
        velscale,
        starts,
        moments=moments,
        degree=-1,
        mdegree=-1,
        lam=lam,
        lam_temp=sps.lam_temp,
        reg_dim=reg_dim,
        component=component,
        gas_component=gas_component,
        reddening=0,
        gas_reddening=gas_redenning,
        gas_names=gas_names,
        mask=mask,
    )
    if plot:
        plt.figure(figsize=(15, 5))
        pp.plot()
        plt.title(f"pPXF fit with {sps_name} SPS templates")
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


def ppxf_fitting_multi(sources, start_in=0):
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
                kwargs = {"silent": True}
                t = Process(target=ppxf_process, args=args, kwargs=kwargs)
                t.start()
                print(f"\rRunning spectra {latest} \tout of {len(a)-1}", end="")
                active.append(t)
        for t in active:
            if not t.is_alive():
                t.terminate()
                active.remove(t)
        time.sleep(0.5)


def ppxf_process(source, save=True, **kwargs):
    spectrum = spectr.get_spectrum_n(source)
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
    if save:
        base = "../Data/Continuum/"
        spectr.save_npy(source, [wav, bestfit], base=base)
    return pp


def continuum(
    source,
    nconv=5,
    save=True,
    plot=False,
    bi="../Data/Continuum/",
    bo="../Data/Subtracted/",
    convolv=True,
):
    spectrum = spectr.get_spectrum_n(source)
    continuu = spectr.get_spectrum_n(source, base=bi)
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
    if save:
        spectr.save_npy(source, subtc, base=bo)
    return subtc


def smooth_to_cont(
    source,
    nconv=40,
    save=True,
    plot=False,
    bi="../Data/Npy/",
    bo="../Data/Continuum_b/",
):
    spectrum = spectr.get_spectrum_n(source)
    if spectrum is None:
        print(f'Spectrum not available for {source["srcid"]}')
        return None
    match source["grat"]:
        case "prism":
            R = 100
        case x if x[-1:] == "h":
            R = 2700
        case x if x[-1:] == "m":
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
    if save:
        spectr.save_npy(source, resa, base=bo)
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
