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
    galaxy = galaxy / np.nanmedian(galaxy)

    noise = np.full_like(galaxy, np.abs(np.nanmean(galaxy) / source["sn50"]))

    sps_name = "fsps"
    ppxf_dir = Path(lib.__file__).parent
    basename = f"spectra_{sps_name}_9.0.npz"
    filename = ppxf_dir / "sps_models" / basename
    if not filename.is_file():
        url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/" + basename
        request.urlretrieve(url, filename)
    FWHM_temp = 2.51
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


def ppxf_fitting_multi(start_in=0):
    a = catalog.fetch_json("../catalog.json")["sources"]

    manag = Manager()
    proc = cpu_count()
    active = []
    latest = start_in
    while latest < len(a) or len(active) > 0:
        pr_no = min(proc, len(a) - latest)
        if len(active) < pr_no:
            for i in range(pr_no - len(active)):
                args = (a[latest + 1],)
                kwargs = {"silent": True}
                t = Process(target=ppxf_process, args=args, kwargs=kwargs)
                t.start()
                print(f"\rRunning spectra {latest} \tout of {len(a)}", end="")
                active.append(t)
                latest += 1
        for t in active:
            if not t.is_alive():
                t.terminate()
                active.remove(t)
        time.sleep(0.1)


def ppxf_process(source, save=True, **kwargs):
    spectrum = spectr.get_spectrum_n(source)
    try:
        pp = ppxf_fit(source, spectrum, **kwargs)
        wav = pp.lam / 10**4
        bestfit = pp.bestfit
        if pp.gas_bestfit is not None:
            bestfit = bestfit - pp.gas_bestfit
        bestf = spectr.resample([wav, bestfit], spectrum[0])
        spectn = [spectrum[0], spectrum[1] - bestf[1]]
    except:
        print(f'Spectrum of {source["srcid"]} not subtracted.')
        return None
    if save:
        base = "../Data/Subtracted/"
        spectr.save_npy(source, spectn, base=base)


if __name__ == "__main__":
    ppxf_fitting_multi(start_in=400)
