import os
import time

import astropy.io.fits as fits
import numpy as np

glob = dict()


def get_spectrum(source, base="../Data/Fits/"):
    path = base + source["root"] + "/" + source["file"]
    try:
        x = fits.open(path)
        wave = x[1].data["wave"]
        flux = x[1].data["flux"]
        # wave, flux = clippint([wave, flux])
        x.close()
        return [wave, flux]
    except:
        return None


def get_spectrum_n(source, base="../Data/Npy/"):
    path = base + source["root"] + "/" + source["file"]
    path = path[:-5] + ".npy"
    try:
        if path in glob:
            return glob[path]
        else:
            spectr = np.load(path)
            glob[path] = spectr
            # spectr = clipping(spectr)
            return spectr
    except:
        return None


def save_npy(source, spectrum, base="../Data/Npy/"):
    path = base + source["root"] + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    path = path + source["file"][:-5] + ".npy"
    np.save(path, np.array(spectrum))


def rm_npy(source, base="../Data/Fits/"):
    path = base + source["root"] + "/"
    path = path + source["file"][:-5] + ".npy"
    try:
        os.remove(path)
    except:
        print(f"File {path} not found and not removed.")


def clipping(spectrum, sup=30, sdo=15, ite=1):
    flux = spectrum[1]
    mean = np.nanmean(flux)
    std = np.nanstd(flux)
    for i in range(ite):
        mask = ((mean + std * sup) > flux) * (flux > (mean - std * sdo))
        flux = np.where(mask, flux, np.nan)
    return [spectrum[0], flux]


def remove_nan(spectrum):
    wav, flux = spectrum
    wavn = []
    fluxn = []
    for w, f in zip(wav, flux):
        if np.isfinite(f):
            wavn.append(w)
            fluxn.append(f)
    return np.array([np.array(wavn), np.array(fluxn)])


def resampled_spectra(sources, rang, reso, **kwargs):
    spectra = []
    sourn = []
    space = wave_sp(rang[0], rang[1], reso)
    for i, s in enumerate(sources):
        try:
            t0 = time.time()
            sp = get_spectrum_n(s, **kwargs)
            t1 = time.time()
            if sp is not None:
                sr = resample(sp, space, s["z"])
                t2 = time.time()
                spectra.append(sr)
                sourn.append(s)
                print(
                    f'\r\033[KResampled: {s["srcid"]:<14}\t({i} of {len(sources)})\tTime: {(t2-t1)/(t1-t0):.2f}',
                    end="",
                )
            else:
                print(
                    f'\r\033[KNo spectral file found for: {str(s["srcid"]):<14}\t({i} of {len(sources)})',
                    end="",
                )
        except:
            print(
                f'\r\033[KSomething strange with: {str(s["srcid"]):<14}\t({i} of {len(sources)})',
                end="",
            )
    print("")
    return spectra, sourn


def wave_sp(minw, maxw, points):
    return np.linspace(minw, maxw, points)


def useful_range(spectrum):
    wav = spectrum[0]
    ins = -1
    inc = False
    wavr = []
    for i, v in enumerate(~np.isnan(spectrum[1])):
        if v:
            if not inc:
                inc = True
                ins = i
        else:
            if inc:
                inc = False
                wavr.append((wav[ins], wav[i - 1]))
    return wavr


def useful_sp_parts(spectrum):
    x = np.array(spectrum)
    spectra = np.split(x, np.argwhere(~np.isfinite(x[1])).T[0], axis=1)
    spectran = [s[:, 1:] for s in spectra if s[:, 1:].size > 0]
    return spectran


def resample_legacy(spectrum, space, z=0.0, normalise=False):
    news = []
    flux = spectrum[1]
    for w in space * (1 + z):
        spect = spectrum[0] - w
        imin = np.where(~(spect > 0), spect, -np.inf).argmax()
        imax = np.where(~(spect < 0), spect, np.inf).argmin()
        if imax > 0 and imin < len(spect):
            val = flux[imin] + (flux[imax] - flux[imin]) * (w - spectrum[0][imin]) / (
                spectrum[0][imax] - spectrum[0][imin]
            )
            news.append(val)
        else:
            news.append(np.nan)
    nspectrum = np.array([space, news])
    if normalise:
        return spect_norm(nspectrum)
    else:
        return nspectrum


def resample(spectrum, space, z=0.0, normalise=False):
    wav = spectrum[0]
    flux = spectrum[1]
    z_space = space * (1 + z)
    flux_n = np.interp(z_space, wav, flux, left=np.nan, right=np.nan)
    nspectrum = np.array([space, flux_n])
    if normalise:
        return spect_norm(nspectrum)
    else:
        return nspectrum


def spect_norm(spectrum, frequency=True):
    if not frequency:
        diffwav = np.abs(np.diff(spectrum[0]))
        flux = np.nansum(diffwav * np.array(spectrum[1])[:-1])
        return np.array([spectrum[0], spectrum[1] / flux])
    else:
        c = 1
        frequencies = c / np.array(spectrum[0])
        diffreq = np.abs(np.diff(frequencies))
        flux = np.nansum(diffreq * np.array(spectrum[1])[:-1])
        return np.array([spectrum[0], spectrum[1] / flux])


def spects_norm(spectra, frequency=True):
    if not frequency:
        flux = 0
        for spectrum in spectra:
            diffwav = np.abs(np.diff(spectrum[0]))
            flux += np.nansum(diffwav * np.array(spectrum[1])[:-1])
        return np.array([[spectrum[0], spectrum[1] / flux] for spectrum in spectra])
    else:
        c = 1
        flux = 0
        for spectrum in spectra:
            frequencies = c / np.array(spectrum[0])
            diffreq = np.abs(np.diff(frequencies))
            flux += np.nansum(diffreq * np.array(spectrum[1])[:-1])
        return np.array([[spectrum[0], spectrum[1] / flux] for spectrum in spectra])


def relat_diff(spectr1, spectr2, frequency=True):
    spect1 = spectr1
    spect2 = resample(spectr2, spectr1[0])
    difspe = [spect2[0], np.abs(spectr1[1] - spect2[1])]
    spect1 = [spect2[0], np.where(np.isfinite(difspe[1]), spect1[1], np.nan)]
    spect2 = [spect2[0], np.where(np.isfinite(difspe[1]), spect2[1], np.nan)]
    if not frequency:
        diffwav = np.abs(np.diff(difspe[0]))
        fluxdif = np.abs(np.nansum(diffwav * np.array(difspe[1])[:-1]))
        fluxsp1 = np.abs(np.nansum(diffwav * np.array(spect1[1])[:-1]))
        fluxsp2 = np.abs(np.nansum(diffwav * np.array(spect2[1])[:-1]))
        return fluxdif / min(fluxsp1, fluxsp2)
    else:
        c = 1
        frequen = c / np.array(difspe[0])
        diffreq = np.abs(np.diff(frequen))
        fluxdif = np.abs(np.nansum(diffreq * np.array(difspe[1])[:-1]))
        fluxsp1 = np.abs(np.nansum(diffreq * np.array(spect1[1])[:-1]))
        fluxsp2 = np.abs(np.nansum(diffreq * np.array(spect2[1])[:-1]))
        return fluxdif / min(fluxsp1, fluxsp2)


def stack(sp_stack, sources, typ="median", normalise=False):
    wav = sp_stack[0]
    sp_cube = sp_stack[1]
    if len(sources) != len(sp_cube[0]) and typ not in {"median", "mean"}:
        print(
            f"Number of sources and spectra does not match. Switched to median (from {typ}) for stacking!"
        )
        typ = "median"
    if normalise:
        sp_cube = np.array([spect_norm((wav, c))[1] for c in sp_cube.T]).T
    match typ:
        case "mean":
            return (wav, np.mean(sp_cube, axis=1))
        case "sn":
            weights = []
            specn = []
            for i, so in enumerate(sources):
                sn = so["sn50"]
                if sn is not None and sn > 0:
                    weights.append(sn)
                    specn.append(True)
                else:
                    specn.append(False)
            sp_cubn = sp_cube[:, np.array(specn)]
            return (wav, np.average(sp_cubn, axis=1, weights=weights))
        case "ha":
            weights = []
            specn = []
            for i, so in enumerate(sources):
                ha = so["Ha"]
                if ha is not None and ha > 0:
                    weights.append(ha)
                    specn.append(True)
                else:
                    specn.append(False)
            sp_cubn = sp_cube[:, np.array(specn)]
            return (wav, np.average(sp_cubn, axis=1, weights=weights))
        case "median" | _:
            return (wav, np.median(sp_cube, axis=1))


def combine_spectra(spectra):
    refer = spectra[0]
    cube = np.transpose([refer[1]])
    for s in spectra[1:]:
        if (s[0] == refer[0]).all():
            cube = np.hstack((cube, np.transpose([s[1]])))
        else:
            print(f"Spectra not matching in wavelength space!")
    return refer[0], cube
