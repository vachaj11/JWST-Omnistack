import os
import time

import astropy.io.fits as fits
import numpy as np
from astropy.convolution import Gaussian1DKernel, convolve

glob = dict()


def get_spectrum_fits(
    source, base="https://s3.amazonaws.com/msaexp-nirspec/extractions/"
):
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


def get_spectrum(source, base="../Data/Npy/"):
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
    glob[path] = np.array(spectrum)


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


def iden_bad(spectrum, nstd=20, nref=3, nei=3):
    """Identify and remove one-pixel erroneous noise peaks from spectra"""
    if spectrum is None:
        return None, None, None
    flux0 = np.copy(spectrum[1])
    mean = np.nanmean(flux0)
    std = np.nanstd(flux0)
    mask0 = ((mean + std * nstd) < flux0) | (flux0 < (mean - std * nstd))
    indb0 = np.where(mask0)[0]
    indb1 = []
    for i in indb0:
        fluxr = flux0[max(i - 50, 0) : i + 50 + 1]
        fluxr[50] = np.nan
        meanr = np.nanmean(fluxr)
        stdr = np.nanstd(fluxr)
        maskr = ((meanr + stdr * nref) < flux0) | (flux0 < (meanr - stdr * nref))
        indbr = np.where(maskr)[0]
        if (
            i in indbr
            and np.nanmin(np.where(np.abs(indbr - i) == 0, np.nan, np.abs(indbr - i)))
            > nei
        ):
            indb1.append(i)
    mask1 = np.zeros(mask0.shape)
    mask1[indb1] = True
    flux1 = np.where(mask1, np.nan, flux0)
    return indb1, [spectrum[0][i] for i in indb1], [spectrum[0], flux1]


def iden_bad_e(spectrum, nstd=5, ncou=10):
    """Identify and remove noise-dominated edges of spectra"""
    if spectrum is None:
        return None, None, None
    flux0 = np.copy(spectrum[1])
    nnan = np.where(np.isfinite(flux0))[0]
    mean = np.nanmean(flux0[nnan.min() + ncou : nnan.max() + 1 - ncou])
    std = np.nanstd(flux0[nnan.min() + ncou : nnan.max() + 1 - ncou])

    indb = []
    fluxs = flux0[nnan.min() : nnan.min() + ncou]
    masks = ((mean + std * nstd) < fluxs) | (fluxs < (mean - std * nstd))
    if masks.any():
        flux0[nnan.min() : nnan.min() + ncou] = np.nan
        indb += list(np.where(masks)[0] + nnan.min())
    fluxe = flux0[nnan.max() + 1 - ncou : nnan.max() + 1]
    maske = ((mean + std * nstd) < fluxe) | (fluxe < (mean - std * nstd))
    if maske.any():
        flux0[nnan.max() + 1 - ncou : nnan.max() + 1] = np.nan
        indb += list(np.where(maske)[0] + nnan.max() + 1 - ncou)
    return indb, [spectrum[0][i] for i in indb], [spectrum[0], flux0]


def remove_nan(spectrum):
    wav, flux = spectrum
    wavn = []
    fluxn = []
    for w, f in zip(wav, flux):
        if np.isfinite(f):
            wavn.append(w)
            fluxn.append(f)
    return np.array([np.array(wavn), np.array(fluxn)])


def resampled_spectra(sources, rang, reso, prin=False, **kwargs):
    spectra = []
    sourn = []
    space = wave_sp(rang[0], rang[1], reso)
    for i, s in enumerate(sources):
        try:
            t0 = time.time()
            sp = get_spectrum(s, **kwargs)
            t1 = time.time()
            if sp is not None:
                sr = resample(sp, space, s["z"])
                t2 = time.time()
                spectra.append(sr)
                sourn.append(s)
                if prin:
                    print(
                        f'\r\033[KResampled: {s["srcid"]:<14}\t({i} of {len(sources)})\tTime: {(t2-t1)/(t1-t0):.2f}',
                        end="",
                    )
            else:
                if prin:
                    print(
                        f'\r\033[KNo spectral file found for: {str(s["srcid"]):<14}\t({i} of {len(sources)})',
                        end="",
                    )
        except:
            if prin:
                print(
                    f'\r\033[KSomething strange with: {str(s["srcid"]):<14}\t({i} of {len(sources)})',
                    end="",
                )
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


def degrade_spectrum(spectrum, tpx=0.0005, fact=2.2):
    opx = spectrum[0][1] - spectrum[0][0]
    tstd = tpx * fact / (2 * np.sqrt(2 * np.log(2)))
    ostd = opx * fact / (2 * np.sqrt(2 * np.log(2)))
    kstd = np.sqrt(tstd**2 - ostd**2)
    ker = Gaussian1DKernel(kstd / opx)
    convolved = convolve(spectrum[1], ker, boundary="extend")
    convolved = np.where(np.isfinite(spectrum[1]), convolved, np.nan)
    return [spectrum[0], convolved]


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
            weights = np.array(weights) / np.sum(weights)
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
    Tspectra = []
    for s in spectra:
        if (s[0] == refer[0]).all():
            Tspectra.append(np.transpose([s[1]]))
        else:
            print(f"Spectra not matching in wavelength space!")
    cube = np.hstack(Tspectra)
    return refer[0], cube
