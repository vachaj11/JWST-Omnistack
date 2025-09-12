"""Holds methods for (down)loading, saving, modifying and otherwise manipulating spectral data.

Attributes:
    glob (dict): Global dictionary holding data of sources which have been already loaded, so that their data can be accessed more quickly.
    flatten (function): Small function to recursively flatten whatever iterable provided into a 1D list
"""

import os
import time

import astropy.io.fits as fits
import numpy as np
from astropy.convolution import Gaussian1DKernel, convolve
from astropy.utils.data import clear_download_cache
from scipy.ndimage import gaussian_filter1d

glob = dict()

flatten = lambda l: sum(map(flatten, list(l)), []) if hasattr(l, "__iter__") else [l]


def get_spectrum_fits(
    source, base="https://s3.amazonaws.com/msaexp-nirspec/extractions/"
):
    """Load or download a spectral fits file and extract 1D spectra from it."""
    path = base + source["root"] + "/" + source["file"]
    try:
        x = fits.open(path)
        wave = x[1].data["wave"]
        flux = x[1].data["flux"]
        # wave, flux = clippint([wave, flux])
        x.close()
        if "https://" in path:
            clear_download_cache(hashorurl=path, pkgname="astropy")
        return [wave, flux]
    except:
        return None


def get_spectrum(source, base="../Data/Npy_v4/"):
    """Load or download a spectra data for the provided catalog entry from the provided directory."""
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


def save_npy(source, spectrum, base="../Data/Npy_v4/"):
    """Save spectra data for the provided catalog entry and the provided directory."""
    path = base + source["root"] + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    path = path + source["file"][:-5] + ".npy"
    np.save(path, np.array(spectrum))
    glob[path] = np.array(spectrum)


def rm_npy(source, base="../Data/Fits/"):
    """Delete spectral file for the provided catalog entry and the provided directory."""
    path = base + source["root"] + "/"
    path = path + source["file"][:-5] + ".npy"
    try:
        os.remove(path)
    except:
        print(f"File {path} not found and not removed.")


def clipping(spectrum, sup=30, sdo=15, ite=1):
    """Iterativelly clip outlying values of the spectra, replacing them by nan values."""
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


def iden_bad_e_legacy(spectrum, nstd=5, ncou=10):
    """Identify and remove noise-dominated edges of spectra.

    Legacy method based on mean and std of all non-edge spectra data.
    """
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


def iden_bad_e(spectrum, nstd=5, ncou=10):
    """Identify and remove noise-dominated edges of spectra.

    Updated method based on mean and std of median 50 % of spectra data.
    """
    if spectrum is None:
        return None, None, None
    flux0 = np.copy(spectrum[1])
    nnan = np.where(np.isfinite(flux0))[0]
    nn50 = nnan[(np.percentile(nnan, 25) < nnan) & (nnan < np.percentile(nnan, 75))]
    ce50 = flux0[nn50]
    mean = np.nanmean(ce50)
    std = np.nanstd(ce50)

    indb = []
    fluxs = flux0[nnan.min() : nnan.min() + ncou]
    masks = ((mean + std * nstd) < fluxs) | (fluxs < (mean - std * nstd))
    if sum(masks) > ncou / 2 and np.mean(fluxs) - np.std(fluxs) < mean < np.mean(
        fluxs
    ) + np.std(fluxs):
        flux0[nnan.min() : nnan.min() + ncou] = np.nan
        indb += list(np.where(masks)[0] + nnan.min())
    fluxe = flux0[nnan.max() + 1 - ncou : nnan.max() + 1]
    maske = ((mean + std * nstd) < fluxe) | (fluxe < (mean - std * nstd))
    if sum(maske) > ncou / 2:
        flux0[nnan.max() + 1 - ncou : nnan.max() + 1] = np.nan
        indb += list(np.where(maske)[0] + nnan.max() + 1 - ncou)
    return indb, [spectrum[0][i] for i in indb], [spectrum[0], flux0]


def remove_nan(spectrum):
    """Return spectra data with all nan-values removed."""
    wav, flux = spectrum
    wavn = []
    fluxn = []
    for w, f in zip(wav, flux):
        if np.isfinite(f):
            wavn.append(w)
            fluxn.append(f)
    return np.array([np.array(wavn), np.array(fluxn)])


def resampled_spectra(sources, rang, reso, prin=False, degrade=850, z=None, **kwargs):
    """Resample spectra into a given rest-frame-wavelenth range and pixel resolution. And if requested also degrade the spectra into a given line-spread-function resolution."""
    spectra = []
    sourn = []
    lws = []
    space = wave_sp(rang[0], rang[1], reso)
    T0 = time.time()
    for i, s in enumerate(sources):
        try:
            t0 = time.time()
            sp = get_spectrum(s, **kwargs)
            t1 = time.time()
            if sp is not None:
                zv = s["z"] if z is None else z
                sr = resample(sp, space, zv)
                t2 = time.time()
                spectra.append(sr)
                sourn.append(s)
                if degrade:
                    lw = (
                        pixel_at(sp, np.mean(rang) * (1 + s["z"]), source=s)
                        * 2.2
                        / (1 + s["z"])
                    )
                    lws.append(lw)
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
    T1 = time.time()
    if degrade and lws:
        if type(degrade) is bool:
            lwm = np.max(lws)
        else:
            lwm = np.mean(rang) / degrade
        for i, lw in enumerate(lws):
            spectra[i] = degrade_spectrum(
                spectra[i], tpx=lwm, opx=lw, fact=1, use_astropy=False
            )
    T2 = time.time()
    if prin and degrade:
        print(f"Relative time spent degrading spectra: {(T2-T1)/(T1-T0):.2f}")
    return spectra, sourn


def join_source(sources):
    """Joins spectra of multiple sources into a single array, resolving overlaps with median stacking."""
    rgs = flatten([s["range"] for s in sources])
    ran = [min(rgs), max(rgs)]
    res = int((ran[1] - ran[0]) / (ran[0] / 3000))
    sp, sn = resampled_spectra(sources, ran, reso=res, z=0.0)
    com = combine_spectra(sp)
    stacc = stack(com, sn, typ="median")
    ms = max(sources, key=lambda x: x["sn50"])
    ms["sn50"] = min([s["sn50"] for s in sources])
    return ms, stacc


def wave_sp(minw, maxw, points):
    """Create wavelength grid of given range and number of points"""
    return np.linspace(minw, maxw, points)


def useful_range(spectrum):
    """Identify wavelength ranges at which given spectra has continuous coverage."""
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
    """Cut given spectra into sections where it has continuous finite coverage."""
    x = np.array(spectrum)
    spectra = np.split(x, np.argwhere(~np.isfinite(x[1])).T[0], axis=1)
    spectran = [s[:, 1:] for s in spectra if s[:, 1:].size > 0]
    return spectran


def pixel_at(spectrum, wav, source=None):
    """Find pixel size of the spectra at a given wavelength."""
    mpx = {
        "g140h": 0.00060000,
        "g235h": 0.00100952,
        "g395h": 0.00170476,
    }
    wavr = np.array(spectrum[0])
    mind = np.argmin(np.abs(wavr - wav))
    if (
        source is not None
        and source.get("grat_orig") in mpx.keys()
        and source["grat"][-1] == "m"
    ):
        return mpx[source["grat_orig"]]
    if mind + 1 < wavr.size:
        return wavr[mind + 1] - wavr[mind]
    else:
        print("No spectral coverage at requested wavelength.")
        return wavr[1] - wavr[0]


def resample_legacy(spectrum, space, z=0.0, normalise=False):
    """Resample provided spectra into given wavelength space, linearly interpolating any intermediate values.

    Legacy method which is comparatively quite slow.
    """
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
    """Resample provided spectra into given wavelength space, linearly interpolating any intermediate values.

    Updated fast method which uses build-in numpy function with similar purpose.
    """
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
    """Normalise given spectra to have total flux == 1 in wavelength or frequency space."""
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
    """Jointly normalise given list of spectra to have total summed flux == 1 in wavelength or frequency space."""
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


def degrade_spectrum(spectrum, tpx=0.0005, opx=None, fact=2.2, use_astropy=True):
    """Degrade a spectrum to target pixel resolution by convolving it with appropriate Gaussian."""
    opxo = spectrum[0][1] - spectrum[0][0]
    opx = opxo if opx is not None else opx
    if tpx < opx:
        return spectrum
    tstd = tpx * fact / (2 * np.sqrt(2 * np.log(2)))
    ostd = opx * fact / (2 * np.sqrt(2 * np.log(2)))
    kstd = np.sqrt(tstd**2 - ostd**2)
    if use_astropy:
        ker = Gaussian1DKernel(kstd / opxo)
        convolved = convolve(spectrum[1], ker, boundary="extend")
    else:
        convolved = gaussian_filter1d(spectrum[1], sigma=kstd / opxo, mode="nearest")
    convolved = np.where(np.isfinite(spectrum[1]), convolved, np.nan)
    return [spectrum[0], convolved]


def relat_diff(spectr1, spectr2, frequency=True):
    """Calculate relative total flux sizes of two given spectra across regions where they overlap."""
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
    """Stack provided cube of spectra via the specified method and thus obtain a joint 1D spectra"""
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
            return (wav, np.nanmean(sp_cube, axis=1))
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
            sp_cubn = np.ma.MaskedArray(sp_cubn, mask=np.isnan(sp_cubn))
            weights = np.array(weights) / np.sum(weights)
            return (wav, np.ma.average(sp_cubn, axis=1, weights=weights))
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
            sp_cubn = np.ma.MaskedArray(sp_cubn, mask=np.isnan(sp_cubn))
            return (wav, np.average(sp_cubn, axis=1, weights=weights))
        case "median" | _:
            return (wav, np.nanmedian(sp_cube, axis=1))


def combine_spectra(spectra):
    """Combine spectra assumed to have the same wavelength grid into a joint cube."""
    refer = spectra[0]
    Tspectra = []
    for s in spectra:
        if (s[0] == refer[0]).all():
            Tspectra.append(np.transpose([s[1]]))
        else:
            print(f"Spectra not matching in wavelength space!")
    cube = np.hstack(Tspectra)
    return refer[0], cube
