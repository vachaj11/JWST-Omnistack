import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling.fitting import TRFLSQFitter

import line_man as lm


class Offset(Fittable1DModel):
    yoff = Parameter()

    @staticmethod
    def evaluate(x, yoff):
        return yoff

    @staticmethod
    def fit_deriv(x, yoff):
        d_yoff = np.ones_like(x)
        return [d_yoff]


class Gaussian1D(Fittable1DModel):
    amplitude = Parameter()
    mean = Parameter()
    stddev = Parameter()
    stddev.min = 0.0

    @staticmethod
    def evaluate(x, amplitude, mean, stddev):
        return amplitude * np.exp((-(1 / (2.0 * stddev**2)) * (x - mean) ** 2))

    @staticmethod
    def fit_deriv(x, amplitude, mean, stddev):
        d_amplitude = np.exp((-(1 / (stddev**2)) * (x - mean) ** 2))
        d_mean = (
            2
            * amplitude
            * np.exp((-(1 / (stddev**2)) * (x - mean) ** 2))
            * (x - mean)
            / (stddev**2)
        )
        d_stddev = (
            2
            * amplitude
            * np.exp((-(1 / (stddev**2)) * (x - mean) ** 2))
            * ((x - mean) ** 2)
            / (stddev**3)
        )
        return [d_amplitude, d_mean, d_stddev]

    @property
    def flux(self):
        """In W/m^2"""
        c = 3 * 10**-14
        return self.amplitude * self.stddev * np.sqrt(2 * np.pi) * c / self.mean**2


class Gaussian1Dof(Fittable1DModel):
    amplitude = Parameter()
    mean = Parameter()
    stddev = Parameter()
    yoff = Parameter()
    stddev.min = 0.0

    @staticmethod
    def evaluate(x, amplitude, mean, stddev, yoff):
        return amplitude * np.exp((-(1 / (2.0 * stddev**2)) * (x - mean) ** 2)) + yoff

    @staticmethod
    def fit_deriv(x, amplitude, mean, stddev, yoff):
        d_amplitude = np.exp((-(1 / (stddev**2)) * (x - mean) ** 2))
        d_mean = (
            2
            * amplitude
            * np.exp((-(1 / (stddev**2)) * (x - mean) ** 2))
            * (x - mean)
            / (stddev**2)
        )
        d_stddev = (
            2
            * amplitude
            * np.exp((-(1 / (stddev**2)) * (x - mean) ** 2))
            * ((x - mean) ** 2)
            / (stddev**3)
        )
        d_yoff = np.ones_like(x)
        return [d_amplitude, d_mean, d_stddev, d_yoff]

    @property
    def flux(self):
        """In W/m^2"""
        c = 3 * 10**-14
        return self.amplitude * self.stddev * np.sqrt(2 * np.pi) * c / self.mean**2


def cut_range(spectrum, rang):
    spectrum = np.array(spectrum)
    wav, flux = spectrum
    imin = np.argwhere(wav > rang[0]).T[0]
    imax = np.argwhere(wav < rang[1]).T[0]
    if imin.size + imax.size == 0:
        return None
    elif imin.size == 0:
        print("Spectrum does not cover all of line.")
        return spectrum[:, : imax.max() + 1]
    elif imax.size == 0:
        print("Spectrum does not cover all of line.")
        return spectrum[:, imin.min() :]
    else:
        return spectrum[:, imin.min() : imax.max() + 1]


def get_closest(spectrum, line):
    wav, flux = spectrum
    ind = np.argmin(np.abs(wav - line))
    return flux[ind]


def line_range(lines, grat="", dwidth=8):
    if type(grat) is not str:
        R = grat
    else:
        match grat:
            case "prism":
                R = 100
            case x if x[-1:] == "h":
                R = 2700
            case x if x[-1:] == "m":
                R = 1000
            case _:
                R = len(spectrum[0])
    delta = max(lines) / R / 2.35 * dwidth
    rang = [min(lines) - delta, max(lines) + delta]
    return rang, R


def fit_lines(spectrum, lines, delta=None, grat="", dwidth=8, manual=False, mline=None):
    rang, R = line_range(lines, grat=grat, dwidth=dwidth)
    spect = cut_range(spectrum, rang)
    yav = np.nanmedian(spect[1])
    models = []
    for line in lines:
        std = line / R / 2.35
        amplitude = abs(get_closest(spectrum, line) - yav)
        if amplitude is not None:
            models.append(Gaussian1D(mean=line, stddev=std, amplitude=amplitude))
            dev = 1.4 if grat == R else 2
            coe = 1.5 if grat == R else 4
            models[-1].stddev.max = std * dev
            models[-1].stddev.min = std / dev
            models[-1].mean.min = max(rang[0], line - std * coe)
            models[-1].mean.max = min(rang[1], line + std * coe)
            models[-1].amplitude.min = 0
    mline = mline if mline is not None else lines[0]
    if models:
        msum = Offset(yoff=yav)
        for m in models:
            msum += m
        fitter = TRFLSQFitter()
        fit = fitter(msum, spect[0], spect[1])
        x = np.linspace(rang[0], rang[1], 200)
        if not manual:
            return fit, x, spect
        else:
            return lm.manfit(fit, x, spect, mline, grat=grat, info=False)
    else:
        return None


def fit_infos(fit):
    lines = []
    for g in fit._leaflist[1:]:
        line = dict()
        line["mean"] = g.mean.value
        line["amp"] = g.amplitude.value
        line["stddev"] = g.stddev.value
        line["flux"] = g.flux
        lines.append(line)
    return lines


def flux_at(fit, line, grat="g140m", std=1):
    _, R = line_range([line], grat=grat)
    lstd = line / R / 2.35
    lines = fit_infos(fit)
    overlap = dict()
    for l in lines:
        if (
            l["mean"] - std * l["stddev"] < line + lstd * std
            and l["mean"] + std * l["stddev"] > line - lstd * std
        ):
            overlap[l["mean"]] = l["flux"]
    if overlap:
        return min(overlap.items(), key=lambda x: abs(x[0] - line))[1]
    else:
        return 0.0


def flux_extract(fit, mline, grat="g140m", red=lambda f, l: f):
    fluxes = dict()
    outd = True
    if type(mline) is not dict:
        mline = {"_": mline}
        outd = False
    for l, v in mline.items():
        if hasattr(v, "__iter__"):
            fluxes[l] = sum([red(flux_at(fit, i), i) for i in v])
        else:
            fluxes[l] = red(flux_at(fit, v), v)
    if outd:
        return fluxes
    else:
        return next(f for f in fluxes.values())


def flux_nan(mline):
    outd = True
    if type(mline) is not dict:
        mline = {"_": mline}
        outd = False
    fluxes = {l: np.nan for l in mline}
    if outd:
        return fluxes
    else:
        return next(f for f in fluxes.values())
