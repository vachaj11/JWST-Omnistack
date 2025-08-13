import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling.fitting import TRFLSQFitter


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


def line_range(lines, grat="", dwidth = 8):
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


def fit_lines(spectrum, lines, delta=None, grat="", dwidth = 8):
    rang, R = line_range(lines, grat=grat, dwidth = dwidth)
    spect = cut_range(spectrum, rang)
    models = []
    for line in lines:
        std = line / R / 2.35
        amplitude = get_closest(spectrum, line)
        if amplitude is not None:
            models.append(Gaussian1D(mean=line, stddev=std, amplitude=amplitude))
    if models:
        msum = Offset(yoff=0.0)
        for m in models:
            msum += m
        fitter = TRFLSQFitter()
        fit = fitter(msum, spect[0], spect[1])
        x = np.linspace(rang[0], rang[1], 200)
        return fit, x, spect
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


def flux_at(fit, line, std=1):
    lines = fit_infos(fit)
    flux = np.float64(0.0)
    for l in lines:
        if l["mean"] - std * l["stddev"] < line < l["mean"] + std * l["stddev"]:
            flux += l["flux"]
    return flux
