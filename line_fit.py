import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling.models import Fittable1DModel, Parameter
from astropy.modeling.fitting import TRFLSQFitter

@custom_model
class Gaussian1Dof(Fittable1DModel):
    amplitude = Parameter()
    mean = Parameter()
    sigma = Parameter()
    yoff = Parameter()
    @staticmethod
    def evaluate(x, amplitude, mean, stddev, yoff):
        return amplitude * np.exp((-(1 / (2. * stddev**2)) * (x - mean)**2))+yoff

    @staticmethod
    def fit_deriv(x, amplitude, mean, stddev, yoff):
        d_amplitude = np.exp((-(1 / (stddev**2)) * (x - mean)**2))
        d_mean = (2 * amplitude *
                  np.exp((-(1 / (stddev**2)) * (x - mean)**2)) *
                  (x - mean) / (stddev**2))
        d_stddev = (2 * amplitude *
                    np.exp((-(1 / (stddev**2)) * (x - mean)**2)) *
                    ((x - mean)**2) / (stddev**3))
        d_yoff = 1
        return [d_amplitude, d_mean, d_stddev, d_yoff]
    
def cut_range(spectrum, rang):
    spectrum=np.array(spectrum)
    wav, flux = spectrum
    imin = np.argwhere(wav>rang[0]).T[0]
    imax = np.argwhere(waw<rang[1]).T[0]
    if imin.size+imax.size==0:
        return None
    elif imin.size == 0:
        print('Spectrum does not cover all of line.')
        return spectrum[:,:imax.max()+1]
    elif imax.size==0:
        print('Spectrum does not cover all of line.')
        return spectrum[:,imin.min():]
    else:
        return spectrum[:,imin.min():imax.max()+1]

def get_closest(spectrum, line):
    wav, flux = spectrum
    ind = np.argmin(np.abs(wav-line))
    return flux[ind]
    
        
def fit_line(source, spectrum, line, delta):
    match source["grat"]:
        case "prism":
            R = 100
        case x if x[-1:] == "h":
            R = 2700
        case x if x[-1:] == "m":
            R = 1000
        case _:
            R = len(spectrum[0])
    std = np.sqrt(min(spectrum[0]) * max(spectrum[0])) / R / 2.35
    amplitude = get_closest(spectrum, line)
    spect = cut_range(spectrum, [line-delta,line+delta])
    if amplitude is not None:
        gauss = Gaussian1Dof(mean = line, sigma=std, amplitude=amplitude, yoff=0.)
        fit = TRFLSQFitter()
        m = fit(gauss, spect[0], spect[1])
        return m.mean, m.sigma, m.amplitude, myoff
    else:
        return None
