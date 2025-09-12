"""Holds classes and methods specifying a minimalist gui for manual evaluation and adjustments of spectral line fitting.

Attributes:
    colors (list): Specifies order of colors elements of gui are to be drawn with.
    fl (function): Small function to recursively flatten whatever iterable provided into a 1D list
"""

import importlib
import random
import sys

import matplotlib as mpl
import numpy as np
from astropy.modeling.fitting import TRFLSQFitter
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QEventLoop, Qt
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

import line_fit as lf
import plots

colors = ["b", "g", "r", "c", "m", "y"]

fl = lambda l: sum(map(fl, list(l)), []) if hasattr(l, "__iter__") else [l]


class App(QWidget):
    """A PySide6 QWidget object which specifies meta-window of the gui and its contents."""

    def __init__(self, fit, spectrum, mline, grat="g140m", parent=None):
        """Draws the meta-window and its contents on initialisation."""
        super(App, self).__init__(parent)

        self.resize(700, 700)
        self.title = f"Fitting {mline}"
        self.setWindowTitle(self.title)
        self.canv = PlotCanvas(fit, spectrum, mline, grat=grat, parent=self)

        layout = QVBoxLayout()
        layout.addWidget(self.canv)
        self.setLayout(layout)

    def keyPressEvent(self, event):
        """Registers keystrokes with the window open and sends signal to corresponding function to interpret them."""
        self.canv.update_key(event)


class PlotCanvas(FigureCanvas):
    """A PySide6 FigureCanvas object which draws the spectral visualisation within the gui and allows for interaction with it."""

    def __init__(self, fit, spectrum, mline, grat="g140m", parent=None, dpi=100):
        """On initialisation assigns all passed arguments to appropriate objects and draws the Matplotlib plot making up core of the window."""
        super(PlotCanvas, self).__init__(Figure())
        self.setParent(parent)
        self.fit = fit
        self.no = (len(self.fit.param_names) - 1) // 3
        self.mline = fl(mline.values()) if type(mline) is dict else mline
        self.spectrum = spectrum
        fig = Figure(dpi=dpi)
        self.figure = fig
        self.canvas = FigureCanvas(self.figure)
        self.axes = fig.add_subplot()
        self.sel = 1
        self.grat = grat
        _, R = lf.line_range([np.median(spectrum[0])], grat=grat)
        self.std = np.sqrt(spectrum[0][0] * spectrum[0][-1]) / R / 2.35
        self.plot_lines()
        self.draw()

    def plot_lines(self):
        """When called redraws the visualisation of the desired spectral region with overlayed fitted model and its properties."""
        plots.plot_fit(self.spectrum, self.fit, axis=self.axes, text=True, plot=False)
        for ml in np.array(self.mline).flatten():
            lw = 2.5 if lf.flux_at(self.fit, ml, grat=self.grat) else 1
            self.axes.axvline(x=ml, lw=lw, c="gold", ls=":")
            self.axes.axvspan(
                xmin=ml - self.std, xmax=ml + self.std, fc="gold", ls="", alpha=0.1
            )
        names = self.fit.param_names
        val = {nam: getattr(self.fit, nam).value for nam in names}
        lws = [1] * (self.no + 1)
        if self.sel < len(lws):
            lws[self.sel] = 2.5
        yof = val["yoff_0"]
        alp = 0.6
        self.axes.axhline(y=yof, lw=lws[0], c="gray", alpha=alp, ls="--")
        for i in range(1, self.no + 1):
            amp = val[f"amplitude_{i}"]
            mea = val[f"mean_{i}"]
            std = val[f"stddev_{i}"]
            c = colors[i - 1 % len(colors)]
            self.axes.plot(
                [mea - std, mea - std],
                [yof, yof + amp],
                lw=lws[i],
                alpha=alp,
                ls="--",
                c=c,
            )
            self.axes.plot(
                [mea - std, mea + std],
                [yof + amp, yof + amp],
                lw=lws[i],
                alpha=alp,
                ls="--",
                c=c,
            )
            self.axes.plot(
                [mea + std, mea + std],
                [yof + amp, yof],
                lw=lws[i],
                alpha=alp,
                ls="--",
                c=c,
            )
            self.axes.plot(
                [mea + std, mea - std], [yof, yof], lw=lws[i], alpha=alp, ls="--", c=c
            )

    def update_plot(self):
        """When called clears the existing matplotlib axis and redraws it anew."""
        self.axes.cla()
        self.plot_lines()
        self.draw()

    def update_key(self, key):
        """Upon passed keystroke modifies corresponding aspects of the gui or the fitted model and subsequently redraws the central matplotlib plot."""
        mult = 10 if (key.modifiers()._name_ == "ShiftModifier") else 1
        if key.text().isdigit():
            self.sel = int(key.text())
        elif key.key() == 16777236:
            if self.no >= self.sel > 0:
                mea = getattr(self.fit, f"mean_{self.sel}")
                std = getattr(self.fit, f"stddev_{self.sel}")
                setattr(self.fit, f"mean_{self.sel}", mea + std / 20 * mult)
        elif key.key() == 16777237:
            if self.no >= self.sel > 0:
                amp = getattr(self.fit, f"amplitude_{self.sel}")
                setattr(self.fit, f"amplitude_{self.sel}", amp * (1 - 0.02 * mult))
            elif self.sel == 0:
                yof = getattr(self.fit, "yoff_0")
                setattr(self.fit, "yoff_0", yof * (1 - 0.01 * mult))
        elif key.key() == 16777234:
            if self.no >= self.sel > 0:
                mea = getattr(self.fit, f"mean_{self.sel}")
                std = getattr(self.fit, f"stddev_{self.sel}")
                setattr(self.fit, f"mean_{self.sel}", mea - std / 20 * mult)
        elif key.key() == 16777235:
            if self.no >= self.sel > 0:
                amp = getattr(self.fit, f"amplitude_{self.sel}")
                setattr(self.fit, f"amplitude_{self.sel}", amp * (1 + 0.02 * mult))
            elif self.sel == 0:
                yof = getattr(self.fit, "yoff_0")
                setattr(self.fit, "yoff_0", yof * (1 + 0.01 * mult))
        elif key.key() == 66:
            if self.no >= self.sel > 0:
                std = getattr(self.fit, f"stddev_{self.sel}")
                setattr(self.fit, f"stddev_{self.sel}", std * (1 + 0.03 * mult))
        elif key.key() == 78:
            if self.no >= self.sel > 0:
                std = getattr(self.fit, f"stddev_{self.sel}")
                setattr(self.fit, f"stddev_{self.sel}", std * (1 - 0.03 * mult))
        elif key.key() == 83:
            if self.no >= self.sel > 0:
                amp = getattr(self.fit, f"amplitude_{self.sel}")
                setattr(self.fit, f"amplitude_{self.sel}", -amp)
            elif self.sel == 0:
                yof = getattr(self.fit, "yoff_0")
                setattr(self.fit, "yoff_0", -yof)
        elif key.key() == 70:
            fitter = TRFLSQFitter()
            self.fit = fitter(
                self.fit, self.spectrum[0], self.spectrum[1], inplace=True
            )
        elif key.key() == 16777220:
            self.window().close()
            # QApplication.instance().quit()
        self.update_plot()


@mpl.rc_context(
    {
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.size": 12,
    }
)
def manfit(fit, x, spectrum, mline, grat="g140m", info=True):
    """Based on passed AstroPy fittable model and spectral data creates and shows a gui window for manual evaluation of the model."""
    if info:
        print(
            "=====\nUse numerical keys to switch between components of the fit\nUse up/down arrow to adjust amplitudes/y-offset\nUse left/right arrows to adjust mean position\nUse B/N keys to adjust distribution width\nUse S key to switch amplitude/y-offset sign\nUse F key to repeat the fitting process\n====="
        )
    if QApplication.instance():
        app = QApplication.instance()
    else:
        app = QApplication()
    app.setQuitOnLastWindowClosed(False)
    loop = QEventLoop()
    widget = App(fit, spectrum, mline, grat=grat)
    widget.setAttribute(Qt.WA_DeleteOnClose)
    widget.destroyed.connect(loop.quit)
    widget.show()
    loop.exec()
    app.setQuitOnLastWindowClosed(True)
    return fit, x, spectrum
