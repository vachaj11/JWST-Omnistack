import random
import sys

import matplotlib as mpl
from astropy.modeling.fitting import TRFLSQFitter
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

import plots

colors = ["b", "g", "r", "c", "m", "y"]


class App(QWidget):

    def __init__(self, fit, spectrum, mline, parent=None):
        super(App, self).__init__(parent)

        self.resize(700, 700)
        self.title = "The Window Title"
        self.setWindowTitle(self.title)
        self.canv = PlotCanvas(fit, spectrum, mline, parent=self)

        layout = QVBoxLayout()
        layout.addWidget(self.canv)
        self.setLayout(layout)

    def keyPressEvent(self, event):
        self.canv.update_key(event)


class PlotCanvas(FigureCanvas):
    def __init__(self, fit, spectrum, mline, parent=None, dpi=100):
        super(PlotCanvas, self).__init__(Figure())
        self.setParent(parent)
        self.fit = fit
        self.no = (len(self.fit.param_names) - 1) // 3
        self.mline = mline
        self.spectrum = spectrum
        fig = Figure(dpi=dpi)
        self.figure = fig
        self.canvas = FigureCanvas(self.figure)
        self.axes = fig.add_subplot()
        self.sel = 1
        self.plot_lines()
        self.draw()

    def plot_lines(self):
        plots.plot_fit(self.spectrum, self.fit, axis=self.axes, text=True, plot=False)
        self.axes.axvline(x=self.mline, lw=1.5, c="gold", ls=":")
        names = self.fit.param_names
        val = {nam: getattr(self.fit, nam).value for nam in names}
        lws = [0.7] * (self.no + 1)
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
        self.axes.cla()
        self.plot_lines()
        self.draw()

    def update_key(self, key):
        if key.text().isdigit():
            self.sel = int(key.text())
        elif key.key() == 16777236:
            if self.no >= self.sel > 0:
                mea = getattr(self.fit, f"mean_{self.sel}")
                std = getattr(self.fit, f"stddev_{self.sel}")
                setattr(self.fit, f"mean_{self.sel}", mea + std / 20)
        elif key.key() == 16777237:
            if self.no >= self.sel > 0:
                amp = getattr(self.fit, f"amplitude_{self.sel}")
                setattr(self.fit, f"amplitude_{self.sel}", amp * 0.97)
            elif self.sel == 0:
                yof = getattr(self.fit, "yoff_0")
                setattr(self.fit, "yoff_0", yof * 0.97)
        elif key.key() == 16777234:
            if self.no >= self.sel > 0:
                mea = getattr(self.fit, f"mean_{self.sel}")
                std = getattr(self.fit, f"stddev_{self.sel}")
                setattr(self.fit, f"mean_{self.sel}", mea - std / 20)
        elif key.key() == 16777235:
            if self.no >= self.sel > 0:
                amp = getattr(self.fit, f"amplitude_{self.sel}")
                setattr(self.fit, f"amplitude_{self.sel}", amp * 1.03)
            elif self.sel == 0:
                yof = getattr(self.fit, "yoff_0")
                setattr(self.fit, "yoff_0", yof * 1.03)
        elif key.key() == 66:
            if self.no >= self.sel > 0:
                std = getattr(self.fit, f"stddev_{self.sel}")
                setattr(self.fit, f"stddev_{self.sel}", std * 1.03)
        elif key.key() == 78:
            if self.no >= self.sel > 0:
                std = getattr(self.fit, f"stddev_{self.sel}")
                setattr(self.fit, f"stddev_{self.sel}", std * 0.97)
        elif key.key() == 16777220:
            QApplication.instance().quit()
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
        self.update_plot()


@mpl.rc_context(
    {
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.size": 12,
    }
)
def manfit(fit, x, spectrum, mline, info=True):
    if info:
        print(
            "=====\nUse numerical keys to switch between components of the fit\nUse up/down arrow to adjust amplitudes/y-offset\nUse left/right arrows to adjust mean position\nUse B/N keys to adjust distribution width\nUse S key to switch amplitude/y-offset sign\nUse F key to repeat the fitting process\n====="
        )
    if QApplication.instance():
        app = QApplication.instance()
        app.shutdown()
    app = QApplication()
    widget = App(fit, spectrum, mline)
    widget.show()
    app.exec()
    app.shutdown()
    return fit, x, spectrum
