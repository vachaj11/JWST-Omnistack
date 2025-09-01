import matplotlib as mpl

mpl.use("qtagg")

import matplotlib.pyplot as plt
import numpy as np

import abundc as ac
import catalog

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Computer Modern",
        "font.size": 15,
    }
)


flatten = lambda l: sum(map(flatten, list(l)), []) if hasattr(l, "__iter__") else [l]


def flux_conv(sources, lines, lind, save=None, axis=None, typ="median"):
    if axis is None:
        fig, axs = plt.subplots()
    else:
        fig = plt.gcf()
        axs = axis
    mx = len(sources)
    n = np.linspace(mx / 20, mx, 200)
    y = []
    x = []
    for i in n:
        sample = np.random.choice(sources, size=int(i), replace=False)
        flux = ac.fit_lines(sample, lines, lines[lind], plot=False, typ=typ)
        y.append(flux)
        x.append(i)
    x, y = zip(*sorted(zip(x, y)))
    axs.plot(x, y)
    axs.set_xlabel("Number of spectra stacked")
    axs.set_ylabel("Flux")
    axs.set_title("Convergence of line flux with stack size")
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)
    else:
        plt.show()
    plt.close(fig)


def abundance_in_z(
    sources,
    zrangs,
    val="z",
    val_name=None,
    abund=ac.Oxygen,
    save=None,
    title=None,
    yax=None,
    indiv=True,
    indso=None,
    manual=False,
    **kwargs,
):
    n = int(-(-np.sqrt(len(abund)) // 1))
    indso = sources if indso is None else indso
    fig = plt.figure()
    gs = fig.add_gridspec(n, n, hspace=0, wspace=0)
    axes = gs.subplots(sharex="col", sharey="row")
    yrang = []
    val_name = val_name if val_name is not None else val
    if n == 1:
        axs = [axes]
    else:
        axs = axes.flatten()
    valss = []
    for i, (nam, ab) in enumerate(abund.items()):
        for zrang in zrangs:
            sourz = [
                s
                for s in sources
                if s.get(val) is not None and zrang[0] < s[val] < zrang[1]
            ]
            indrz = [
                s
                for s in indso
                if s.get(val) is not None and zrang[0] < s[val] < zrang[1]
            ]
            cal_red = ac.red_const(indrz) if indso is None else None
            if indiv:
                ind, _ = ac.indiv_stat(ab, indrz, cal_red=cal_red, val=val, **kwargs)
                x = sum(list(ind[0]), [])
                y = sum(list(ind[1]), [])
                if x:
                    axs[i].plot(
                        x,
                        y,
                        ls="",
                        marker=".",
                        c="gray",
                        alpha=0.15,
                        markersize=2,
                    )
            v, m, st = ac.boots_stat(ab, sourz, cal_red=None, manual=manual, **kwargs)
            if v.size:
                zmean = [(zrang[1] + zrang[0]) / 2] * len(v)
                zerr = [(zrang[1] - zrang[0]) / 2] * len(v)
                axs[i].plot(zmean, v, ls="", marker="D", c="black")
                axs[i].errorbar(zmean, v, xerr=zerr, ls="", c="black", capsize=5)
                axs[i].errorbar(zmean, m, yerr=st, ls="", c="black", capsize=5)
                yrang += [np.nanmin(m - st[0]), np.nanmax(m + st[1]), v]
            valss = np.concatenate([valss, v])
        axs[i].set_title(nam, y=0.85)
    for i in range(len(axs) - len(abund)):
        axs[len(abund) + i].tick_params(axis="y", left=False, labelleft=False)
    minz = min([min(zr) for zr in zrangs])
    maxz = max([max(zr) for zr in zrangs])
    ranz = maxz - minz
    minz = minz - ranz * 0.1
    maxz = maxz + ranz * 0.1
    yrang = np.nan_to_num(flatten(yrang), nan=np.nan, posinf=np.nan, neginf=np.nan)
    ylims = (np.nanmin(yrang), np.nanmax(yrang))
    rana = ylims[1] - ylims[0]
    mina = ylims[0] - rana * 0.1
    maxa = ylims[1] + rana * 0.25
    for i in range(n):
        axs[-i - 1].set_xlim(minz, maxz)
        axs[i * n].set_ylim(mina, maxa)
        axs[-i - 1].set_xlabel(val_name)
        axs[i * n].set_ylabel(yax)
    fig.set_size_inches((max(n, 2) + 0.3) * 2.5, (max(n, 2) + 0.4) * 2.5)
    fig.suptitle(title)
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)
    else:
        plt.show()
    plt.close(fig)


def ratios_in_z(
    sources,
    zrangs,
    val="z",
    val_name=None,
    abund=ac.Oxygen,
    save=None,
    title=None,
    indiv=True,
    indso=None,
    manual=False,
    **kwargs,
):
    n = int(-(-np.sqrt(len(abund)) // 1))
    indss = sources if indso is None else indso
    fig = plt.figure()
    gs = fig.add_gridspec(n, n, hspace=0)
    axes = gs.subplots(sharex="col")
    if n == 1:
        axs = [axes]
    else:
        axs = axes.flatten()
    val_name = val_name if val_name is not None else val
    valss = []
    for i, (nam, ab) in enumerate(abund.items()):
        yrang = []
        for zrang in zrangs:
            sourz = [
                s
                for s in sources
                if s.get(val) is not None and zrang[0] < s[val] < zrang[1]
            ]
            indrz = [
                s
                for s in indss
                if s.get(val) is not None and zrang[0] < s[val] < zrang[1]
            ]
            cal_red = ac.red_const(indrz) if indso is None else None
            if indiv:
                ind, sta = ac.indiv_stat(
                    ab, indrz, val=val, calib=False, cal_red=cal_red, **kwargs
                )
                x = sum(list(ind[0]), [])
                y = sum(list(ind[1]), [])
                if x:
                    zmean = [(zrang[1] + zrang[0]) / 2] * len(sta[0])
                    zerr = [(zrang[1] - zrang[0]) / 2] * len(sta[0])
                    axs[i].plot(
                        x,
                        y,
                        ls="",
                        marker=".",
                        c="gray",
                        alpha=0.15,
                        markersize=2,
                    )
                    axs[i].errorbar(
                        zmean,
                        sta[0],
                        yerr=sta[1],
                        xerr=zerr,
                        ls="",
                        c="gainsboro",
                        capsize=5,
                    )
                    yrang += [
                        np.nanmin(sta[0] - sta[1][0]),
                        np.nanmax(sta[0] + sta[1][1]),
                    ]
            v, m, st = ac.boots_stat(
                ab, sourz, calib=False, cal_red=None, manual=manual, **kwargs
            )
            if v.size:
                zmean = [(zrang[1] + zrang[0]) / 2] * len(v)
                zerr = [(zrang[1] - zrang[0]) / 2] * len(v)
                axs[i].plot(zmean, v, ls="", marker="D", c="black")
                axs[i].errorbar(zmean, v, xerr=zerr, ls="", c="black", capsize=5)
                axs[i].errorbar(zmean, m, yerr=st, ls="", c="black", capsize=5)
                axs[i].set_ylabel(ac.Names[nam], fontsize=11)
                axs[i].yaxis.set_tick_params(labelsize=11)
                yrang += [np.nanmin(m - st[0]), np.nanmax(m + st[1]), v]
            valss = np.concatenate([valss, v])
        yrang = np.nan_to_num(flatten(yrang), nan=np.nan, posinf=np.nan, neginf=np.nan)
        ylims = (np.nanmin(yrang), np.nanmax(yrang))
        axs[i].set_title(nam, y=0.85)
        axs[i].set_ylim(
            ylims[0] - 0.1 * (ylims[1] - ylims[0]),
            ylims[1] + 0.25 * (ylims[1] - ylims[0]),
        )
    for i in range(len(axs) - len(abund)):
        axs[len(abund) + i].tick_params(axis="y", left=False, labelleft=False)
    minz = min([min(zr) for zr in zrangs])
    maxz = max([max(zr) for zr in zrangs])
    ranz = maxz - minz
    minz = minz - ranz * 0.1
    maxz = maxz + ranz * 0.1
    for i in range(n):
        axs[-i - 1].set_xlim(minz, maxz)
        axs[-i - 1].set_xlabel(val_name)
    fig.set_size_inches((max(n, 2) + 0.3) * 2.5, (max(n, 2) + 0.4) * 2.5)
    fig.suptitle(title)
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)
    else:
        plt.show()
    plt.close(fig)


def abundance_calib(
    sources,
    xmetr=ac.O_Dir,
    abund=ac.Oxygen,
    binval="z",
    bins=10,
    save=None,
    title=None,
    xax=None,
    indiv=True,
    indso=None,
    manual=False,
    xbins=None,
    **kwargs,
):
    n = int(-(-np.sqrt(len(abund)) // 1))
    indss = sources if indso is None else indso
    fig = plt.figure()
    gs = fig.add_gridspec(n, n, hspace=0)
    axes = gs.subplots(sharex="col")
    xrang = []
    if n == 1:
        axs = [axes]
    else:
        axs = axes.flatten()
    if xbins is None:
        sbins = catalog.inbins(sources, binval, nbin=bins)
        xbins = []
        for sb in sbins:
            xbins.append((sb, ac.boots_stat(xmetr, sb, manual=manual, **kwargs)))
    vbins = []
    for x in xbins:
        vs = [l[0].get(binval) for l in s]
        vbins.append([min(vs), max(vs)])
    if indiv:
        ibins = [
            [
                [],
            ]
            for v in vbins
        ]
        for s in indss:
            for i, (vl, vh) in enumerate(vbins):
                if (v := s.get(binval)) is not None and vl < v < vh:
                    ibins[i][0].append(s)
        for ib in ibins:
            cal_red = ac.red_const(ib) if indso is None else None
            ib.append(ac.indiv_stat(xmetr, ib, cal_red=cal_red, calib=False)[0])
    valss = []
    for i, (nam, ab) in enumerate(abund.items()):
        yrang = []
        for l, sb in enumerate(xbins):
            sourz = sb[0]
            if indiv:
                cal_red = ac.red_const(ibins[l][0]) if indso is None else None
                indy, _ = ac.indiv_stat(
                    ab, ibins[l][0], calib=False, cal_red=cal_red, **kwargs
                )
                indx, _ = ibins[l][1]
                x, y = ([], [])
                for k in range(len(indy[1])):
                    y += indy[1][i]
                    x += indx[1][0] * len(indy[1][i])
                if x:
                    axs[i].plot(
                        x,
                        y,
                        ls="",
                        marker=".",
                        c="gray",
                        alpha=0.15,
                        markersize=2,
                    )
            v, m, st = ac.boots_stat(
                ab, sourz, calib=False, manual=manual, cal_red=None, **kwargs
            )
            if v.size:
                zv = [sb[1][0][0]] * len(v)
                zm = [sb[1][1][0]] * len(v)
                zerr = [[sb[1][2][0][0]] * len(v), [sb[1][2][1][0]] * len(v)]
                axs[i].plot(zv, v, ls="", marker="D", c="black")
                axs[i].errorbar(zm, m, xerr=zerr, yerr=st, ls="", c="black", capsize=5)
                axs[i].set_ylabel(ac.Names[nam], fontsize=11)
                axs[i].yaxis.set_tick_params(labelsize=11)
                yrang += [np.nanmin(m - st[0]), np.nanmax(m + st[1])]
                xrang += [zm[0] - zerr[0][0], zm[0] + zerr[1][0]]
            valss = np.concatenate([valss, v])
        yrang = np.nan_to_num(flatten(yrang), nan=np.nan, posinf=np.nan, neginf=np.nan)
        ylims = (np.nanmin(yrang), np.nanmax(yrang))
        axs[i].set_title(nam, y=0.85)
        axs[i].set_ylim(
            ylims[0] - 0.1 * (ylims[1] - ylims[0]),
            ylims[1] + 0.25 * (ylims[1] - ylims[0]),
        )
    for i in range(len(axs) - len(abund)):
        axs[len(abund) + i].tick_params(axis="y", left=False, labelleft=False)

    xrang = np.nan_to_num(flatten(xrang), nan=np.nan, posinf=np.nan, neginf=np.nan)
    xlims = (np.nanmin(xrang), np.nanmax(xrang))
    ranx = xlims[1] - xlims[0]
    minx = xlims[0] - ranx * 0.1
    maxx = xlims[1] + ranx * 0.1
    for i in range(n):
        axs[-i - 1].set_xlim(minx, maxx)
        axs[-i - 1].set_xlabel(xax)
    fig.set_size_inches((max(n, 2) + 0.3) * 2.5, (max(n, 2) + 0.4) * 2.5)
    fig.suptitle(title)
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)
    else:
        plt.show()
    plt.close(fig)
    return xbins


def abundance_compar(
    sources,
    xmetr=ac.O_Dir,
    abund=ac.Oxygen,
    binval="z",
    bins=10,
    save=None,
    title=None,
    yax=None,
    xax=None,
    indiv=True,
    indso=None,
    manual=False,
    xbins=None,
    **kwargs,
):
    n = int(-(-np.sqrt(len(abund)) // 1))
    indss = sources if indso is None else indso
    xax = yax if xax is None else xax
    yax = xax if yax is None else yax
    fig = plt.figure()
    gs = fig.add_gridspec(n, n, hspace=0, wspace=0)
    axes = gs.subplots(sharex="col", sharey="row")
    yrang = []
    xrang = []
    if n == 1:
        axs = [axes]
    else:
        axs = axes.flatten()
    if xbins is None:
        sbins = catalog.inbins(sources, binval, nbin=bins)
        xbins = []
        for sb in sbins:
            xbins.append((sb, ac.boots_stat(xmetr, sb, manual=manual, **kwargs)))
    if indiv:
        ibins = [
            [
                [],
            ]
            for v in vbins
        ]
        for s in indss:
            for i, (vl, vh) in enumerate(vbins):
                if (v := s.get(binval)) is not None and vl < v < vh:
                    ibins[i][0].append(s)
        for ib in ibins:
            cal_red = ac.red_const(ib) if indso is None else None
            ib.append(ac.indiv_stat(xmetr, ib, cal_red=cal_red, calib=False)[0])
    valss = []
    for i, (nam, ab) in enumerate(abund.items()):
        for l, sb in enumerate(xbins):
            sourz = sb[0]
            if indiv:
                cal_red = ac.red_const(ibins[l][0]) if indso is None else None
                indy, _ = ac.indiv_stat(ab, ibins[l][0], cal_red=cal_red, **kwargs)
                indx, _ = ibins[l][1]
                x, y = ([], [])
                for k in range(len(indy[1])):
                    y += indy[1][i]
                    x += indx[1][0] * len(indy[1][i])
                if x:
                    axs[i].plot(
                        x,
                        y,
                        ls="",
                        marker=".",
                        c="gray",
                        alpha=0.15,
                        markersize=2,
                    )
            if not sourz:
                continue
            v, m, st = ac.boots_stat(ab, sourz, cal_red=None, manual=manual, **kwargs)
            if v.size:
                zv = [sb[1][0][0]] * len(v)
                zm = [sb[1][1][0]] * len(v)
                zerr = [[sb[1][2][0][0]] * len(v), [sb[1][2][1][0]] * len(v)]
                axs[i].plot(zv, v, ls="", marker="D", c="black")
                axs[i].errorbar(zm, m, xerr=zerr, yerr=st, ls="", c="black", capsize=5)
                yrang += [np.nanmin(m - st[0]), np.nanmax(m + st[1])]
                xrang += [zm[0] - zerr[0][0], zm[0] + zerr[1][0]]
            valss = np.concatenate([valss, v])
        axs[i].set_title(nam, y=0.85)
    for i in range(len(axs) - len(abund)):
        axs[len(abund) + i].tick_params(axis="y", left=False, labelleft=False)

    yrang = np.nan_to_num(flatten(yrang), nan=np.nan, posinf=np.nan, neginf=np.nan)
    ylims = (np.nanmin(yrang), np.nanmax(yrang))
    rana = ylims[1] - ylims[0]
    mina = ylims[0] - rana * 0.1
    maxa = ylims[1] + rana * 0.25
    xrang = np.nan_to_num(flatten(xrang), nan=np.nan, posinf=np.nan, neginf=np.nan)
    xlims = (np.nanmin(xrang), np.nanmax(xrang))
    ranx = xlims[1] - xlims[0]
    minx = xlims[0] - ranx * 0.1
    maxx = xlims[1] + ranx * 0.1
    for i in range(n):
        axs[-i - 1].set_xlim(minx, maxx)
        axs[i * n].set_ylim(mina, maxa)
        axs[-i - 1].set_xlabel(xax)
        axs[i * n].set_ylabel(yax)
    maxc = max(maxx, maxa)
    minc = min(minx, mina)
    for ax in axs:
        ax.axline((0, 0), (1, 1), c="gray", ls="--", alpha=0.7)
        ax.set_xlim(minc, maxc)
        ax.set_ylim(minc, maxc)
    fig.set_size_inches((max(n, 2) + 0.3) * 2.5, (max(n, 2) + 0.4) * 2.5)
    fig.suptitle(title)
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)
    else:
        plt.show()
    plt.close(fig)
    return xbins


def constr_ax(fig, n):
    """Legacy overcomplicated method replaceable by two lines of gridspec"""
    inds = np.linspace(1, n**2, n**2).reshape((n, n))
    inds = np.hstack((inds, np.zeros((n, 1))))
    order = np.delete(d := np.flip(inds, axis=0).flatten(), np.where(d == 0))
    axs = dict()
    for i in order:
        ind = np.where(inds.flatten() == i)[0][0] + 1
        vs = (n, n + 1, ind)
        if i > n * (n - 1) and (i - 1) % n == 0:
            axs[i] = fig.add_subplot(*vs)
        elif i > n * (n - 1):
            axs[i] = fig.add_subplot(*vs, sharey=axs[(i - 1) // n * n + 1])
            axs[i].tick_params(labelleft=False)
        elif (i - 1) % n == 0:
            axs[i] = fig.add_subplot(*vs, sharex=axs[n * (n - 1) + (i - 1) % n + 1])
            axs[i].tick_params(labelbottom=False)
        else:
            axs[i] = fig.add_subplot(
                *vs,
                sharex=axs[n * (n - 1) + (i - 1) % n + 1],
                sharey=axs[(i - 1) // n * n + 1],
            )
            axs[i].tick_params(labelleft=False, labelbottom=False)
    axs = [s[1] for s in sorted(axs.items(), key=lambda x: x[0])]
    axc = fig.add_subplot(
        n,
        n + 1,
        tuple((np.where(inds.flatten() == 0)[0] + 1)[[0, -1]]),
        visible=True,
        aspect=100 * n,
    )
    axc.tick_params(labelleft=False, labelbottom=False)
    return axs, axc


def abundance_in_val_z(
    sources,
    zrangs,
    valrangs,
    val="phot_mass",
    val_name=None,
    zval="z",
    zval_name=None,
    abund=ac.Oxygen,
    save=None,
    title=None,
    yax=None,
    indiv=True,
    indso=None,
    manual=False,
    **kwargs,
):
    val_name = val_name if val_name is not None else val
    zval_name = zval_name if zval_name is not None else zval
    n = int(-(-np.sqrt(len(abund)) // 1))
    indss = sources if indso is None else indso
    rat = 15 * n
    fig = plt.figure(figsize=((max(n, 2) + 0.3) * 2.5, (max(n, 2) + 0.4) * 2.5))
    spec = mpl.gridspec.GridSpec(ncols=2, nrows=1, figure=fig, width_ratios=[rat, 1])
    gs_plots = mpl.gridspec.GridSpecFromSubplotSpec(
        n, n, subplot_spec=spec[0], hspace=0, wspace=0
    )
    axes = gs_plots.subplots(sharex="col", sharey="row")
    cbar_ax = fig.add_subplot(spec[1])
    if n == 1:
        axs = np.array([axes])
    else:
        axs = axes.flatten()
    cmap = mpl.cm.ScalarMappable(cmap="inferno")
    cmap.set_clim((min(flatten(zrangs)), max(flatten(zrangs))))
    cbr = fig.colorbar(
        cmap, location="right", label=zval_name, cax=cbar_ax, aspect=25 * n, pad=0
    )

    # axs, cbar_ax = constr_ax(fig, n)
    yrang = []
    for i, (nam, ab) in enumerate(abund.items()):
        for zrang in zrangs:
            valss = []
            zsour = [
                s
                for s in sources
                if s.get(zval) is not None and zrang[0] < s[zval] < zrang[1]
            ]
            izsour = [
                s
                for s in indss
                if s.get(zval) is not None and zrang[0] < s[zval] < zrang[1]
            ]
            cl = cmap.to_rgba(np.mean(zrang))
            for vrang in valrangs:
                sourz = [
                    s
                    for s in zsour
                    if s.get(val) is not None and vrang[0] < s[val] < vrang[1]
                ]
                isourz = [
                    s
                    for s in izsour
                    if s.get(val) is not None and vrang[0] < s[val] < vrang[1]
                ]
                if isourz:
                    cal_red = ac.red_const(isourz) if indso is None else None
                    if indiv:
                        ind, _ = ac.indiv_stat(
                            ab, isourz, cal_red=cal_red, val=val, **kwargs
                        )
                        ind[0] = sum(list(ind[0]), [])
                        ind[1] = sum(list(ind[1]), [])
                        if ind[0]:
                            axs[i].plot(
                                ind[0],
                                ind[1],
                                ls="",
                                marker=".",
                                c=cl,
                                alpha=0.1,
                                markersize=1.5,
                            )
                if not sourz:
                    continue
                v, m, st = ac.boots_stat(
                    ab, sourz, cal_red=None, manual=manual, **kwargs
                )
                if v.size:
                    vmean = [(vrang[1] + vrang[0]) / 2] * len(v)
                    verr = [(vrang[1] - vrang[0]) / 2] * len(v)
                    axs[i].plot(vmean, v, ls="", marker="D", c=cl)
                    axs[i].errorbar(
                        vmean, v, xerr=verr, ls="", c=cl, capsize=5, alpha=0.3
                    )
                    axs[i].errorbar(vmean, m, yerr=st, ls="", c=cl, capsize=5)
                    yrang += [np.nanmin(m - st[0]), np.nanmax(m + st[1])]
                    valss.append([vmean[0]] + list(m))
            if valss:
                vsm = max([len(i) for i in valss])
                valss = np.array([v + [np.nan] * (vsm - len(v)) for v in valss])
                axs[i].plot(valss[:, 0], valss[:, 1:], c=cl, alpha=0.5)
        axs[i].set_title(nam, y=0.85)
    for i in range(len(axs) - len(abund)):
        axs[len(abund) + i].tick_params(axis="y", left=False, labelleft=False)
    minz = min([min(zr) for zr in valrangs])
    maxz = max([max(zr) for zr in valrangs])
    ranz = maxz - minz
    minz = minz - ranz * 0.1
    maxz = maxz + ranz * 0.1
    yrang = np.nan_to_num(flatten(yrang), nan=np.nan, posinf=np.nan, neginf=np.nan)
    ylims = (np.nanmin(yrang), np.nanmax(yrang))
    rana = ylims[1] - ylims[0]
    mina = ylims[0] - rana * 0.1
    maxa = ylims[1] + rana * 0.25
    for i in range(n):
        axs[-i - 1].set_xlim(minz, maxz)
        axs[i * n].set_ylim(mina, maxa)
        axs[-i - 1].set_xlabel(val_name)
        axs[i * n].set_ylabel(yax)
    # for ax in axs: ax.tick_params(**{k:True for k in ['top', 'bottom', 'left', 'right']}, direction='in')
    fig.suptitle(title)
    fig.set_layout_engine(layout="tight")

    if save is not None:
        fig.savefig(save)
    else:
        plt.show()
    plt.close(fig)


def abundance_compar_z(
    sources,
    zrangs,
    valrangs,
    val="phot_mass",
    zval="z",
    zval_name=None,
    xmetr=ac.O_Dir,
    abund=ac.Oxygen,
    save=None,
    title=None,
    yax=None,
    xax=None,
    lim=None,
    indiv=True,
    indso=None,
    manual=False,
    **kwargs,
):
    xax = yax if xax is None else xax
    yax = xax if yax is None else yax
    indss = sources if indso is None else indso
    zval_name = zval_name if zval_name is not None else zval
    n = int(-(-np.sqrt(len(abund)) // 1))

    rat = 15 * n
    fig = plt.figure(figsize=((max(n, 2) + 0.3) * 2.5, (max(n, 2) + 0.4) * 2.5))
    spec = mpl.gridspec.GridSpec(ncols=2, nrows=1, figure=fig, width_ratios=[rat, 1])
    gs_plots = mpl.gridspec.GridSpecFromSubplotSpec(
        n, n, subplot_spec=spec[0], hspace=0, wspace=0
    )
    axes = gs_plots.subplots(sharex="col", sharey="row")
    cbar_ax = fig.add_subplot(spec[1])
    if n == 1:
        axs = np.array([axes])
    else:
        axs = axes.flatten()
    cmap = mpl.cm.ScalarMappable(cmap="inferno")
    cmap.set_clim((min(flatten(zrangs)), max(flatten(zrangs))))
    cbr = fig.colorbar(
        cmap, location="right", label=zval_name, cax=cbar_ax, aspect=25 * n, pad=0
    )

    yrang = []
    xrang = []
    zs = []
    izs = []
    for zrang in zrangs:
        zsour = [
            s
            for s in sources
            if s.get(zval) is not None and zrang[0] < s[zval] < zrang[1]
        ]
        izsour = [
            s
            for s in indss
            if s.get(zval) is not None and zrang[0] < s[zval] < zrang[1]
        ]
        vs = []
        ivs = []
        for vrang in valrangs:
            sourz = [
                s
                for s in zsour
                if s.get(val) is not None and vrang[0] < s[val] < vrang[1]
            ]
            if sourz:
                vs.append((sourz, ac.boots_stat(xmetr, sourz, manual=manual, **kwargs)))
            else:
                vs.append((sourz, None))
            isourz = [
                s
                for s in izsour
                if s.get(val) is not None and vrang[0] < s[val] < vrang[1]
            ]
            if isourz and indiv:
                ivs.append((isourz, ac.indiv_stat(xmetr, isourz, **kwargs)))
            else:
                ivs.append((isourz, None))
        zs.append((zrang, vs))
        izs.append(ivs)
    for i, (nam, ab) in enumerate(abund.items()):
        for l in range(len(zrangs)):
            zrang, vs = zs[l]
            ivs = izs[l]
            cl = cmap.to_rgba(np.mean(zrang))
            valss = []
            for k in range(len(valrangs)):
                so, db = vs[k]
                iso, idb = ivs[k]
                if indiv:
                    cal_red = ac.red_const(iso) if indso is None else None
                    indy, _ = ac.indiv_stat(ab, iso, cal_red=cal_red, **kwargs)
                    indx, _ = idb
                    x, y = ([], [])
                    for k in range(len(indy[1])):
                        y += indy[1][i]
                        x += indx[1][0] * len(indy[1][i])
                    if x:
                        axs[i].plot(
                            x,
                            y,
                            ls="",
                            marker=".",
                            c="gray",
                            alpha=0.15,
                            markersize=2,
                        )
                if not so:
                    continue
                v, m, st = ac.boots_stat(ab, so, cal_red=None, manual=manual, **kwargs)
                if v.size:
                    zv = [db[0][0]] * len(v)
                    zm = [db[1][0]] * len(v)
                    zerr = [[db[2][0][0]] * len(v), [db[2][1][0]] * len(v)]
                    axs[i].plot(zv, v, ls="", marker="D", c=cl)
                    axs[i].errorbar(zm, m, xerr=zerr, yerr=st, ls="", c=cl, capsize=5)
                    yrang += [np.nanmin(m - st[0]), np.nanmax(m + st[1])]
                    xrang += [zm[0] - zerr[0][0], zm[0] + zerr[1][0]]
                    valss.append([zm[0]] + list(m))
            if valss:
                vsm = max([len(i) for i in valss])
                valss = np.array([v + [np.nan] * (vsm - len(v)) for v in valss])
                axs[i].plot(valss[:, 0], valss[:, 1:], c=cl, alpha=0.5)
        axs[i].set_title(nam, y=0.85)
    for i in range(len(axs) - len(abund)):
        axs[len(abund) + i].tick_params(axis="y", left=False, labelleft=False)
    crang = np.nan_to_num(
        flatten(xrang + yrang), nan=np.nan, posinf=np.nan, neginf=np.nan
    )
    clims = (np.nanmin(crang), np.nanmax(crang))
    ranc = clims[1] - clims[0]
    minc = clims[0] - ranc * 0.1 if lim is None else lim[0]
    maxc = clims[1] + ranc * 0.1 if lim is None else lim[1]
    for ax in axs:
        ax.axline((0, 0), (1, 1), c="gray", ls="--", alpha=0.7)
        ax.set_xlim(minc, maxc)
        ax.set_ylim(minc, maxc)
    for i in range(n):
        axs[-i - 1].set_xlabel(xax)
        axs[i * n].set_ylabel(yax)

    fig.suptitle(title)
    fig.set_layout_engine(layout="tight")

    if save is not None:
        fig.savefig(save)
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    f = catalog.fetch_json("../catalog_f.json")["sources"]
    for s in f:
        s["_pmass"] = np.log10(m) if (m := s.get("phot_mass")) is not None else None
    ff = catalog.rm_bad(f)
    ffm = [s for s in ff if s["grat"][0] == "g" and s["grat"][-1] == "m"]
    ffmu = catalog.unique(ffm)
    """
    abundance_in_z(#yess
        ffm,
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6.5], [6.5, 8], [8, 12]],
        abund=ac.Sulphur,
        title="Sulphur abundance in medium resolution\n via different calibrations",
        yax="$12+\\mathrm{log}(\\mathrm{S}/\\mathrm{H})$",
        save="../Plots/abund/sulphur_cal.pdf",
        indso=ffmu,
    )
    abundance_in_z(#yess
        ffm,
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6.5], [6.5, 8], [8, 12]],
        abund=ac.Nitrogen,
        title="Nitrogen abundance in medium resolution\n via different calibrations",
        yax="$\\mathrm{log}(\\mathrm{N}/\\mathrm{O})$",
        save="../Plots/abund/nitrogen_cal.pdf",
        indso=ffmu,
    )
    abundance_in_z(#yess
        ffm,
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6.5], [6.5, 8], [8, 12]],
        abund=ac.Oxygen,
        title="Oxygen abundance in medium resolution via different calibrations",
        yax="$12+\\mathrm{log}(\\mathrm{O}/\\mathrm{H})$",
        save="../Plots/abund/oxygen_cal.pdf",
        indso=ffmu,
    )
    
    ratios_in_z(#yess
        ffm,
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6.5], [6.5, 8], [8, 12]],
        abund=ac.Sulphur,
        title="Line fluxes for sulphur abundance calibration\n in medium resolution",
        save="../Plots/abund/sulphur_flu.pdf",
        indso=ffmu,
    )
    ratios_in_z(
        ffm,
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6.5], [6.5, 8], [8, 12]],
        abund=ac.Nitrogen,
        title="Line fluxes for nitrogen abundance calibration\n in medium resolution",
        save="../Plots/abund/nitrogen_flu.pdf",
        indso=ffmu,
    )
    ratios_in_z(
        ffm,
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6.5], [6.5, 8], [8, 12]],
        abund=ac.Oxygen,
        title="Line fluxes for oxygen abundance calibration in medium resolution",
        save="../Plots/abund/oxygen_flu.pdf",
        indso=ffmu,
    )
    
    abundance_compar(
        ffm,
        xmetr=ac.S_Dir,
        abund=ac.Sulphur,
        binval="z",
        bins=10,
        save="../Plots/abund/sulphur_com.pdf",
        title="Sulphur abundance in medium resolution\n via direct method and strong lines",
        yax="$12+\\mathrm{log}(\\mathrm{S}/\\mathrm{H})$",
        indso=ffmu,
    )
    abundance_compar(
        ffm,
        xmetr=ac.N_Dir,
        abund=ac.Nitrogen,
        binval="z",
        bins=10,
        save="../Plots/abund/nitrogen_com.pdf",
        title="Nitrogen abundance in medium resolution\n via direct method and strong lines",
        yax="$\\mathrm{log}(\\mathrm{N}/\\mathrm{O})$",
        indso=ffmu,
    )
    abundance_compar(
        ffm,
        xmetr=ac.O_Dir,
        abund=ac.Oxygen,
        binval="z",
        bins=10,
        save="../Plots/abund/oxygen_com.pdf",
        title="Oxygen abundance in medium resolution\n via direct method and strong lines",
        yax="$12+\\mathrm{log}(\\mathrm{O}/\\mathrm{H})$",
        indso=ffmu,
    )
    
    abundance_calib(
        ffm,
        xmetr=ac.S_Dir,
        abund=ac.Sulphur,
        binval="z",
        bins=10,
        save="../Plots/abund/sulphur_clf.pdf",
        title="Sulphur abundance in medium resolution\n via compared to broad line calibrations",
        xax="$12+\\mathrm{log}(\\mathrm{S}/\\mathrm{H})$",
        indso=ffmu,
    )
    abundance_calib(
        ffm,
        xmetr=ac.N_Dir,
        abund=ac.Nitrogen,
        binval="z",
        bins=10,
        save="../Plots/abund/nitrogen_clf.pdf",
        title="Nitrogen abundance in medium resolution\n via compared to broad line calibrations",
        xax="$\\mathrm{log}(\\mathrm{N}/\\mathrm{O})$",
        indso=ffmu,
    )
    abundance_calib(
        ffm,
        xmetr=ac.O_Dir,
        abund=ac.Oxygen,
        binval="z",
        bins=10,
        save="../Plots/abund/oxygen_clf.pdf",
        title="Oxygen abundance in medium resolution\n via compared to broad line calibrations",
        xax="$12+\\mathrm{log}(\\mathrm{O}/\\mathrm{H})$",
        indso=ffmu,
    )
    
    abundance_in_z(
        ffm,
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6.5], [6.5, 8], [8, 12]],
        abund={"S Direct": ac.S_Dir},
        title="Sulphur abundance in medium resolution\n via direct method",
        yax="$12+\\mathrm{log}(\\mathrm{S}/\\mathrm{H})$",
        save="../Plots/abund/sulphur_dir.pdf",
        #manual=True,
        indso=ffmu,
    )
    abundance_in_z(
        ffm,
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6.5], [6.5, 8], [8, 12]],
        abund={"N Direct": ac.N_Dir},
        title="Nitrogen abundance in medium resolution\n via direct method",
        yax="$12+\\mathrm{log}(\\mathrm{N}/\\mathrm{H})$",
        save="../Plots/abund/nitrogen_dir.pdf",
        #manual=True,
        indso=ffmu,
    )
    abundance_in_z(
        ffm,
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6.5], [6.5, 8], [8, 12]],
        abund={"O Direct": ac.O_Dir},
        title="Oxygen abundance in medium resolution\n via direct method",
        yax="$12+\\mathrm{log}(\\mathrm{O}/\\mathrm{H})$",
        save="../Plots/abund/oxygen_dir.pdf",
        #manual=True,
        indso=ffmu,
    )

    abundance_in_z(
        ffm,
        [[i, i + 1] for i in range(6, 12)],
        val="_pmass",
        val_name="$\\mathrm{log} (M_\\star/M_\\odot)$",
        abund=ac.Sulphur,
        title="Sulphur abundance in medium resolution\n via different calibrations",
        yax="$12+\\mathrm{log}(\\mathrm{S}/\\mathrm{H})$",
        save="../Plots/abund/sulphur_cal_mass.pdf",
        indso=ffmu,
    )
    abundance_in_z(
        ffm,
        [[i, i + 1] for i in range(6, 12)],
        val="_pmass",
        val_name="$\\mathrm{log} (M_\\star/M_\\odot)$",
        abund=ac.Nitrogen,
        title="Nitrogen abundance in medium resolution\n via different calibrations",
        yax="$\\mathrm{log}(\\mathrm{N}/\\mathrm{O})$",
        save="../Plots/abund/nitrogen_cal_mass.pdf",
        indso=ffmu,
    )
    abundance_in_z(
        ffm,
        [[i, i + 1] for i in range(6, 12)],
        val="_pmass",
        val_name="$\\mathrm{log} (M_\\star/M_\\odot)$",
        abund=ac.Oxygen,
        title="Oxygen abundance in medium resolution via different calibrations",
        yax="$12+\\mathrm{log}(\\mathrm{O}/\\mathrm{H})$",
        save="../Plots/abund/oxygen_cal_mass.pdf",
        indso=ffmu,
    )

    ratios_in_z(
        ffm,
        [[i, i + 1] for i in range(6, 12)],
        val="_pmass",
        val_name="$\\mathrm{log} (M_\\star/M_\\odot)$",
        abund=ac.Sulphur,
        title="Line fluxes for sulphur abundance calibration\n in medium resolution",
        save="../Plots/abund/sulphur_flu_mass.pdf",
        indso=ffmu,
    )
    ratios_in_z(
        ffm,
        [[i, i + 1] for i in range(6, 12)],
        val="_pmass",
        val_name="$\\mathrm{log} (M_\\star/M_\\odot)$",
        abund=ac.Nitrogen,
        title="Line fluxes for nitrogen abundance calibration\n in medium resolution",
        save="../Plots/abund/nitrogen_flu_mass.pdf",
        indso=ffmu,
    )
    ratios_in_z(
        ffm,
        [[i, i + 1] for i in range(6, 12)],
        val="_pmass",
        val_name="$\\mathrm{log} (M_\\star/M_\\odot)$",
        abund=ac.Oxygen,
        title="Line fluxes for oxygen abundance calibration in medium resolution",
        save="../Plots/abund/oxygen_flu_mass.pdf",
        indso=ffmu,
    )
    
    abundance_compar(
        ffm,
        xmetr=ac.S_Dir,
        abund=ac.Sulphur,
        binval="phot_mass",
        bins=10,
        save="../Plots/abund/sulphur_com_mass.pdf",
        title="Sulphur abundance in medium resolution\n via direct method and strong lines",
        yax="$12+\\mathrm{log}(\\mathrm{S}/\\mathrm{H})$",
        indso=ffmu,
    )
    abundance_compar(
        ffm,
        xmetr=ac.N_Dir,
        abund=ac.Nitrogen,
        binval="phot_mass",
        bins=10,
        save="../Plots/abund/nitrogen_com_mass.pdf",
        title="Nitrogen abundance in medium resolution\n via direct method and strong lines",
        yax="$\\mathrm{log}(\\mathrm{N}/\\mathrm{O})$",
        indso=ffmu,
    )
    abundance_compar(
        ffm,
        xmetr=ac.O_Dir,
        abund=ac.Oxygen,
        binval="phot_mass",
        bins=10,
        save="../Plots/abund/oxygen_com_mass.pdf",
        title="Oxygen abundance in medium resolution\n via direct method and strong lines",
        yax="$12+\\mathrm{log}(\\mathrm{O}/\\mathrm{H})$",
        indso=ffmu,
    )
    
    abundance_calib(
        ffm,
        xmetr=ac.S_Dir,
        abund=ac.Sulphur,
        binval="phot_mass",
        bins=10,
        save="../Plots/abund/sulphur_clf_mass.pdf",
        title="Sulphur abundance in medium resolution\n via compared to broad line calibrations",
        xax="$12+\\mathrm{log}(\\mathrm{S}/\\mathrm{H})$",
        indso=ffmu,
    )
    abundance_calib(
        ffm,
        xmetr=ac.N_Dir,
        abund=ac.Nitrogen,
        binval="phot_mass",
        bins=10,
        save="../Plots/abund/nitrogen_clf_mass.pdf",
        title="Nitrogen abundance in medium resolution\n via compared to broad line calibrations",
        xax="$\\mathrm{log}(\\mathrm{N}/\\mathrm{O})$",
        indso=ffmu,
    )
    abundance_calib(
        ffm,
        xmetr=ac.O_Dir,
        abund=ac.Oxygen,
        binval="phot_mass",
        bins=10,
        save="../Plots/abund/oxygen_clf_mass.pdf",
        title="Oxygen abundance in medium resolution\n via compared to broad line calibrations",
        xax="$12+\\mathrm{log}(\\mathrm{O}/\\mathrm{H})$",
        indso=ffmu,
    )
    
    abundance_in_z(
        ffm,
        [[i, i + 1] for i in range(6, 12)],
        val="_pmass",
        val_name="$\\mathrm{log} (M_\\star/M_\\odot)$",
        abund={"S Direct": ac.S_Dir},
        title="Sulphur abundance in medium resolution\n via direct method",
        yax="$12+\\mathrm{log}(\\mathrm{S}/\\mathrm{H})$",
        save="../Plots/abund/sulphur_dir_mass.pdf",
        #manual=True,
        indso=ffmu,
    )
    abundance_in_z(
        ffm,
        [[i, i + 1] for i in range(6, 12)],
        val="_pmass",
        val_name="$\\mathrm{log} (M_\\star/M_\\odot)$",
        abund={"N Direct": ac.N_Dir},
        title="Nitrogen abundance in medium resolution\n via direct method",
        yax="$12+\\mathrm{log}(\\mathrm{N}/\\mathrm{H})$",
        save="../Plots/abund/nitrogen_dir_mass.pdf",
        #manual=True,
        indso=ffmu,
    )
    abundance_in_z(
        ffm,
        [[i, i + 1] for i in range(6, 12)],
        val="_pmass",
        val_name="$\\mathrm{log} (M_\\star/M_\\odot)$",
        abund={"O Direct": ac.O_Dir},
        title="Oxygen abundance in medium resolution\n via direct method",
        yax="$12+\\mathrm{log}(\\mathrm{O}/\\mathrm{H})$",
        save="../Plots/abund/oxygen_dir_mass.pdf",
        #manual=True,
        indso=ffmu,
    )
    
    abundance_in_val_z(
        ffm,
        [[0, 1.5], [1.5, 3], [3, 5], [5, 7], [7, 12]],
        [[i, i + 1] for i in range(6, 12)],
        val="_pmass",
        val_name="$\\mathrm{log} (M_\\star/M_\\odot)$",
        abund=ac.Sulphur,
        title="Sulphur abundance in medium resolution\n via different calibrations",
        yax="$12+\\mathrm{log}(\\mathrm{S}/\\mathrm{H})$",
        save="../Plots/abund/sulphur_cal_z_mass.pdf",
        zval_name = "Redshift $z$",
        indso=ffmu,
    )
    abundance_in_val_z(
        ffm,
        [[0, 1.5], [1.5, 3], [3, 5], [5, 7], [7, 12]],
        [[i, i + 1] for i in range(6, 12)],
        val="_pmass",
        val_name="$\\mathrm{log} (M_\\star/M_\\odot)$",
        abund=ac.Nitrogen,
        title="Nitrogen abundance in medium resolution\n via different calibrations",
        yax="$\\mathrm{log}(\\mathrm{N}/\\mathrm{O})$",
        save="../Plots/abund/nitrogen_cal_z_mass.pdf",
        zval_name = "Redshift $z$",
        indso=ffmu,
    )
    abundance_in_val_z(
        ffm,
        [[0, 1.5], [1.5, 3], [3, 5], [5, 7], [7, 12]],
        [[i, i + 1] for i in range(6, 12)],
        val="_pmass",
        val_name="$\\mathrm{log} (M_\\star/M_\\odot)$",
        abund=ac.Oxygen,
        title="Oxygen abundance in medium resolution via different calibrations",
        yax="$12+\\mathrm{log}(\\mathrm{O}/\\mathrm{H})$",
        save="../Plots/abund/oxygen_cal_z_mass.pdf",
        zval_name = "Redshift $z$",
        indso=ffmu,
    )
    
    abundance_in_val_z(
        ffm,
        [[0, 1.5], [1.5, 3], [3, 5], [5, 7], [7, 12]],
        [[i, i + 1] for i in range(6, 12)],
        val="_pmass",
        val_name="$\\mathrm{log} (M_\\star/M_\\odot)$",
        abund={"S Direct": ac.S_Dir},
        title="Sulphur abundance in medium resolution\n via direct method",
        yax="$12+\\mathrm{log}(\\mathrm{S}/\\mathrm{H})$",
        save="../Plots/abund/sulphur_dir_z_mass.pdf",
        zval_name = "Redshift $z$",
        indso=ffmu,
    )
    abundance_in_val_z(
        ffm,
        [[0, 1.5], [1.5, 3], [3, 5], [5, 7], [7, 12]],
        [[i, i + 1] for i in range(6, 12)],
        val="_pmass",
        val_name="$\\mathrm{log} (M_\\star/M_\\odot)$",
        abund={"N Direct": ac.N_Dir},
        title="Nitrogen abundance in medium resolution\n via direct method",
        yax="$\\mathrm{log}(\\mathrm{N}/\\mathrm{O})$",
        save="../Plots/abund/nitrogen_dir_z_mass.pdf",
        zval_name = "Redshift $z$",
        indso=ffmu,
    )
    abundance_in_val_z(
        ffm,
        [[0, 1.5], [1.5, 3], [3, 5], [5, 7], [7, 12]],
        [[i, i + 1] for i in range(6, 12)],
        val="_pmass",
        val_name="$\\mathrm{log} (M_\\star/M_\\odot)$",
        abund={"O Direct": ac.O_Dir},
        title="Oxygen abundance in medium resolution\n via direct method",
        yax="$12+\\mathrm{log}(\\mathrm{O}/\\mathrm{H})$",
        save="../Plots/abund/oxygen_dir_z_mass.pdf",
        zval_name = "Redshift $z$",
        indso=ffmu,
    )
    
    abundance_compar_z(
        ffm,
        [[0, 1.5], [1.5, 3], [3, 5], [5, 7], [7, 12]],
        [[i, i + 1] for i in range(6, 12)],
        val="_pmass",
        zval="z",
        xmetr=ac.S_Dir,
        abund=ac.Sulphur,
        save="../Plots/abund/sulphur_com_z_mass.pdf",
        title="Sulphur abundance in medium resolution\n via direct method and strong lines",
        yax="$12+\\mathrm{log}(\\mathrm{S}/\\mathrm{H})$",
        zval_name = "Redshift $z$",
        indso=ffmu,
    )
    abundance_compar_z(
        ffm,
        [[0, 1.5], [1.5, 3], [3, 5], [5, 7], [7, 12]],
        [[i, i + 1] for i in range(6, 12)],
        val="_pmass",
        zval="z",
        xmetr=ac.N_Dir,
        abund=ac.Nitrogen,
        save="../Plots/abund/nitrogen_com_z_mass.pdf",
        title="Nitrogen abundance in medium resolution\n via direct method and strong lines",
        yax="\\mathrm{log}(\\mathrm{N}/\\mathrm{O})$",
        zval_name = "Redshift $z$",
        lim = [-2.15,0],
        indso=ffmu,
    )
    abundance_compar_z(
        ffm,
        [[0, 1.5], [1.5, 3], [3, 5], [5, 7], [7, 12]],
        [[i, i + 1] for i in range(6, 12)],
        val="_pmass",
        zval="z",
        xmetr=ac.O_Dir,
        abund=ac.Oxygen,
        save="../Plots/abund/oxygen_com_z_mass.pdf",
        title="Oxygen abundance in medium resolution\n via direct method and strong lines",
        yax="$12+\\mathrm{log}(\\mathrm{O}/\\mathrm{H})$",
        zval_name = "Redshift $z$",
        lim = [7.35,8.8],
        indso=ffmu,
    )
    """
