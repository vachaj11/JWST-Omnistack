"""Holds various methods for construction, loading, saving and manipulation of local version of source catalog

Attributes:
    quasars (list): List of individual sources names which have been identified as LRDs or quasars.
"""

import json

import numpy as np


def save_as_json(thing, path):
    """Saves provided object as `json` file at specified path.

    Args:
        thing (dict, list, etc): Thing to be saved onto the path, should
            be serialisable in the ``.json`` file-structure, otherwise fails.
        path (str): Path where the object is to be saved. Should include the
            ".json" ending.
    """
    fil = open(path, "w")
    json.dump(thing, fil, indent=4)
    fil.close()


def fetch_json(path):
    """Fetches data of `json` object at specified path.

    Args:
        path (str): Full path at which the ``.json`` file is to be found.

    Returns:
        dict, list, etc.: Contents of the `json` file found at the path
            converted into equivalent python objects.
    """
    fil = open(path, "r")
    data = json.load(fil)
    fil.close()
    return data


def join_sources(sources, adiff=0.0003):
    """Creates dictionary of lists joining different spectra of the same source under a unique key."""
    snew = dict()
    for source in sources:
        iden = f"{source['root']}/{source['srcid']}"
        if (s := snew.get(iden)) is not None and (
            abs(source["ra"] - s[-1]["ra"]) < adiff
            and abs(source["dec"] - s[-1]["dec"]) < adiff
            and abs(source["z"] - s[-1]["z"] < 0.5)
        ):
            snew[iden].append(source)
        else:
            snew[iden] = [source]
    return snew


def unique(sources, adiff=0.0003):
    """From a list of spectra constructs a list of unique sources, identified by source id or coordinates.

    Assumes that same "srcid" are assigned only to same sources within catalogs. Otherwise also checks by coordinate matching.
    """
    sourn = []
    for source in sources:
        new = True
        sn = source["sn50"]
        for i, s in enumerate(sourn):
            if (source["srcid"] == s["srcid"] and source["root"] == s["root"]) and (
                abs(source["ra"] - s["ra"]) < adiff
                and abs(source["dec"] - s["dec"]) < adiff
                and abs(source["z"] - s["z"] < 0.5)
            ):
                if sn is not None and sn > s["sn50"]:
                    sourn[i] = copy_params(s, source)
                else:
                    sourn[i] = copy_params(source, s)
                new = False
                mn = np.log10(m) if (m := source["phot_mass"]) is not None else np.nan
                m = np.log10(m) if (m := s["phot_mass"]) is not None else np.nan
                if (d := abs(mn - m)) > 0.5 or not np.isfinite(d):
                    print(
                        f'Disagreeing masses for {s["srcid"]}\n {mn:.2f} and {m:.2f}.'
                    )
                    print(
                        f'Differences {source["ra"]-s["ra"]:.2e}, {source["dec"]-s["dec"]:.2e}, {source["z"]:.2f}|{s["z"]:.2f}.'
                    )
                break
        if new:
            sourn.append(source)
    return sourn


def copy_params(s_from, s_to):
    """Copy specific parameters from one catalog item to another."""
    copynam = ["phot_mass", "z"]
    for k, v in s_from.items():
        if (k[:4] == "rec_" or k in copynam) and s_to.get(k) is None:
            s_to[k] = v
    return s_to


def rm_bad(sources, ppxf=False, agn=False):
    """Create instance of inputted catalog with only sources fulfilling certain criteria."""
    sourn = []
    for source in sources:
        sn = source["sn50"]
        gr = source["grade"]
        cm = str(source["comment"]) + str(source.get("old_comment"))
        ha = source.get("spec_Ha")
        cont = source.get("cont_diff")
        agn = source.get("AGN_cand")
        n2 = source.get("N2")
        if (
            sn is not None
            and sn > 0
            and gr is not None
            and gr > 2
            and "star" not in cm
            and "Star" not in cm
            # and (ha is None or ha < 80)
            and (cont is None or cont < 10 or not ppxf)
            # and (agn is None or agn < 3)
            and (n2 is None or n2 < 0 or agn)
        ):
            sourn.append(source)
    return rm_quasars(sourn) if not agn else sourn


quasars = [
    ["gdn-fujimoto-v", "4762_33609.spec.fits"],
    ["gdn-fujimoto-v", "4762_37393.spec.fits"],
    ["j0226-wang-v", "3325_6699.spec.fits"],
    ["j0910-wang-v", "2028_12910.spec.fits"],
    ["jades-gdn2-v", "1181_954.spec.fits"],
    ["rubies-egs61-v", "4233_55604.spec.fits"],
    ["rubies-egs63-v", "4233_42803.spec.fits"],
    ["rubies-egs63-v", "4233_49140.spec.fits"],
    ["egs-mason-v", "4287_62859.spec.fits"],
    ["excels-uds03-v", "3543_70639.spec.fits"],
    ["rubies-egs52-v", "4233_37124.spec.fits"],
    ["rubies-uds42-v", "4233_807469.spec.fits"],
    ["valentino-cosmos04-v", "3567_47567.spec.fits"],
    ["rubies-uds42-v", "4233_36171.spec.fits"],
]


def rm_quasars(sources, quasars=quasars):
    """Remove sources from a catalog, which have been individually identified as LRDs or quasars."""
    sourn = []
    for source in sources:
        fn = source["file"]
        link = False
        for nam in quasars:
            if fn.startswith(nam[0]) and fn.endswith(nam[1]):
                link = True
        if not link:
            sourn.append(source)
    return sourn


def check_zrange(source, rang, z_shift=True):
    """Check whether catalog item falls within given redshift range"""
    if z_shift:
        z = source["z"]
    else:
        z = 0.0
    r = [rang[0] * (1 + z), rang[1] * (1 + z)]
    inr = False
    for rs in source["range"]:
        if rs[0] < r[0] < rs[1] and rs[0] < r[1] < rs[1]:
            return True
    return False


def filter_zrange(sources, rang, z_shift=True):
    """Filter given catalog for only sources which have full coverage of a given redshift range."""
    sourn = []
    for source in sources:
        if check_zrange(source, rang, z_shift=z_shift):
            sourn.append(source)
    return sourn


def filter_zranges(sources, ranges, z_shift=True):
    """Filter given catalog for only sources which have full coverage of provided redshift ranges."""
    for rang in ranges:
        sources = filter_zrange(sources, rang, z_shift=z_shift)
    return sources


def value_range(sources, value, rang):
    """Filter given catalog for only sources which have a specified parameter and the parameter's value falls within specified range."""
    sourn = []
    for source in sources:
        if (v := source.get(value)) is not None and rang[0] < v < rang[1]:
            sourn.append(source)
        elif v is None:
            print(f"Value {value} not found for source {source['srcid']}.")
    return sourn


def value_bins(sources, value, **kwargs):
    """Divide a provided catalog into binned sub-catalogs on the basis of a given value.

    The bins are created equally spaced in the value range.
    """
    values = []
    sbins = []
    for source in sources:
        if (v := source.get(value)) is not None:
            values.append(v)
        else:
            print(f"\r Value {value} not found for source {source['srcid']}.", end="")
    hist, bins = np.histogram(values, **kwargs)
    for i in range(len(bins) - 1):
        sbins.append(value_range(sources, value, (bins[i], bins[i + 1])))
    return hist, bins, sbins


def inbins(sources, value, nbin=10):
    """Divide a provided catalog into binned sub-catalogs on the basis of a given value.

    The bins are created such that each hold equal number of sources.
    """
    sours = []
    for s in sources:
        if value in s.keys() and s[value] is not None:
            sours.append((s, s[value]))
    sours.sort(key=lambda x: x[1])
    sbinned = []
    size = int(-((-len(sours) / nbin) // 1))
    for i in range(nbin):
        imin = i * size
        imax = (i + 1) * size
        sbinned.append(list(zip(*sours[imin:imax]))[0])
    return sbinned


def getffm():
    """Convenience function to quickly fetch catalog of filtered middle resolution sources."""
    f = fetch_json("../catalog_v4.json")["sources"]
    ff = rm_bad(f)
    ffm = [s for s in ff if s["grat"][0] == "g"]
    return ffm


def counts_table(
    sources,
    zranges=[[0, 1.5], [1.5, 3], [3, 5], [5, 7], [7, 12]],
    mranges=[[6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12]],
):
    """Return LaTeX table detailing numbers of sources in provided catalogue in specified redshift and mass bins."""
    swm = [s for s in sources if s.get("phot_mass") is not None]
    for s in swm:
        s["_pmass"] = np.log10(m) if (m := s.get("phot_mass")) is not None else None
    mvals = []
    for mr in mranges:
        zvals = []
        for zr in zranges:
            l = len(
                [
                    True
                    for s in swm
                    if zr[0] < s["z"] < zr[1] and mr[0] < s["_pmass"] < mr[1]
                ]
            )
            zvals.append(l)
        mvals.append(zvals)
    mv = np.array(mvals)
    mvz = np.vstack([mv, mv.sum(axis=0)])
    mvh = np.hstack([mvz, np.array([mvz.sum(axis=1)]).T])
    mvs = mvh.astype(str)
    trows = [f"{mr[0]}-{mr[1]} & " for mr in mranges] + ["Combined & "]
    erows = [r"\\\hline" for mr in mranges]
    erows.insert(-1, r"\\\hline\hline")
    lines = ["\t\t" + trows[i] + " & ".join(r) + erows[i] for i, r in enumerate(mvs)]
    sline = "\t" + r"\begin{tabular}{|c||" + "|".join(["c" for i in zranges]) + "||l|}"
    fline = (
        "\t\t"
        + r"\hline\diagbox{$\mathrm{log}(M_\star/M_\odot)$}{$z$} & "
        + " & ".join([f"{zr[0]}-{zr[1]}" for zr in zranges])
        + r" & Combined\\\hline\hline"
    )
    eline = "\t" + r"\end{tabular}"
    return "\n".join([sline, fline] + lines + [eline])
