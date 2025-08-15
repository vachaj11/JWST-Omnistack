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


def unique(sources):
    """Assumes that same "srcid" are assigned only to same sources, which might not hold across catalogues."""
    sourn = []
    for source in sources:
        new = True
        sn = source["sn50"]
        for i in range(len(sourn)):
            s = sourn[i]
            if (
                source["srcid"] == s["srcid"]
                and source["root"] == s["root"]
                and abs(source["ra"] - s["ra"]) < 0.005
                and abs(source["dec"] - s["dec"]) < 0.005
            ):
                if sn is not None and s["sn50"] < sn:
                    sourn[i] = source
                new = False
                if source["phot_mass"] != s["phot_mass"]:
                    print(
                        f'Disagreeing masses for {s["srcid"]}\n {source["phot_mass"]} and {s["phot_mass"]}.'
                    )
                    print(
                        f'Differences {source["ra"]-s["ra"]}, {source["dec"]-s["dec"]}, {source["z"]}|{s["z"]}.'
                    )
                break
        if new:
            sourn.append(source)
    return sourn


def rm_bad(sources, ppxf=False):
    sourn = []
    for source in sources:
        sn = source["sn50"]
        gr = source["grade"]
        cm = str(source["comment"])
        ha = source["Ha"]
        if "cont_diff" in source.keys():
            cont = source["cont_diff"]
        else:
            cont = None
        if (
            sn is not None
            and sn > 0
            and gr is not None
            and gr > 2
            and "star" not in cm
            and "Star" not in cm
            # and (ha is None or ha < 80)
            and (cont is None or cont < 10 or not ppxf)
        ):
            sourn.append(source)
    return sourn


quasars = [
    ["gdn-fujimoto-v", "4762_33609.spec.fits"],
    ["gdn-fujimoto-v", "4762_37393.spec.fits"],
    ["j0226-wang-v", "3325_6699.spec.fits"],
    ["j0910-wang-v", "2028_12910.spec.fits"],
    ["jades-gdn2-v", "1181_954.spec.fits"],
    ["rubies-egs61-v", "4233_55604.spec.fits"],
    ["rubies-egs63-v", "4233_42803.spec.fits"],
    ["rubies-egs63-v", " 4233_49140.spec.fits"],
]


def rm_quasars(sources, quasars=quasars):
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
    sourn = []
    for source in sources:
        if check_zrange(source, rang, z_shift=z_shift):
            sourn.append(source)
    return sourn


def filter_zranges(sources, ranges, z_shift=True):
    for rang in ranges:
        sources = filter_zrange(sources, rang, z_shift=z_shift)
    return sources


def value_range(sources, value, rang):
    sourn = []
    for source in sources:
        try:
            if rang[0] < source[value] < rang[1]:
                sourn.append(source)
        except:
            print(f"Value {value} not found for source {source['srcid']}.")
    return sourn


def value_bins(sources, value, **kwargs):
    values = []
    sbins = []
    for source in sources:
        try:
            v = source[value]
            if v is not None:
                values.append(v)
        except:
            print(f"\r Value {value} not found for source {source['srcid']}.", end="")
    hist, bins = np.histogram(values, **kwargs)
    for i in range(len(bins) - 1):
        sbins.append(value_range(sources, value, (bins[i], bins[i + 1])))
    return hist, bins, sbins


def inbins(sources, value, nbin=10):
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
