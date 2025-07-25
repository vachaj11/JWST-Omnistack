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
    '''Assumes that same "srcid" are assigned only to same sources, which might not hold across catalogues.
    '''
    sourn = []
    for source in sources:
        new = True
        sn = source["sn50"]
        for i in range(len(sourn)):
            s = sourn[i]
            if source["srcid"] == s["srcid"]:
                if sn is not None and s["sn50"] < sn:
                    sourn[i] = source
                new = False
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
            and gr > 1
            and "star" not in cm
            and "Star" not in cm
            # and (ha is None or ha < 80)
            and (cont is None or cont < 10 or not ppxf)
        ):
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
