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
    """Assumes that same "srcid" are assigned only to same sources within catalogs. Otherwise also checks by coordinate matching"""
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
    copynam = ["phot_mass", "z"]
    for k, v in s_from.items():
        if (k[:4] == "rec_" or k in copynam) and s_to.get(k) is None:
            s_to[k] = v
    return s_to


def rm_bad(sources, ppxf=False, agn=False):
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


def getffm():
    f = fetch_json("../catalog_v4.json")["sources"]
    ff = rm_bad(f)
    ffm = [s for s in ff if s["grat"][0] == "g"]
    return ffm
