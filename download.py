import csv
import os
import time
import warnings

warnings.filterwarnings("ignore")
import urllib.request
from multiprocessing import Manager, Process, cpu_count

"""
import grizli
from grizli import utils
"""

import catalog
import spectr

typeint = ["ndup", "uid", "srcid", "nGr", "grade", "nRef"]
typefloat = [
    "ra",
    "dec",
    "zfit",
    "z",
    "sn50",
    "wmin",
    "wmax",
    "Lya",
    "Ha",
    "OIII",
    "L_Ha",
    "L_OIII",
]
typedel = ["HST", "NIRCam", "slit", "FITS", "Fnu", "Flam"]

typeint_c = {
    "srcid": "srcid",
    "grade": "grade",
}
typefloat_c = {
    "ra": "ra",
    "dec": "dec",
    "zfit": "z_best",
    "z": "zgrade",
    "sn50": "sn50",
    "flux50": "flux50",
    "err50": "err50",
    "wmin": "wmin",
    "wmax": "wmax",
    "Lya": "line_lya",
    "SII": "line_sii",
    "Ha": "line_ha",
    "OII": "line_oii",
    "OIII": "line_oiii",
    "SIII_63": "line_siii_6314",
    "SIII_90": "line_siii_9068",
    "NII_65": "line_nii_6549",
    "NII_66": "line_nii_6584",
    "phot_Av": "phot_Av",
    "phot_mass": "phot_mass",
    "phot_restU": "phot_restU",
    "phot_restV": "phot_restV",
    "phot_restJ": "phot_restJ",
    "z_phot": "z_phot",
    "phot_LHa": "phot_LHa",
    "phot_LOIII": "phot_LOIII",
    "phot_LOII": "phot_LOII",
}
typestr_c = {
    "file": "file",
    "root": "root",
    "comment": "comment",
    "grat": "grating",
}


def construct_dict(path):
    fil = open(path, "r")
    cs = csv.DictReader(fil, delimiter=",", quotechar='"')
    gall = []
    for row in cs:
        gal = dict(row)
        for k in typedel:
            if k in gal.keys():
                gal.pop(k)
        for k in gal:
            if gal[k] == "":
                gal[k] = None
            elif k in typeint:
                gal[k] = int(gal[k])
            elif k in typefloat:
                gal[k] = float(gal[k])
        gall.append(gal)
    return gall


def construct_dict_n(path):
    fil = open(path, "r")
    cs = csv.DictReader(fil, delimiter=",", quotechar='"')
    gall = []
    for row in cs:
        gal = dict(row)
        galn = dict()
        for k, v in typeint_c.items():
            if gal[v] != "":
                galn[k] = int(gal[v])
            else:
                galn[k] = None
        for k, v in typefloat_c.items():
            if gal[v] != "":
                galn[k] = float(gal[v])
            else:
                galn[k] = None
        for k, v in typestr_c.items():
            if gal[v] != "":
                galn[k] = gal[v]
            else:
                galn[k] = None
        gall.append(galn)
    return gall


'''
def download_spectra():
    BASE_URL = "https://s3.amazonaws.com/msaexp-nirspec/extractions/"
    PATH_TO_FILE = BASE_URL + "{root}/{file}"

    nrs = utils.read_catalog("../catalog.csv")

    print("By grade:")
    un = utils.Unique(nrs["grade"])

    print("By source:")
    root = utils.Unique(nrs["root"])

    print("By SNR:")
    print("N\t value")
    print("=\t =====")
    data = nrs["sn50"].data
    rang = [(0, 0.5), (0.5, 1), (1, 3), (3, 8), (8, 20), (20, 10**6)]
    dic = {(-(10**6), 0): 0}
    for i in rang:
        dic[i] = 0
    for i in data:
        t = False
        for r in rang:
            if r[0] < i < r[1]:
                dic[r] += 1
                t = True
                break
        if not t:
            dic[(-(10**6), 0)] += 1
    for v in dic:
        print(f"{dic[v]}\t {v}")

    ind = 1

    """
    try:
        inp1 = int(input("From entry: "))
    except:
        inp1 = None
    try:
        inp2 = int(input("To entry: "))
    except:
        inp2 = None
    for row in nrs[inp1:inp2]:
        print(f'\r\033[K ({ind} out of {inp2-inp1}) File: {row["file"]}', end=" ")
        url = PATH_TO_FILE.format(**row)
        pathl = "/home/vachaj11/Documents/MPE/2025/Fits/" + str(row["root"] + "/")
        if not os.path.exists(pathl):
            os.makedirs(pathl)
        urllib.request.urlretrieve(url, filename=pathl + str(row["file"]))
        ind += 1
    """
'''


def downloadall(sources, start_in=0):
    a = sources
    proc = cpu_count()
    active = []
    latest = start_in - 1
    while latest < len(a) - 1 or len(active) > 0:
        pr_no = min(proc, len(a) - 1 - latest)
        if len(active) < pr_no:
            for i in range(pr_no - len(active)):
                latest += 1
                args = (a[latest],)
                t = Process(target=download, args=args)
                t.start()
                print(f"\rRunning spectra {latest} \tout of {len(a)-1}", end="")
                active.append(t)
        for t in active:
            if not t.is_alive():
                t.terminate()
                active.remove(t)
        time.sleep(0.5)


def download(source):
    sp = spectr.get_spectrum(
        source, base="https://s3.amazonaws.com/msaexp-nirspec/extractions/"
    )
    spectr.save_npy(source, sp, base="../Data/Npy/")


def move_around():
    a = catalog.fetch_json("../catalog.json")["sources"]
    for s in a:
        spectrum = spectr.get_spectrum_n(s)
        base = "../Data/Npy/"
        spectr.save_npy(s, spectrum, base=base)
        spectr.rm_npy(s)


def add_photometry(sources, pathp):
    dic = catalog.construct_dict(pathp)
    print("here")
    values = [
        "phot_Av",
        "phot_mass",
        "phot_restU",
        "phot_restV",
        "phot_restJ",
        "z_phot",
        "phot_LHa",
        "phot_LOIII",
        "phot_LOII",
    ]
    for i, source in enumerate(sources):
        sid = source["file"]
        g = None
        for s in dic:
            if sid == s["file"].replace("-v4_", "-v3_"):
                g = s
                break
        if g is not None:
            for v in values:
                source[v] = g[v]
        else:
            for v in values:
                source[v] = None
        print(f"\r{i} {type(g)}", end="")
    return sources
